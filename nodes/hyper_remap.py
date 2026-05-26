"""
HyperRemap — Unified four-phase prompt remapping pipeline.

Combines string replacement, token-level embedding blending,
concept-direction nudging, and full-sequence residual delta into a
single node. Each phase cascades its results to the next:

  Phase 1 (String Replace):  comma-separated pairs modify the prompt
           text directly.  "red, blue" replaces "red" with "blue".

  Phase 2 (Token Remap):  arrow pairs (source -> target) encode the
           prompt twice and blend embeddings at changed positions.
           The text is NOT modified — this is embedding-space only.

  Phase 3 (Concept Remap):  fat-arrow pairs (source => target) nudge
           the conditioning along concept-direction vectors.  Position
           weights are derived from cosine similarity between each
           incoming conditioning position and the source concept vector.

  Phase 4 (Delta Remap):  double-tilde pairs (base ~~ subtracted) encode
           both prompts as full sequences, subtract them to produce a
           residual delta, and add it to the incoming conditioning.
           Position weights are the product of two independent layers:
             - Intrinsic (sx/tx): based on per-position delta L2 norm —
               how strongly each position contributes to the difference
               between the two encoded prompts.  Defaults: sx=1.0, tx=0.0.
             - Incoming-conditioning (s/t): based on cosine similarity
               between each incoming conditioning position and the pooled
               output of the base encoding — how much each position in the
               main prompt relates to the base concept.  Uses global
               sharpness/threshold widgets; overridable with (s:, t:).

Separator syntax:
  source, target       — string replacement (text-level)
  source -> target     — token remap (embedding-level)
  source => target     — concept remap (embedding-level)
  base ~~ subtracted   — delta remap (embedding-level, additive residual)

Per-pair overrides (optional, appended to ->, =>, ~~ entries):
  ship -> starship (1.5)                     — blend override
  ship -> starship (b:1.5)                   — same, explicit key
  water => fire (b:0.8, s:2.0)              — blend + sharpness
  water => fire (b:0.8, s:2.0, t:0.1)       — blend + sharpness + threshold
  bee in wild ~~ insect (b:0.5, s:1.0)      — incoming-conditioning override
  bee in wild ~~ insect (sx:2.0, tx:0.1)    — intrinsic delta override
  bee in wild ~~ insect (b:0.5, s:1.0, sx:2.0, tx:0.1)  — all overrides

  b  = blend (all operators)
  s  = sharpness against incoming conditioning (=> and ~~ only)
  t  = threshold against incoming conditioning (=> and ~~ only)
  sx = intrinsic delta sharpness (~~ only, default 1.0)
  tx = intrinsic delta threshold (~~ only, default 0.0)
  Bare number is always blend. Unspecified params use node globals / internal defaults.

Entries are delimited by semicolons or newlines.  Lines starting
with # are comments.
"""

import re
import torch
import logging

from .token_remap import apply_text_remappings
from .concept_remap import (
    _encode_concept,
    _cosine_similarity_per_position,
    _differential_weights,
)

logger = logging.getLogger("CCN.HyperRemap")


# ---------------------------------------------------------------------------
#  Override parsing — extracts (param) annotations from end of line
# ---------------------------------------------------------------------------

# Matches a trailing parenthetical like (1.5) or (b:0.8, s:2.0)
_OVERRIDE_RE = re.compile(r'\(([^)]+)\)\s*$')

# Matches a named param like "b:1.5", "s: 2.0", "sx:1.0", "tx:0.1"
_NAMED_PARAM_RE = re.compile(r'^(sx|tx|[bst])\s*:\s*(-?[0-9.]+)$')


def _parse_overrides(line):
    """
    Extract optional per-pair overrides from the end of a line.

    Returns (clean_line, overrides_dict). The dict may contain keys:
      'b'  — blend (all operators)
      's'  — incoming-conditioning sharpness (=> and ~~ only)
      't'  — incoming-conditioning threshold (=> and ~~ only)
      'sx' — intrinsic delta sharpness (~~ only)
      'tx' — intrinsic delta threshold (~~ only)
    Missing keys mean "use global / internal default".
    """
    match = _OVERRIDE_RE.search(line)
    if not match:
        return line, {}

    clean_line = line[:match.start()].rstrip()
    raw = match.group(1).strip()
    overrides = {}

    # Try bare number first (= blend override)
    try:
        overrides['b'] = float(raw)
        return clean_line, overrides
    except ValueError:
        pass

    # Parse named params: "b:0.8, s:2.0, t:0.1"
    for segment in raw.split(','):
        segment = segment.strip()
        param_match = _NAMED_PARAM_RE.match(segment)
        if param_match:
            key = param_match.group(1)
            try:
                overrides[key] = float(param_match.group(2))
            except ValueError:
                logger.warning(f"HyperRemap: Invalid override value in '{segment}'")
        else:
            logger.warning(f"HyperRemap: Could not parse override '{segment}'")

    return clean_line, overrides


# ---------------------------------------------------------------------------
#  Parsing — routes entries to the correct phase by separator type
# ---------------------------------------------------------------------------

def _parse_hyper_entries(text):
    """
    Parse a remapping string into four buckets based on separator.

    Splits on semicolons and newlines, then inspects each entry:
      ~~  routes to delta_pairs   (base, subtracted, overrides)
      ->  routes to token_pairs   (source, target, overrides)
      =>  routes to concept_pairs (source, target, overrides)
      ,   routes to string_pairs  (source, target)

    Token, concept, and delta pairs include an overrides dict parsed from
    optional trailing parentheticals. String pairs have no overrides.

    ~~ is checked first to avoid any ambiguity with -> or =>.
    """
    string_pairs = []
    token_pairs = []
    concept_pairs = []
    delta_pairs = []

    normalized = text.replace(";", "\n")

    for line in normalized.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Check separators in priority order: ~~ then -> then => then ,
        if "~~" in line:
            clean_line, overrides = _parse_overrides(line)
            parts = re.split(r'\s*~~\s*', clean_line, maxsplit=1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                delta_pairs.append((parts[0].strip(), parts[1].strip(), overrides))
            else:
                logger.warning(f"HyperRemap: Could not parse delta remap entry: '{line}'")
        elif "->" in line:
            clean_line, overrides = _parse_overrides(line)
            parts = re.split(r'\s*->\s*', clean_line, maxsplit=1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                for key in ('s', 't', 'sx', 'tx'):
                    if key in overrides:
                        label = {'s': 's (sharpness)', 't': 't (threshold)',
                                 'sx': 'sx (intrinsic sharpness)', 'tx': 'tx (intrinsic threshold)'}[key]
                        logger.warning(
                            f"HyperRemap: Override {label} is ignored "
                            f"on token remap entry: '{line}'"
                        )
                token_pairs.append((parts[0].strip(), parts[1].strip(), overrides))
            else:
                logger.warning(f"HyperRemap: Could not parse token remap entry: '{line}'")
        elif "=>" in line:
            clean_line, overrides = _parse_overrides(line)
            parts = re.split(r'\s*=>\s*', clean_line, maxsplit=1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                concept_pairs.append((parts[0].strip(), parts[1].strip(), overrides))
            else:
                logger.warning(f"HyperRemap: Could not parse concept remap entry: '{line}'")
        elif "," in line:
            parts = line.split(",", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                string_pairs.append((parts[0].strip(), parts[1].strip()))
            else:
                logger.warning(f"HyperRemap: Could not parse string replacement entry: '{line}'")
        else:
            logger.warning(f"HyperRemap: Ambiguous entry (no recognized separator): '{line}'")

    return string_pairs, token_pairs, concept_pairs, delta_pairs


# ---------------------------------------------------------------------------
#  Phase 1 — String replacement (modifies text)
# ---------------------------------------------------------------------------

def _apply_string_replacements(text, pairs, case_sensitive=True):
    """
    Apply find/replace pairs to the text. Returns (modified_text, count).
    """
    total = 0
    result = text
    for find_str, replace_str in pairs:
        if case_sensitive:
            count = result.count(find_str)
            result = result.replace(find_str, replace_str)
        else:
            pattern = re.compile(re.escape(find_str), re.IGNORECASE)
            count = len(pattern.findall(result))
            result = pattern.sub(replace_str, result)
        total += count
    return result, total


# ---------------------------------------------------------------------------
#  Phase 2 — Token remap (embedding blend, text unchanged)
# ---------------------------------------------------------------------------

def _build_change_mask(orig_tokens, remap_tokens, cond_shape):
    """
    Compare two token dicts and return a boolean mask of which
    sequence positions differ. Returns None if comparison fails.
    """
    try:
        key = next(iter(orig_tokens))
        if key not in remap_tokens:
            return None

        orig_ids = [tid for chunk in orig_tokens[key] for tid, _w in chunk]
        remap_ids = [tid for chunk in remap_tokens[key] for tid, _w in chunk]

        if len(orig_ids) != len(remap_ids):
            return None

        seq_len = cond_shape[1]
        mask = torch.zeros(1, seq_len, dtype=torch.bool)
        for i in range(min(len(orig_ids), seq_len)):
            if orig_ids[i] != remap_ids[i]:
                mask[0, i] = True
        return mask

    except (StopIteration, IndexError, KeyError):
        return None


def _apply_token_remap(clip, text, pairs, global_blend, debug=False):
    """
    Encode text with per-pair token-level embedding blending.
    Each pair is processed individually against the base encoding
    so per-pair blend overrides are respected.

    Returns (blended_cond, pooled_dict).
    """
    orig_tokens = clip.tokenize(text)
    orig_output = clip.encode_from_tokens(
        orig_tokens, return_pooled=True, return_dict=True,
    )
    orig_cond = orig_output.pop("cond")

    if not pairs:
        return orig_cond, orig_output

    working_cond = orig_cond.clone()

    for source, target, overrides in pairs:
        blend = overrides.get('b', global_blend)

        remapped_text = apply_text_remappings(text, [(source, target)])
        if remapped_text == text:
            if debug:
                print(f"[HyperRemap] Token remap: '{source}' not found in prompt")
            continue

        remap_tokens = clip.tokenize(remapped_text)
        remap_output = clip.encode_from_tokens(
            remap_tokens, return_pooled=True, return_dict=True,
        )
        remap_cond = remap_output.pop("cond")

        if orig_cond.shape == remap_cond.shape:
            changed_mask = _build_change_mask(orig_tokens, remap_tokens, orig_cond.shape)

            if changed_mask is not None:
                if debug:
                    override_note = " [override]" if 'b' in overrides else ""
                    print(
                        f"[HyperRemap] Token remap: '{source}' -> '{target}' "
                        f"precise blend at "
                        f"{changed_mask.sum().item():.0f}/{changed_mask.numel()} "
                        f"positions (b={blend:.3f}){override_note}"
                    )
                blended_positions = torch.lerp(orig_cond, remap_cond, blend)
                working_cond = torch.where(
                    changed_mask.unsqueeze(-1).expand_as(working_cond),
                    blended_positions,
                    working_cond,
                )
            else:
                if debug:
                    print(
                        f"[HyperRemap] Token remap: '{source}' -> '{target}' "
                        f"global blend (b={blend:.3f})"
                    )
                working_cond = torch.lerp(orig_cond, remap_cond, blend)
        else:
            if debug:
                print(
                    f"[HyperRemap] Token remap: '{source}' -> '{target}' "
                    f"shape mismatch, global blend (b={blend:.3f})"
                )
            min_seq = min(orig_cond.shape[1], remap_cond.shape[1])
            working_cond = torch.lerp(
                orig_cond[:, :min_seq, :],
                remap_cond[:, :min_seq, :],
                blend,
            )

    return working_cond, orig_output


# ---------------------------------------------------------------------------
#  Phase 3 — Concept remap (embedding-direction nudge)
# ---------------------------------------------------------------------------

def _apply_concept_remap(
    cond_tensor, clip, pairs, global_blend, global_sharpness, global_threshold,
    prompt_text=None, debug=False,
):
    """
    Nudge a conditioning tensor along concept-direction vectors.
    Per-pair overrides for blend (b), sharpness (s), and threshold (t)
    are pulled from each pair's overrides dict, falling back to globals.
    """
    if not pairs:
        return cond_tensor

    use_differential = prompt_text is not None and prompt_text.strip()

    if debug:
        mode = "differential" if use_differential else "cosine"
        print(f"[HyperRemap] Concept remap: {mode} mode, {len(pairs)} pair(s)")

    # Precompute direction vectors
    directions = []
    for source, target, overrides in pairs:
        src_vec = _encode_concept(clip, source)
        tgt_vec = _encode_concept(clip, target)
        directions.append((source, target, overrides, src_vec, tgt_vec - src_vec))

    modified = cond_tensor.clone()

    for source, target, overrides, src_vec, direction in directions:
        blend = overrides.get('b', global_blend)
        sharpness = overrides.get('s', global_sharpness)
        threshold = overrides.get('t', global_threshold)

        dir_vec = direction.to(device=modified.device, dtype=modified.dtype)

        # Determine per-position influence weights
        diff_weights = None
        if use_differential:
            diff_weights = _differential_weights(
                clip, prompt_text, source,
                modified.shape, debug=debug,
            )
            if diff_weights is not None:
                diff_weights = diff_weights.to(
                    device=modified.device, dtype=modified.dtype,
                )

        if diff_weights is not None:
            raw_weights = diff_weights
            if debug:
                nonzero = raw_weights[raw_weights > 0]
                if nonzero.numel() > 0:
                    print(
                        f"[HyperRemap] Concept differential: "
                        f"'{source}' => '{target}' "
                        f"| nonzero: {(raw_weights > 0).sum().item():.0f}"
                        f"/{raw_weights.numel()} "
                        f"| max: {raw_weights.max().item():.4f} "
                        f"| mean(nz): {nonzero.mean().item():.4f}"
                    )
                else:
                    print(
                        f"[HyperRemap] Concept differential: "
                        f"'{source}' => '{target}' | all zero"
                    )
        else:
            # Cosine fallback
            sim = _cosine_similarity_per_position(modified, src_vec)
            if threshold > 0:
                sim = sim * (sim >= threshold).float()
            raw_weights = sim

        # Apply sharpness
        invert = sharpness < 0
        abs_sharpness = abs(sharpness)

        if abs_sharpness == 0:
            weights = torch.ones_like(raw_weights)
        elif abs_sharpness != 1.0:
            weights = raw_weights.sign() * raw_weights.abs().pow(abs_sharpness)
        else:
            weights = raw_weights

        weights = weights.clamp(min=0.0, max=1.0)

        if invert:
            weights = 1.0 - weights

        if diff_weights is not None and threshold > 0:
            weights = weights * (weights >= threshold).float()

        # Apply directional nudge
        nudge = weights.unsqueeze(-1) * dir_vec * blend
        modified = modified + nudge

        if debug:
            override_str = ""
            if overrides:
                parts = [f"{k}={v:.3f}" for k, v in overrides.items()]
                override_str = f" | overrides: {', '.join(parts)}"

            pos_mask = weights > 0
            if pos_mask.any():
                affected = weights[pos_mask]
                print(
                    f"[HyperRemap] Concept final: "
                    f"'{source}' => '{target}' "
                    f"| affected: {pos_mask.sum().item():.0f}"
                    f"/{weights.numel()} "
                    f"| max_w: {affected.max().item():.4f} "
                    f"| mean_w: {affected.mean().item():.4f} "
                    f"| dir_norm: {dir_vec.norm().item():.4f}"
                    f"{override_str}"
                )
            else:
                print(
                    f"[HyperRemap] Concept final: "
                    f"'{source}' => '{target}' "
                    f"| NO positions affected"
                    f"{override_str}"
                )

    return modified



# ---------------------------------------------------------------------------
#  Phase 4 — Delta remap (full-sequence residual, additive)
# ---------------------------------------------------------------------------

# Internal defaults for intrinsic delta weighting — not exposed as widgets.
_DELTA_SX_DEFAULT = 1.0
_DELTA_TX_DEFAULT = 0.0


def _apply_delta_remap(
    cond_tensor, clip, pairs, global_blend, global_sharpness, global_threshold,
    normalize=True, debug=False,
):
    """
    Add a residual delta to the conditioning tensor.

    For each pair (base ~~ subtracted):

      1. Encode both prompts as full token sequences [1, seq, dim].
         Retain the base prompt's pooled output as a semantic anchor.

      2. Compute delta = base_seq - sub_seq.  Optionally L2-normalise
         the whole delta tensor so blend has a consistent magnitude
         regardless of how different the two prompts are.

      3. Build per-position weights as the product of two independent layers:

         Intrinsic layer (sx / tx overrides, internal defaults 1.0 / 0.0):
           Weight each position by its normalised delta L2 norm — how much
           this position contributes to the difference between the two
           encoded prompts.  Sharpness curves the distribution; threshold
           masks low-contribution positions.

         Incoming-conditioning layer (s / t global widgets / overrides):
           Weight each position by its cosine similarity to the pooled
           output of the base encoding — how much each position in the
           main prompt's conditioning relates to the overall meaning of
           the base prompt.  Sharpness and threshold work identically to
           concept remap's incoming-conditioning weighting.

      4. Apply: modified += combined_weights * delta * blend
    """
    if not pairs:
        return cond_tensor

    modified = cond_tensor.clone()

    if debug:
        print(f"[HyperRemap] Delta remap: {len(pairs)} pair(s), normalize={normalize}")

    for base_text, sub_text, overrides in pairs:
        blend     = overrides.get('b',  global_blend)
        sharpness = overrides.get('s',  global_sharpness)
        threshold = overrides.get('t',  global_threshold)
        sx        = overrides.get('sx', _DELTA_SX_DEFAULT)
        tx        = overrides.get('tx', _DELTA_TX_DEFAULT)

        # -- Encode base prompt, keep pooled output as similarity anchor --
        base_tokens = clip.tokenize(base_text)
        base_output = clip.encode_from_tokens(
            base_tokens, return_pooled=True, return_dict=True,
        )
        base_seq    = base_output.pop("cond")                      # [1, seq, dim]
        base_pooled = base_output.get("pooled_output", None)       # [1, dim] or None

        # -- Encode subtracted prompt --
        sub_tokens = clip.tokenize(sub_text)
        sub_output = clip.encode_from_tokens(
            sub_tokens, return_pooled=True, return_dict=True,
        )
        sub_seq = sub_output.pop("cond")                           # [1, seq, dim]

        # -- Align to incoming conditioning sequence length --
        seq_len  = modified.shape[1]
        base_seq = base_seq[:, :seq_len, :]
        sub_seq  = sub_seq[:,  :seq_len, :]

        delta = (base_seq - sub_seq).to(device=modified.device, dtype=modified.dtype)

        # -- Optional normalisation --
        if normalize:
            delta_norm = delta.norm()
            if delta_norm > 1e-8:
                delta = delta / delta_norm

        # -- Intrinsic layer: weight by per-position delta L2 norm --
        pos_norms = delta.norm(dim=-1)                             # [1, seq]
        pn_max = pos_norms.max()
        if pn_max > 1e-8:
            pos_norms = pos_norms / pn_max
        else:
            pos_norms = torch.zeros_like(pos_norms)

        invert_sx   = sx < 0
        abs_sx      = abs(sx)
        if abs_sx == 0:
            intrinsic_w = torch.ones_like(pos_norms)
        elif abs_sx != 1.0:
            intrinsic_w = pos_norms.pow(abs_sx)
        else:
            intrinsic_w = pos_norms.clone()
        intrinsic_w = intrinsic_w.clamp(0.0, 1.0)
        if invert_sx:
            intrinsic_w = 1.0 - intrinsic_w
        if tx > 0:
            intrinsic_w = intrinsic_w * (intrinsic_w >= tx).float()

        # -- Incoming-conditioning layer: cosine sim to base pooled --
        if base_pooled is not None:
            base_pooled_dev = base_pooled.to(device=modified.device, dtype=modified.dtype)
            raw_sim = _cosine_similarity_per_position(modified, base_pooled_dev)
        else:
            # No pooled output available — uniform weight
            raw_sim = torch.ones(modified.shape[:2], device=modified.device, dtype=modified.dtype)

        invert_s   = sharpness < 0
        abs_s      = abs(sharpness)
        if abs_s == 0:
            cond_w = torch.ones_like(raw_sim)
        elif abs_s != 1.0:
            cond_w = raw_sim.sign() * raw_sim.abs().pow(abs_s)
        else:
            cond_w = raw_sim.clone()
        cond_w = cond_w.clamp(0.0, 1.0)
        if invert_s:
            cond_w = 1.0 - cond_w
        if threshold > 0:
            cond_w = cond_w * (cond_w >= threshold).float()

        # -- Combine layers and apply nudge --
        weights = intrinsic_w * cond_w                             # [1, seq]
        nudge   = weights.unsqueeze(-1) * delta * blend
        modified = modified + nudge

        if debug:
            override_str = ""
            if overrides:
                override_str = " | overrides: " + ", ".join(
                    f"{k}={v:.3f}" for k, v in overrides.items()
                )
            pos_mask = weights > 0
            if pos_mask.any():
                affected = weights[pos_mask]
                print(
                    f"[HyperRemap] Delta final: "
                    f"'{base_text}' ~~ '{sub_text}' "
                    f"| affected: {pos_mask.sum().item():.0f}/{weights.numel()} "
                    f"| max_w: {affected.max().item():.4f} "
                    f"| mean_w: {affected.mean().item():.4f} "
                    f"| intrinsic max: {intrinsic_w.max().item():.4f} "
                    f"| cond sim max: {cond_w.max().item():.4f} "
                    f"| delta_norm: {delta.norm().item():.4f}"
                    f"{override_str}"
                )
            else:
                print(
                    f"[HyperRemap] Delta final: "
                    f"'{base_text}' ~~ '{sub_text}' | NO positions affected"
                    f"{override_str}"
                )

    return modified


class HyperRemap:
    """
    Unified four-phase remapping: string replace, token blend, concept nudge,
    delta residual.

    Parses a single remappings field and routes entries by separator:
      comma (,)         — text-level string replacement
      arrow (->)        — embedding-level token blending
      fat arrow (=>)    — embedding-level concept direction shift
      double tilde (~~) — embedding-level additive residual delta

    A single set of blend / sharpness / threshold globals applies as the
    default for all embedding-space operators.  Per-pair overrides in
    parentheses take precedence.  sharpness and threshold control
    incoming-conditioning similarity weighting for => and ~~; ~~ also
    supports sx / tx overrides for intrinsic delta weighting.

    Phases execute in order, each feeding its result to the next.
    Outputs both modified and untouched conditioning for comparison.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                }),
                "remappings": ("STRING", {
                    "multiline": True,
                    "default": (
                        "# string replace:  find, replace\n"
                        "# token remap:     source -> target\n"
                        "# token remap:     source -> target (0.8)\n"
                        "# concept remap:   source => target\n"
                        "# concept remap:   source => target (b:0.8, s:2.0, t:0.1)\n"
                        "# delta remap:     base ~~ subtracted\n"
                        "# delta remap:     base ~~ subtracted (b:0.5, s:1.0, sx:2.0)\n"
                    ),
                    "dynamicPrompts": False,
                }),
                "blend": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.001,
                    "tooltip": (
                        "Default blend for all operators. "
                        "For ->: lerp between original and remapped embeddings. "
                        "For => and ~~: magnitude of the nudge vector. "
                        ">1 overshoots, negative inverts direction. "
                        "Per-pair (b:X) overrides take precedence."
                    ),
                }),
                "sharpness": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": (
                        "Default incoming-conditioning sharpness for => and ~~. "
                        "Controls how sharply positions are weighted by their "
                        "cosine similarity to the source/base concept. "
                        "0 = uniform across all positions. "
                        "Higher = concentrated on most-similar positions. "
                        "Negative = favour least-similar positions. "
                        "Ignored by ->. Per-pair (s:X) overrides take precedence."
                    ),
                }),
                "threshold": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": (
                        "Default incoming-conditioning threshold for => and ~~. "
                        "Masks out positions whose similarity weight falls below "
                        "this value after sharpness is applied. "
                        "0 = all positions eligible. "
                        "Ignored by ->. Per-pair (t:X) overrides take precedence."
                    ),
                }),
            },
            "optional": {
                "normalize_delta": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "L2-normalise the delta tensor before blending for ~~ entries. "
                        "When on, blend has a consistent magnitude regardless of how "
                        "different the two prompts are. When off, larger semantic "
                        "differences produce stronger effects at the same blend value."
                    ),
                }),
                "case_sensitive": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Case sensitivity for string replacement pairs.",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print phase diagnostics to console.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = (
        "conditioning",
        "untouched_conditioning",
        "original_prompt",
        "modified_prompt",
    )
    FUNCTION = "execute"
    CATEGORY = "CCN/conditioning"
    DESCRIPTION = (
        "Unified four-phase remapping pipeline. "
        "comma (,) = text replacement. "
        "arrow (->) = token embedding blend. "
        "fat arrow (=>) = concept direction nudge. "
        "double tilde (~~) = additive residual delta. "
        "blend / sharpness / threshold are shared defaults for all operators. "
        "sharpness and threshold govern incoming-conditioning similarity weighting "
        "for => and ~~. ~~ additionally supports sx / tx overrides for intrinsic "
        "delta weighting. Per-pair (b:X, s:X, t:X, sx:X, tx:X) overrides in "
        "parentheses take precedence over globals."
    )

    def execute(
        self,
        clip,
        text,
        remappings,
        blend,
        sharpness,
        threshold,
        normalize_delta=True,
        case_sensitive=True,
        debug=False,
    ):
        original_prompt = text
        string_pairs, token_pairs, concept_pairs, delta_pairs = _parse_hyper_entries(remappings)

        if debug:
            print(
                f"[HyperRemap] Parsed: "
                f"{len(string_pairs)} string, "
                f"{len(token_pairs)} token, "
                f"{len(concept_pairs)} concept, "
                f"{len(delta_pairs)} delta"
            )

        # -- Phase 1: String replacement (modifies text) --

        modified_prompt = original_prompt
        if string_pairs:
            modified_prompt, count = _apply_string_replacements(
                modified_prompt, string_pairs, case_sensitive,
            )
            if debug:
                print(f"[HyperRemap] String replace: {count} substitution(s)")

        # -- Encode untouched conditioning from the original prompt --

        untouched_tokens = clip.tokenize(original_prompt)
        untouched_output = clip.encode_from_tokens(
            untouched_tokens, return_pooled=True, return_dict=True,
        )
        untouched_cond = untouched_output.pop("cond")
        untouched_conditioning = [[untouched_cond, untouched_output]]

        # -- Phase 2: Token remap (embedding blend on modified_prompt) --

        working_cond, working_pooled = _apply_token_remap(
            clip, modified_prompt, token_pairs, blend, debug=debug,
        )

        # -- Phase 3: Concept remap (direction nudge on working conditioning) --

        if concept_pairs:
            working_cond = _apply_concept_remap(
                working_cond,
                clip,
                concept_pairs,
                blend,
                sharpness,
                threshold,
                prompt_text=modified_prompt,
                debug=debug,
            )

        # -- Phase 4: Delta remap (additive residual from prompt pair) --

        if delta_pairs:
            working_cond = _apply_delta_remap(
                working_cond,
                clip,
                delta_pairs,
                blend,
                sharpness,
                threshold,
                normalize=normalize_delta,
                debug=debug,
            )

        modified_conditioning = [[working_cond, working_pooled]]

        return (
            modified_conditioning,
            untouched_conditioning,
            original_prompt,
            modified_prompt,
        )

