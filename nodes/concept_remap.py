"""
Concept Remap — Post-conditioning concept direction nudging.

Takes already-encoded conditioning and shifts concept vectors in
embedding space.  E.g. "water → fire" finds where "water" is
influential in the conditioning and nudges those regions toward
"fire", including all contextual bleed from attention (reflections,
color, mood, etc.).

Two modes for locating concept influence:

  Cosine (default) — Encodes the source word separately and uses
      cosine similarity to estimate where the concept lives.  Fast,
      approximate, doesn't need the original prompt text.

  Differential — If the original prompt text is provided, encodes
      the prompt twice (with and without the source word) and uses
      the actual measured difference as the influence map.  Much more
      precise since it captures the real attention-blended influence
      pattern for this specific prompt.
"""

import re
import torch
import logging

logger = logging.getLogger("CCN.ConceptRemap")


# ---------------------------------------------------------------------------
#  Remapping parser (shared with token_remap.py)
# ---------------------------------------------------------------------------

def parse_remappings(text):
    """
    Parse a multiline remapping string.
    Supports formats:
        water -> fire
        water => fire
        water : fire
        water, fire
    Lines starting with # are comments. Blank lines are skipped.
    Returns list of (source, target) tuples.
    """
    pairs = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for pattern in [r'\s*->\s*', r'\s*=>\s*', r'\s*:\s*', r'\s*,\s*']:
            parts = re.split(pattern, line, maxsplit=1)
            if len(parts) == 2:
                source, target = parts[0].strip(), parts[1].strip()
                if source and target:
                    pairs.append((source, target))
                break
        else:
            logger.warning(
                f"ConceptRemap: Could not parse line: '{line}'"
            )
    return pairs


# ---------------------------------------------------------------------------
#  Helpers — cosine similarity mode
# ---------------------------------------------------------------------------

def _encode_concept(clip, word):
    """
    Encode a word/phrase through the full CLIP pipeline and return a
    single concept vector (averaged across content token positions,
    excluding BOS/EOS padding).

    Returns tensor of shape (embed_dim,).
    """
    tokens = clip.tokenize(word)
    output = clip.encode_from_tokens(
        tokens, return_pooled=True, return_dict=True
    )
    cond = output["cond"]  # (1, seq_len, embed_dim)

    # Figure out how many content tokens there are (exclude BOS/EOS)
    key = next(iter(tokens))
    content_count = 0
    for chunk in tokens[key]:
        if len(chunk) >= 2:
            bos_id = chunk[0][0]
            eos_id = chunk[-1][0]
            for token_id, _ in chunk[1:]:
                if token_id == eos_id:
                    break
                content_count += 1

    if content_count == 0:
        content_count = 1

    concept_vec = cond[0, 1:1 + content_count, :].mean(dim=0)
    return concept_vec


def _cosine_similarity_per_position(conditioning, reference):
    """
    Compute cosine similarity between each position in the conditioning
    sequence and a reference vector.

    Args:
        conditioning: (batch, seq_len, embed_dim)
        reference: (embed_dim,)

    Returns: (batch, seq_len) similarity scores in [-1, 1]
    """
    ref = reference.unsqueeze(0).unsqueeze(0)  # (1, 1, embed_dim)
    ref = ref.to(device=conditioning.device, dtype=conditioning.dtype)

    cos_sim = torch.nn.functional.cosine_similarity(
        conditioning, ref, dim=-1
    )

    return cos_sim


# ---------------------------------------------------------------------------
#  Helpers — differential mode
# ---------------------------------------------------------------------------

def _encode_full(clip, text):
    """Encode a prompt and return the conditioning tensor."""
    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens(
        tokens, return_pooled=True, return_dict=True
    )
    return output["cond"]  # (1, seq_len, embed_dim)


def _remove_word(text, word):
    """
    Remove a word/phrase from text using word boundaries.
    Cleans up leftover punctuation artifacts (double commas, etc.).
    """
    result = re.sub(
        rf'\b{re.escape(word)}\b',
        '',
        text,
        flags=re.IGNORECASE,
    )
    # Clean up artifacts: double commas, leading/trailing commas, extra spaces
    result = re.sub(r',\s*,', ',', result)
    result = re.sub(r'(^\s*,\s*|\s*,\s*$)', '', result)
    result = re.sub(r'\s{2,}', ' ', result)
    return result.strip()


def _differential_weights(clip, prompt_text, source_word, target_shape, debug=False):
    """
    Compute per-position influence weights by encoding the prompt with
    and without the source word and measuring the difference.

    Args:
        clip: CLIP model
        prompt_text: original prompt string
        source_word: word to measure influence of
        target_shape: (batch, seq_len, embed_dim) — shape to match
        debug: print diagnostics

    Returns:
        weights: (batch, seq_len) in [0, 1], or None if differential
                 encoding fails (word not in prompt, shapes mismatch, etc.)
    """
    # Check the source word is actually in the prompt
    if not re.search(rf'\b{re.escape(source_word)}\b', prompt_text, re.IGNORECASE):
        if debug:
            print(
                f"[ConceptRemap] Differential: '{source_word}' not found "
                f"in prompt, falling back to cosine mode for this pair"
            )
        return None

    # Encode with and without the source word
    reduced_text = _remove_word(prompt_text, source_word)

    if not reduced_text.strip():
        # Entire prompt was just this word — every position is influenced
        if debug:
            print(
                f"[ConceptRemap] Differential: prompt is only "
                f"'{source_word}', using uniform weights"
            )
        return torch.ones(
            target_shape[0], target_shape[1],
            dtype=torch.float32
        )

    full_cond = _encode_full(clip, prompt_text)
    reduced_cond = _encode_full(clip, reduced_text)

    # Shape check — if token counts differ the sequences won't align
    if full_cond.shape != reduced_cond.shape:
        if debug:
            print(
                f"[ConceptRemap] Differential: shape mismatch "
                f"({full_cond.shape} vs {reduced_cond.shape}), "
                f"falling back to cosine mode for this pair"
            )
        return None

    # Also check against the target conditioning shape
    if full_cond.shape[1] != target_shape[1]:
        if debug:
            print(
                f"[ConceptRemap] Differential: sequence length mismatch "
                f"with target conditioning ({full_cond.shape[1]} vs "
                f"{target_shape[1]}), falling back to cosine mode"
            )
        return None

    # Per-position L2 difference = how much this position changed
    # when the source word was removed
    diff = (full_cond - reduced_cond).norm(dim=-1)  # (1, seq_len)

    # Normalize to [0, 1] by dividing by the max
    max_diff = diff.max()
    if max_diff > 0:
        weights = diff / max_diff
    else:
        weights = torch.zeros_like(diff)

    return weights


# ---------------------------------------------------------------------------
#  Node
# ---------------------------------------------------------------------------

class ConceptRemap:
    """
    Nudge existing conditioning in concept-direction space.

    Takes already-encoded conditioning and shifts it along the direction
    from source to target concepts.  The shift is strongest at positions
    where the source concept has the most influence.

    Two modes for locating concept influence:

    Cosine mode (default) — Encodes the source word separately and uses
        cosine similarity to estimate where the concept lives.  Fast
        and approximate.

    Differential mode — When prompt_text is provided, encodes the
        prompt with and without the source word to measure the actual
        per-position influence.  Falls back to cosine for any pair
        where differential fails (word not in prompt, shape mismatch).

    Parameters
    ----------
    blend : float
        Overall strength of the remapping effect.
    sharpness : float
        Controls how selectively the effect targets matching positions.
        Higher values concentrate the effect on positions most similar
        to the source concept.  Lower values spread it more broadly.
        At 0, every position gets nudged equally (global shift).
        Negative values invert the targeting: everything EXCEPT the
        source concept gets pushed, leaving matching positions alone.
    threshold : float
        Minimum influence weight for a position to be affected at all.
        Positions below this get zero effect.
    prompt_text : str, optional
        The original prompt text.  When provided, enables differential
        mode for much more precise concept targeting.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "clip": ("CLIP",),
                "remappings": ("STRING", {
                    "multiline": True,
                    "default": (
                        "# source -> target\n"
                        "# water -> fire\n"
                        "# calm -> chaotic"
                    ),
                    "dynamicPrompts": False,
                }),
                "blend": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": (
                        "Overall strength of the concept shift. "
                        "1.0 = full direction vector.  >1 overshoots.  "
                        "Negative = push away from target."
                    ),
                }),
                "sharpness": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": (
                        "How selectively the effect targets matching "
                        "positions.  0 = uniform shift everywhere.  "
                        "Higher = concentrated on matching positions.  "
                        "Negative = inverted: affects everything EXCEPT "
                        "the source concept."
                    ),
                }),
                "threshold": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": (
                        "Minimum influence weight for a position to be "
                        "affected.  0 = everything eligible.  Higher = "
                        "more selective."
                    ),
                }),
            },
            "optional": {
                "prompt_text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "tooltip": (
                        "The original prompt text.  When provided, "
                        "enables differential mode for much more "
                        "precise concept targeting.  Leave disconnected "
                        "for cosine mode."
                    ),
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print per-concept diagnostics to console",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "prompt_text_out")
    FUNCTION = "remap"
    CATEGORY = "CCN/conditioning"
    DESCRIPTION = (
        "Nudge existing conditioning in concept-direction space.  "
        "E.g. 'water → fire' shifts all water-influenced regions "
        "toward fire, including contextual bleed from attention.  "
        "Connect prompt_text for precise differential mode, or "
        "leave disconnected for approximate cosine mode."
    )

    def remap(
        self,
        conditioning,
        clip,
        remappings,
        blend,
        sharpness=1.0,
        threshold=0.0,
        prompt_text=None,
        debug=False,
    ):
        pairs = parse_remappings(remappings)

        text_out = prompt_text if prompt_text is not None else ""

        if not pairs or blend == 0.0:
            return (conditioning, text_out)

        use_differential = prompt_text is not None and prompt_text.strip()

        if debug:
            mode = "differential" if use_differential else "cosine"
            print(f"[ConceptRemap] Mode: {mode} | {len(pairs)} remapping(s)")

        # Precompute direction vectors and source reference embeddings
        directions = []
        for source, target in pairs:
            src_vec = _encode_concept(clip, source)
            tgt_vec = _encode_concept(clip, target)
            direction = tgt_vec - src_vec
            directions.append((source, target, src_vec, direction))

        # Process each conditioning entry
        out = []
        for cond_tensor, cond_dict in conditioning:
            modified = cond_tensor.clone()

            for source, target, src_vec, direction in directions:
                dir_vec = direction.to(
                    device=modified.device, dtype=modified.dtype
                )

                # ---- Determine per-position weights ----

                diff_weights = None
                if use_differential:
                    diff_weights = _differential_weights(
                        clip, prompt_text, source,
                        modified.shape, debug=debug,
                    )
                    if diff_weights is not None:
                        diff_weights = diff_weights.to(
                            device=modified.device, dtype=modified.dtype
                        )

                if diff_weights is not None:
                    # Differential mode — weights are already [0, 1]
                    raw_weights = diff_weights

                    if debug:
                        nonzero = raw_weights[raw_weights > 0]
                        if nonzero.numel() > 0:
                            print(
                                f"[ConceptRemap] Differential: "
                                f"'{source}' → '{target}' "
                                f"| nonzero: "
                                f"{(raw_weights > 0).sum().item():.0f}"
                                f"/{raw_weights.numel()} positions "
                                f"| max: {raw_weights.max().item():.4f} "
                                f"| mean(nonzero): "
                                f"{nonzero.mean().item():.4f}"
                            )
                        else:
                            print(
                                f"[ConceptRemap] Differential: "
                                f"'{source}' → '{target}' | all zero"
                            )
                else:
                    # Cosine mode — compute similarity
                    sim = _cosine_similarity_per_position(modified, src_vec)

                    if threshold > 0:
                        sim = sim * (sim >= threshold).float()

                    raw_weights = sim

                # ---- Apply sharpness ----

                invert = sharpness < 0
                abs_sharpness = abs(sharpness)

                if abs_sharpness == 0:
                    weights = torch.ones_like(raw_weights)
                elif abs_sharpness != 1.0:
                    weights = (
                        raw_weights.sign()
                        * raw_weights.abs().pow(abs_sharpness)
                    )
                else:
                    weights = raw_weights

                weights = weights.clamp(min=0.0, max=1.0)

                # Apply threshold for differential mode
                if diff_weights is not None and threshold > 0:
                    weights = weights * (weights >= threshold).float()

                if invert:
                    weights = 1.0 - weights

                # ---- Apply nudge ----

                nudge = weights.unsqueeze(-1) * dir_vec * blend
                modified = modified + nudge

                if debug:
                    pos_mask = weights > 0
                    if pos_mask.any():
                        affected = weights[pos_mask]
                        print(
                            f"[ConceptRemap] Final: "
                            f"'{source}' → '{target}' "
                            f"| affected: {pos_mask.sum().item():.0f}"
                            f"/{weights.numel()} positions "
                            f"| max_w: {affected.max().item():.4f} "
                            f"| mean_w: {affected.mean().item():.4f} "
                            f"| dir_norm: {dir_vec.norm().item():.4f}"
                        )
                    else:
                        print(
                            f"[ConceptRemap] Final: "
                            f"'{source}' → '{target}' "
                            f"| NO positions affected"
                        )

            out.append([modified, cond_dict])

        return (out, text_out)
