"""
HyperRemapKrea2Edit — Fused four-phase remapping with krea2_edit
grounded encoding.

Runs the HyperRemap pipeline (string replace, token blend, concept
nudge, delta residual) on conditioning produced by the krea2_edit
image-grounded encode: the edit instruction is encoded together with
the source image(s) through Qwen3-VL using the krea2_edit chat
template, then the embedding-space phases are applied to the grounded
result.

Grounded-encode specifics (vs. plain HyperRemap):

  - The base encode and each token-remap re-encode run through
    clip.tokenize(text, images=..., llama_template=...) followed by
    encode_from_tokens_scheduled, so hook-scheduled clip weights
    produce multiple conditioning entries and every entry passes
    through the remap phases.

  - Token remap (->): the token-id change mask is attempted first;
    when the image-bearing token stream defeats it, blending falls
    back to soft per-position weights derived from the normalised L2
    difference between the two grounded encodes (template and vision
    positions are identical inputs, so they diff to exactly zero).

  - Concept remap (=>): concept vectors are built template-robustly —
    mean-pool the full sequence of the word's text-only encode and
    subtract the mean-pooled empty-prompt encode, cancelling the chat
    template's static contribution.  Differential weighting is skipped
    when grounded (text-only re-encodes can never match the grounded
    sequence length), so => runs in cosine mode.

  - Delta remap (~~): aux encodes are text-only and therefore shorter
    than grounded conditioning; _apply_delta_remap's pooled mode
    collapses each delta into a direction vector broadcast across the
    incoming sequence.

With no image connected, the node behaves like HyperRemap with the
scheduled-encode and template-robust concept-vector improvements (and
no untouched output).  With an empty remappings field it reduces to a
plain grounded encode — usable directly as the negative branch.

The appearance path (Krea2EditModelPatch, source latent as RoPE
frame-1 tokens) is orthogonal and remains ComfyUI-Krea2Edit's node.
"""

import logging
import torch

import comfy.utils

from .token_remap import apply_text_remappings
from .hyper_remap import (
    _parse_hyper_entries,
    _apply_string_replacements,
    _build_change_mask,
    _apply_concept_remap,
    _apply_delta_remap,
)

logger = logging.getLogger("CCN.HyperRemapKrea2Edit")


# ---------------------------------------------------------------------------
#  Grounded encode — templates and image prep
# ---------------------------------------------------------------------------

# Copied verbatim from ComfyUI-Krea2Edit (Krea2EditGroundedEncode): the
# krea2_edit LoRA's training-matched chat templates.  Keep in sync if that
# pack's recipe changes.
KREA2_EDIT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "{}<|im_end|>\n<|im_start|>assistant\n"
)

KREA2_EDIT_TEMPLATE_2REF = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{}<|im_end|>\n<|im_start|>assistant\n"
)


def _prep_image(image, grounding_px):
    """Cap the longest side for the VLM and drop any alpha channel."""
    samples = image.movedim(-1, 1)
    h, w = samples.shape[2], samples.shape[3]
    if grounding_px and max(h, w) > grounding_px:
        s = grounding_px / max(h, w)
        samples = comfy.utils.common_upscale(
            samples, round(w * s), round(h * s), "area", "disabled"
        )
    return samples.movedim(1, -1)[:, :, :, :3]


def _grounded_tokenize(clip, text, images):
    if images is None:
        return clip.tokenize(text)
    template = KREA2_EDIT_TEMPLATE if len(images) == 1 else KREA2_EDIT_TEMPLATE_2REF
    return clip.tokenize(text, images=images, llama_template=template)


# ---------------------------------------------------------------------------
#  Template-robust concept vectors
# ---------------------------------------------------------------------------

def _sequence_mean(clip, text):
    """Mean-pool a text-only encode across its full sequence -> (dim,)."""
    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    cond = output["cond"]
    if cond.shape[1] == 0:
        # Real tokenizers always emit template/BOS/EOS tokens, but a mean
        # over zero positions would be NaN and silently poison every
        # concept vector built from it.
        logger.warning(
            f"HyperRemapKrea2Edit: encode of '{text}' returned an empty "
            f"sequence; using a zero vector"
        )
        return torch.zeros(cond.shape[-1], dtype=cond.dtype, device=cond.device)
    return cond[0].mean(dim=0)


def _make_concept_encoder(clip):
    """
    Build an encode_concept_fn for _apply_concept_remap that survives
    templated tokenizers.

    _encode_concept's BOS/EOS content slice assumes a CLIP/T5-style token
    layout; a chat template (Qwen3-VL) puts system-prompt tokens at those
    positions instead.  Mean-pooling the word's full encode and subtracting
    the mean-pooled empty-prompt encode cancels the template's static
    contribution, leaving the word's own contribution (including the
    template-suffix positions it causally shifts).

    Vectors are memoized so repeated words — and repeated application
    across scheduled conditioning entries — cost one encode each.
    """
    cache = {}
    empty_vec = _sequence_mean(clip, "")

    def encode(clip_, word):
        if word not in cache:
            cache[word] = _sequence_mean(clip_, word) - empty_vec
        return cache[word]

    return encode


# ---------------------------------------------------------------------------
#  Phase 2 — token remap against the grounded encode
# ---------------------------------------------------------------------------

def _blend_entry(
    base_cond, working_cond, remap_cond, base_tokens, remap_tokens,
    blend, source, target, has_override, debug,
):
    """
    Blend one scheduled entry's tensor toward its remapped encode.

    Precise token-id mask when the token stream supports it; soft
    per-position weights from the normalised L2 difference between the
    two encodes when it doesn't; truncated global lerp (matching
    HyperRemap's fallback, prior pair blends discarded) on sequence
    length mismatch.
    """
    override_note = " [override]" if has_override else ""

    if base_cond.shape == remap_cond.shape:
        changed_mask = _build_change_mask(base_tokens, remap_tokens, base_cond.shape)

        if changed_mask is not None:
            if debug:
                print(
                    f"[HyperRemapKrea2Edit] Token remap: '{source}' -> '{target}' "
                    f"precise blend at "
                    f"{changed_mask.sum().item():.0f}/{changed_mask.numel()} "
                    f"positions (b={blend:.3f}){override_note}"
                )
            blended = torch.lerp(base_cond, remap_cond, blend)
            return torch.where(
                changed_mask.to(working_cond.device)
                .unsqueeze(-1)
                .expand_as(working_cond),
                blended,
                working_cond,
            )

        # Soft fallback: identical inputs (template, vision blocks, prefix
        # text) diff to exactly zero, so the normalised difference is a
        # soft change mask that also captures causal attention bleed.
        diff = (remap_cond - base_cond).norm(dim=-1)               # [B, seq]
        diff_max = diff.max()
        if diff_max <= 1e-6:
            logger.warning(
                f"HyperRemapKrea2Edit: Token remap '{source}' -> '{target}' "
                f"produced no measurable difference in the encode; skipped"
            )
            return working_cond
        if debug:
            changed = (diff > 1e-6).sum().item()
            print(
                f"[HyperRemapKrea2Edit] Token remap: '{source}' -> '{target}' "
                f"soft diff blend, {changed:.0f}/{diff.numel()} positions "
                f"changed (b={blend:.3f}){override_note}"
            )
        weights = (diff / diff_max).unsqueeze(-1)
        return working_cond + weights * (remap_cond - base_cond) * blend

    logger.warning(
        f"HyperRemapKrea2Edit: Token remap '{source}' -> '{target}' changed "
        f"the sequence length ({base_cond.shape[1]} vs {remap_cond.shape[1]}); "
        f"falling back to truncated global lerp — prior '->' blends on this "
        f"entry are discarded"
    )
    min_seq = min(base_cond.shape[1], remap_cond.shape[1])
    return torch.lerp(
        base_cond[:, :min_seq, :], remap_cond[:, :min_seq, :], blend
    )


def _apply_token_remap_grounded(clip, text, images, pairs, global_blend, debug=False):
    """
    Grounded base encode plus per-pair token-level blending, applied to
    every scheduled conditioning entry.  Returns a conditioning list
    (entries as produced by encode_from_tokens_scheduled, tensors
    replaced by their blended versions).
    """
    base_tokens = _grounded_tokenize(clip, text, images)
    entries = clip.encode_from_tokens_scheduled(base_tokens)

    if not pairs:
        return entries

    # Each pair blends against the untouched base tensors, matching
    # HyperRemap: overlapping pairs overwrite from base, not compound.
    base_conds = [entry[0] for entry in entries]

    for source, target, overrides in pairs:
        blend = overrides.get("b", global_blend)

        remapped_text = apply_text_remappings(text, [(source, target)])
        if remapped_text == text:
            if debug:
                print(
                    f"[HyperRemapKrea2Edit] Token remap: "
                    f"'{source}' not found in prompt"
                )
            continue

        remap_tokens = _grounded_tokenize(clip, remapped_text, images)
        remap_entries = clip.encode_from_tokens_scheduled(remap_tokens)

        if len(remap_entries) != len(entries):
            logger.warning(
                f"HyperRemapKrea2Edit: scheduled entry count mismatch for "
                f"'{source}' -> '{target}' ({len(entries)} vs "
                f"{len(remap_entries)}); blending the first "
                f"{min(len(entries), len(remap_entries))} entries only"
            )

        for i in range(min(len(entries), len(remap_entries))):
            entries[i][0] = _blend_entry(
                base_conds[i],
                entries[i][0],
                remap_entries[i][0],
                base_tokens,
                remap_tokens,
                blend,
                source,
                target,
                "b" in overrides,
                debug,
            )

    return entries


# ---------------------------------------------------------------------------
#  Pipeline — shared by full and slim variants
# ---------------------------------------------------------------------------

def _run_pipeline(
    clip, text, remappings, blend, sharpness, threshold,
    image, image_b, grounding_px, normalize_delta, case_sensitive, debug,
):
    """Full grounded four-phase pass. Returns (conditioning, modified_prompt)."""
    string_pairs, token_pairs, concept_pairs, delta_pairs = _parse_hyper_entries(
        remappings
    )

    images = None
    if image is not None:
        images = [_prep_image(image, grounding_px)]
        if image_b is not None:
            images.append(_prep_image(image_b, grounding_px))
    elif image_b is not None:
        logger.warning(
            "HyperRemapKrea2Edit: image_b connected without image; using it "
            "as the sole grounding reference (multi-ref training order is "
            "scene first, subject second)"
        )
        images = [_prep_image(image_b, grounding_px)]

    grounded = images is not None

    if debug:
        grounded_note = (
            f" ({len(images)} ref(s), cap {grounding_px}px)" if grounded else ""
        )
        print(
            f"[HyperRemapKrea2Edit] Parsed: {len(string_pairs)} string, "
            f"{len(token_pairs)} token, {len(concept_pairs)} concept, "
            f"{len(delta_pairs)} delta | grounded: {grounded}{grounded_note}"
        )

    # -- Phase 1: String replacement (modifies text) --

    modified_prompt = text
    if string_pairs:
        modified_prompt, count = _apply_string_replacements(
            modified_prompt, string_pairs, case_sensitive,
        )
        if debug:
            print(f"[HyperRemapKrea2Edit] String replace: {count} substitution(s)")

    # -- Phase 2: Grounded base encode + token remap --

    entries = _apply_token_remap_grounded(
        clip, modified_prompt, images, token_pairs, blend, debug=debug,
    )

    # -- Phase 3: Concept remap on every scheduled entry --

    if concept_pairs:
        encode_fn = _make_concept_encoder(clip)
        # Differential weighting re-encodes text-only, which can never match
        # a grounded sequence length — skip straight to cosine mode instead
        # of spending encodes on prompts destined to fail the shape check.
        differential_text = None if grounded else modified_prompt
        for entry in entries:
            entry[0] = _apply_concept_remap(
                entry[0],
                clip,
                concept_pairs,
                blend,
                sharpness,
                threshold,
                prompt_text=differential_text,
                debug=debug,
                encode_concept_fn=encode_fn,
            )

    # -- Phase 4: Delta remap on every scheduled entry --

    if delta_pairs:
        for entry in entries:
            entry[0] = _apply_delta_remap(
                entry[0],
                clip,
                delta_pairs,
                blend,
                sharpness,
                threshold,
                normalize=normalize_delta,
                debug=debug,
            )

    return entries, modified_prompt


# ---------------------------------------------------------------------------
#  Node
# ---------------------------------------------------------------------------

class HyperRemapKrea2Edit:
    """
    Four-phase remapping pipeline fused with the krea2_edit grounded
    encode: the prompt is encoded together with the source image(s)
    through the krea2_edit chat template, then string replace, token
    blend, concept nudge, and delta residual are applied to the
    grounded conditioning.
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
                "image": ("IMAGE", {
                    "tooltip": (
                        "Source image to ground the encode on (krea2_edit "
                        "semantic path). Leave disconnected for a text-only "
                        "encode."
                    ),
                }),
                "image_b": ("IMAGE", {
                    "tooltip": (
                        "2nd reference (subject) for multi-ref LoRAs; vision "
                        "blocks in training order: scene first, subject second."
                    ),
                }),
                "grounding_px": ("INT", {
                    "default": 768,
                    "min": 0,
                    "max": 4096,
                    "step": 64,
                    "tooltip": (
                        "Cap the longest side fed to Qwen3-VL; 0 = native "
                        "resolution."
                    ),
                }),
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

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "original_prompt", "modified_prompt")
    FUNCTION = "execute"
    CATEGORY = "CCN/conditioning"
    DESCRIPTION = (
        "Four-phase remapping pipeline fused with the krea2_edit "
        "image-grounded encode. "
        "comma (,) = text replacement. "
        "arrow (->) = token embedding blend. "
        "fat arrow (=>) = concept direction nudge (cosine mode when grounded). "
        "double tilde (~~) = additive residual delta (pooled when grounded). "
        "Per-pair (b:X, s:X, t:X, sx:X, tx:X) overrides in parentheses take "
        "precedence over globals. Connect image (and optionally image_b) to "
        "ground the encode; empty remappings = plain grounded encode."
    )

    def execute(
        self,
        clip,
        text,
        remappings,
        blend,
        sharpness,
        threshold,
        image=None,
        image_b=None,
        grounding_px=768,
        normalize_delta=True,
        case_sensitive=True,
        debug=False,
    ):
        entries, modified_prompt = _run_pipeline(
            clip, text, remappings, blend, sharpness, threshold,
            image, image_b, grounding_px, normalize_delta,
            case_sensitive, debug,
        )
        return (entries, text, modified_prompt)
