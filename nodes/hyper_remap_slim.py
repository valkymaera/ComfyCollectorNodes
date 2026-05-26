"""
HyperRemapSlim — Compact variant of HyperRemap.

Same four-phase pipeline (string replace, token blend, concept nudge,
delta residual) but with a slimmer node profile:

  - text input is wire-only (no widget, forceInput)
  - only outputs modified conditioning
  - no debug or case_sensitive options
  - displayed widgets: remappings, blend, sharpness, threshold
"""

import logging

from .hyper_remap import (
    _parse_hyper_entries,
    _apply_string_replacements,
    _apply_token_remap,
    _apply_concept_remap,
    _apply_delta_remap,
)

logger = logging.getLogger("CCN.HyperRemapSlim")


class HyperRemapSlim:
    """
    Compact four-phase remapping: string replace, token blend, concept nudge,
    delta residual.  Wire-only text input, conditioning-only output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {
                    "forceInput": True,
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
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "execute"
    CATEGORY = "CCN/conditioning"
    DESCRIPTION = (
        "Compact four-phase remapping pipeline (wire-only text input). "
        "comma (,) = text replacement. "
        "arrow (->) = token embedding blend. "
        "fat arrow (=>) = concept direction nudge. "
        "double tilde (~~) = additive residual delta. "
        "Per-pair (b:X, s:X, t:X, sx:X, tx:X) overrides in "
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
    ):
        string_pairs, token_pairs, concept_pairs, delta_pairs = _parse_hyper_entries(remappings)

        # -- Phase 1: String replacement (modifies text) --
        modified_prompt = text
        if string_pairs:
            modified_prompt, _ = _apply_string_replacements(
                modified_prompt, string_pairs, case_sensitive=True,
            )

        # -- Phase 2: Token remap (embedding blend on modified_prompt) --
        working_cond, working_pooled = _apply_token_remap(
            clip, modified_prompt, token_pairs, blend,
        )

        # -- Phase 3: Concept remap (direction nudge) --
        if concept_pairs:
            working_cond = _apply_concept_remap(
                working_cond,
                clip,
                concept_pairs,
                blend,
                sharpness,
                threshold,
                prompt_text=modified_prompt,
            )

        # -- Phase 4: Delta remap (additive residual) --
        if delta_pairs:
            working_cond = _apply_delta_remap(
                working_cond,
                clip,
                delta_pairs,
                blend,
                sharpness,
                threshold,
                normalize=normalize_delta,
            )

        return ([[working_cond, working_pooled]],)
