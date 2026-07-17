"""
HyperRemapKrea2EditSlim — Compact variant of HyperRemapKrea2Edit.

Same grounded four-phase pipeline (string replace, token blend,
concept nudge, delta residual on krea2_edit image-grounded encoding)
but with a slimmer node profile:

  - text input is wire-only (no widget, forceInput)
  - only outputs modified conditioning
  - no debug or case_sensitive options
  - displayed widgets: remappings, blend, sharpness, threshold,
    grounding_px
"""

import logging

from .hyper_remap_krea2edit import _run_pipeline

logger = logging.getLogger("CCN.HyperRemapKrea2EditSlim")


class HyperRemapKrea2EditSlim:
    """
    Compact grounded four-phase remapping: string replace, token blend,
    concept nudge, delta residual on krea2_edit image-grounded encoding.
    Wire-only text input, conditioning-only output.
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
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "execute"
    CATEGORY = "CCN/conditioning"
    DESCRIPTION = (
        "Compact grounded four-phase remapping pipeline (wire-only text "
        "input) on krea2_edit image-grounded encoding. "
        "comma (,) = text replacement. "
        "arrow (->) = token embedding blend. "
        "fat arrow (=>) = concept direction nudge (cosine mode when grounded). "
        "double tilde (~~) = additive residual delta (pooled when grounded). "
        "Per-pair (b:X, s:X, t:X, sx:X, tx:X) overrides in parentheses take "
        "precedence over globals."
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
    ):
        entries, _ = _run_pipeline(
            clip, text, remappings, blend, sharpness, threshold,
            image, image_b, grounding_px, normalize_delta,
            case_sensitive=True, debug=False,
        )
        return (entries,)
