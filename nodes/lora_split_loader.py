"""
LoRA Split Loader -- blends two LoRAs with a single balance slider.

The two fixed rows are provided by js/lora_split_loader.js. They arrive as
widget values named "lora_a" and "lora_b", each a dict:
    {"on": bool, "lora": str, "slot": "a" | "b"}
("slot" is a save/load discriminator for the frontend; Python ignores it.)

The balance slider splits strength between the two: 0.0 applies A at full
strength and B at none, 1.0 the reverse, 0.5 both at half. Both final
strengths are scaled by `strength_multiplier`. Disabling one row skips it
without renormalizing the other. The `enabled` toggle bypasses the whole
node, passing model/CLIP through unchanged. The `triggers` output joins the
top trigger words of every applied row, reusing the multi loader's
extraction so the chips shown in the UI match the output string.
"""

import folder_paths
import comfy.sd
import comfy.utils

from .lora_multi_loader import get_top_triggers, _NONE_VALUES


class AnyType(str):
    """A type string that never fails an equality check against another type."""

    def __ne__(self, other):
        return False


class FlexibleOptionalInputType(dict):
    """Accepts arbitrarily named optional inputs (the row widgets), while
    still declaring explicit known sockets."""

    def __init__(self, input_type, explicit=None):
        super().__init__()
        self.input_type = input_type
        if explicit:
            self.update(explicit)

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        return (self.input_type,)

    def __contains__(self, key):
        return True


_ANY = AnyType("*")


class LoraSplitLoader:
    """Two-LoRA blend: a balance slider splits strength between A and B.

    strength_a = (1 - balance) * strength_multiplier
    strength_b = balance * strength_multiplier
    A side whose resolved strength is 0 (or whose row is toggled off) is
    skipped entirely, including its trigger words.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # enabled -> balance -> strength_multiplier order is load-bearing:
        # the JS configure() maps saved numbers back by this widget order.
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to patch."}),
            },
            "optional": FlexibleOptionalInputType(_ANY, {
                "clip": ("CLIP", {
                    "tooltip": "Optional; strengths also apply to the text "
                               "encoder when connected.",
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When off, the model and CLIP pass through "
                               "unchanged and no triggers are emitted.",
                }),
                "balance": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Blend between the two LoRAs: 0.0 = A only, "
                               "1.0 = B only, 0.5 = both at half strength.",
                }),
                "strength_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Multiplies both final strengths; 0 disables "
                               "both LoRAs.",
                }),
            }),
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "triggers")
    FUNCTION = "load"
    CATEGORY = "CCN"
    DESCRIPTION = (
        "Blends two LoRAs with a balance slider: 0 applies A at full "
        "strength, 1 applies B at full strength, 0.5 applies both at half. "
        "Both final strengths are scaled by the strength multiplier, and "
        "each row has its own enable toggle. The triggers output joins each "
        "applied LoRA's top trigger words from embedded training metadata."
    )

    def load(self, model, clip=None, enabled=True, balance=0.5,
             strength_multiplier=1.0, lora_a=None, lora_b=None, **kwargs):
        if not enabled:
            return (model, clip, "")

        try:
            balance = float(balance)
            strength_multiplier = float(strength_multiplier)
        except (TypeError, ValueError):
            raise ValueError(
                "LoraSplitLoader: balance or strength_multiplier is "
                f"non-numeric: {balance!r} / {strength_multiplier!r}"
            )
        # Converted inputs can feed anything; keep the split in range.
        balance = min(1.0, max(0.0, balance))

        slots = (
            ("A", lora_a, (1.0 - balance) * strength_multiplier),
            ("B", lora_b, balance * strength_multiplier),
        )
        available = set(folder_paths.get_filename_list("loras"))
        loaded = {}  # path -> tensors, so the same lora in both slots loads once
        trigger_words = []
        for label, row, strength in slots:
            if not isinstance(row, dict) or not row.get("on", True):
                continue

            lora_name = row.get("lora")
            if not isinstance(lora_name, str) or lora_name in _NONE_VALUES:
                continue

            if strength == 0.0:
                continue

            if lora_name not in available:
                raise ValueError(
                    f"LoraSplitLoader: slot {label} references {lora_name!r}, "
                    "which is not in the loras folder. Refresh the node or "
                    "fix the row; refusing to run with a silently missing "
                    "LoRA."
                )

            path = folder_paths.get_full_path_or_raise("loras", lora_name)
            lora = loaded.get(path)
            if lora is None:
                try:
                    lora = comfy.utils.load_torch_file(path, safe_load=True)
                except Exception as exc:
                    raise ValueError(
                        f"LoraSplitLoader: slot {label} failed to load "
                        f"{lora_name!r}: {exc}"
                    ) from exc
                loaded[path] = lora
            model, clip = comfy.sd.load_lora_for_models(
                model, clip, lora, strength, strength)

            for word in get_top_triggers(lora_name):
                if word not in trigger_words:
                    trigger_words.append(word)

        return (model, clip, ", ".join(trigger_words))
