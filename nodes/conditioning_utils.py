"""
Conditioning utilities - Lerp, Subtract, and more
"""

import torch
import random


class ConditioningLerp:
    """
    Linear interpolation between two conditionings.

    blend = 0.0 → 100% conditioning_a
    blend = 0.5 → 50/50 mix
    blend = 1.0 → 100% conditioning_b

    Formula: result = a * (1 - blend) + b * blend

    Sequence lengths may differ (e.g. Krea2/Qwen3-VL conditioning with
    vision tokens vs. a plain text encode). Conditioning A is never
    truncated: blending applies over the overlapping token region, and
    positions beyond B's length remain pure A.
    """

    CATEGORY = "ComfyCollectorNodes/Conditioning"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_a": ("CONDITIONING",),
                "conditioning_b": ("CONDITIONING",),
                "blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "reuse_first_b_entry": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Only matters when conditioning_a contains MORE entries "
                        "than conditioning_b (rare; most conditionings are a "
                        "single entry).  ON = blend B's first entry into the "
                        "extra A entries.  OFF = pass the extra A entries "
                        "through completely unchanged."
                    ),
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "lerp"

    def lerp(self, conditioning_a, conditioning_b, blend,
             reuse_first_b_entry=True, debug=False):
        # True identity at the endpoints — avoids any structural changes
        # (truncation, dict rebuilds) when no blending is requested.
        if blend == 0.0:
            return (conditioning_a,)
        if blend == 1.0:
            return (conditioning_b,)

        result = []

        for i in range(len(conditioning_a)):
            entry_a = conditioning_a[i]

            # Extra A entries (beyond B's list) are never dropped: either
            # B's first entry is reused for them, or they pass through.
            if i >= len(conditioning_b):
                if not reuse_first_b_entry:
                    if debug:
                        print(
                            f"[CCN ConditioningLerp] Entry {i}: no matching B "
                            f"entry, passing A through unchanged"
                        )
                    result.append(entry_a)
                    continue
                entry_b = conditioning_b[0]
            else:
                entry_b = conditioning_b[i]

            cond_a = entry_a[0]
            # Align B to A so cross-device or mixed-precision encodes work.
            cond_b = entry_b[0].to(device=cond_a.device, dtype=cond_a.dtype)

            dict_a = entry_a[1].copy() if len(entry_a) > 1 else {}
            dict_b = entry_b[1] if len(entry_b) > 1 else {}

            # Blend only where both sequences have content; A's tail
            # (e.g. vision/template tokens) passes through untouched.
            min_len = min(cond_a.shape[1], cond_b.shape[1])
            blended = cond_a.clone()
            blended[:, :min_len, :] = torch.lerp(
                cond_a[:, :min_len, :], cond_b[:, :min_len, :], blend
            )

            result_dict = dict_a
            pooled_a_tensor = dict_a.get("pooled_output")
            pooled_b_tensor = dict_b.get("pooled_output")
            if pooled_a_tensor is not None and pooled_b_tensor is not None:
                pooled_b_tensor = pooled_b_tensor.to(
                    device=pooled_a_tensor.device, dtype=pooled_a_tensor.dtype
                )
                # Different encoders can produce different pooled dims;
                # keep A's pooled rather than erroring on mismatch.
                if pooled_a_tensor.shape == pooled_b_tensor.shape:
                    result_dict["pooled_output"] = torch.lerp(
                        pooled_a_tensor, pooled_b_tensor, blend
                    )
                elif debug:
                    print(
                        f"[CCN ConditioningLerp] Entry {i}: pooled shape mismatch "
                        f"({list(pooled_a_tensor.shape)} vs {list(pooled_b_tensor.shape)}), "
                        f"keeping A's pooled_output"
                    )

            if debug:
                print(
                    f"[CCN ConditioningLerp] Entry {i}: blend={blend:.3f} | "
                    f"A seq={cond_a.shape[1]}, B seq={cond_b.shape[1]}, "
                    f"blended region=[0:{min_len}], "
                    f"A-only tail={cond_a.shape[1] - min_len} positions"
                )

            result.append([blended, result_dict])

        return (result,)


class ConditioningSubtract:
    """
    Subtract one conditioning from another.

    Useful for conceptual removal:
      full_scene - "snow" = scene without snow concept

    Formula: result = conditioning_a - conditioning_b * strength

    Sequence lengths may differ. Conditioning A is never truncated:
    subtraction applies over the overlapping token region only, which is
    equivalent to zero-padding B out to A's length. A's tail (e.g.
    Krea2/Qwen3-VL vision and template tokens) passes through untouched.
    """

    CATEGORY = "ComfyCollectorNodes/Conditioning"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_a": ("CONDITIONING",),
                "conditioning_b": ("CONDITIONING",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
            },
            "optional": {
                "reuse_first_b_entry": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Only matters when conditioning_a contains MORE entries "
                        "than conditioning_b (rare; most conditionings are a "
                        "single entry).  ON = subtract B's first entry from the "
                        "extra A entries.  OFF = pass the extra A entries "
                        "through completely unchanged (nothing subtracted)."
                    ),
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "subtract"

    def subtract(self, conditioning_a, conditioning_b, strength,
                 reuse_first_b_entry=True, debug=False):
        # True identity at zero strength — previously the node still
        # truncated A here, destroying long conditionings (Krea2).
        if strength == 0.0:
            return (conditioning_a,)

        result = []

        for i in range(len(conditioning_a)):
            entry_a = conditioning_a[i]

            # Extra A entries (beyond B's list) are never dropped: either
            # B's first entry is reused for them, or they pass through.
            if i >= len(conditioning_b):
                if not reuse_first_b_entry:
                    if debug:
                        print(
                            f"[CCN ConditioningSubtract] Entry {i}: no matching "
                            f"B entry, passing A through unchanged"
                        )
                    result.append(entry_a)
                    continue
                entry_b = conditioning_b[0]
            else:
                entry_b = conditioning_b[i]

            cond_a = entry_a[0]
            # Align B to A so cross-device or mixed-precision encodes work.
            cond_b = entry_b[0].to(device=cond_a.device, dtype=cond_a.dtype)

            dict_a = entry_a[1].copy() if len(entry_a) > 1 else {}
            dict_b = entry_b[1] if len(entry_b) > 1 else {}

            # Subtract only where B has content; positions beyond B's
            # length are unchanged (subtracting an implicit zero pad).
            min_len = min(cond_a.shape[1], cond_b.shape[1])
            modified = cond_a.clone()
            modified[:, :min_len, :] = (
                cond_a[:, :min_len, :] - cond_b[:, :min_len, :] * strength
            )

            result_dict = dict_a
            pooled_a_tensor = dict_a.get("pooled_output")
            pooled_b_tensor = dict_b.get("pooled_output")
            if pooled_a_tensor is not None and pooled_b_tensor is not None:
                pooled_b_tensor = pooled_b_tensor.to(
                    device=pooled_a_tensor.device, dtype=pooled_a_tensor.dtype
                )
                # Different encoders can produce different pooled dims;
                # keep A's pooled rather than erroring on mismatch.
                if pooled_a_tensor.shape == pooled_b_tensor.shape:
                    result_dict["pooled_output"] = (
                        pooled_a_tensor - pooled_b_tensor * strength
                    )
                elif debug:
                    print(
                        f"[CCN ConditioningSubtract] Entry {i}: pooled shape mismatch "
                        f"({list(pooled_a_tensor.shape)} vs {list(pooled_b_tensor.shape)}), "
                        f"keeping A's pooled_output"
                    )

            if debug:
                print(
                    f"[CCN ConditioningSubtract] Entry {i}: strength={strength:.3f} | "
                    f"A seq={cond_a.shape[1]}, B seq={cond_b.shape[1]}, "
                    f"subtracted region=[0:{min_len}], "
                    f"A-only tail={cond_a.shape[1] - min_len} positions"
                )

            result.append([modified, result_dict])

        return (result,)


class RandomSelect:
    """
    Randomly select one of up to 5 inputs.

    Only connected inputs are considered.
    Re-rolls each execution unless seed is set (seed >= 0).
    """

    CATEGORY = "ComfyCollectorNodes/Utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "input_1": ("*",),
                "input_2": ("*",),
                "input_3": ("*",),
                "input_4": ("*",),
                "input_5": ("*",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("*", "INT")
    RETURN_NAMES = ("output", "selected_index")
    FUNCTION = "select"

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        # Without this, ComfyUI caches the node when inputs are unchanged
        # and the promised per-execution re-roll never happens. NaN forces
        # re-execution; a fixed seed allows caching (ComfyUI still combines
        # this with the input hash, so changed inputs re-run regardless).
        if seed is not None and seed >= 0:
            return seed
        return float("nan")

    def select(self, input_1=None, input_2=None, input_3=None, input_4=None,
               input_5=None, seed=-1, debug=False):
        inputs = []
        for i, inp in enumerate([input_1, input_2, input_3, input_4, input_5], 1):
            if inp is not None:
                inputs.append((i, inp))

        if not inputs:
            # A None output would fail downstream with a far less clear
            # error, so fail here where the cause is visible.
            raise ValueError(
                "RandomSelect: no inputs connected — connect at least one input"
            )

        # Local RNG instance: seeding the global `random` module here would
        # silently reseed every other consumer of it in the process.
        rng = random.Random(seed) if seed >= 0 else random.Random()

        selected_idx, selected_value = rng.choice(inputs)

        if debug:
            print(
                f"[CCN RandomSelect] Picked input_{selected_idx} "
                f"(of {len(inputs)} connected, seed={'random' if seed < 0 else seed})"
            )

        return (selected_value, selected_idx)
