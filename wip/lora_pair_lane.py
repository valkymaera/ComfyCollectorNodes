"""
LoRA Pair Lane -- one schedulable LoRA (pair) for MoE Sampler Dual.

A lane names a LoRA file, peak strengths for the high and low experts, and
optionally a CCN_CURVE that multiplies those strengths over GLOBAL run
progress (0 = first step of the whole schedule, 1 = last step -- not
per-phase, so a single curve can span the expert boundary, e.g. a speed
LoRA ramping in after the first high-noise steps).

Partner files are resolved with the same existence-guarded token swapping
as LoraPairLoader; an unpaired file applies to both experts. A strength of
0 skips that side, exactly like the loader's escape hatch.

Lanes chain: wire one lane's output into the next lane's `lanes` input and
the final output into MoESamplerDual's `lora_lanes`. The sampler compiles
all lanes' curves into per-phase strength segments (see its lane_segments
input) and applies them by re-patching between segments.

For a CONSTANT strength with no curve, prefer the static LoraPairLoader
upstream -- it costs nothing at sampling time. A constant lane behaves
identically but exists mainly so a curve can be added later without
rewiring.
"""

import os

import folder_paths

from .lora_pair_loader import _resolve_pair, _NONE_VALUES


class LoraPairLane:
    """Build a schedulable LoRA lane for MoESamplerDual."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"), {
                    "tooltip": "LoRA file; a HIGH/LOW partner is auto-resolved "
                               "from the filename when one exists.",
                }),
                "strength_high": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Peak strength on the high-noise expert. "
                               "0 skips that side.",
                }),
                "strength_low": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Peak strength on the low-noise expert. "
                               "0 skips that side.",
                }),
            },
            "optional": {
                "curve": ("CCN_CURVE", {
                    "tooltip": "Strength multiplier over global run progress "
                               "(curve y * strength). Absent = constant.",
                }),
                "lanes": ("CCN_LORA_LANES", {
                    "tooltip": "Chain from a previous LoRA Pair Lane.",
                }),
            },
        }

    RETURN_TYPES = ("CCN_LORA_LANES", "STRING")
    RETURN_NAMES = ("lanes", "info")
    FUNCTION = "build"
    CATEGORY = "CCN"
    DESCRIPTION = (
        "One schedulable LoRA (pair) with a curve-driven strength over the "
        "whole run; chain lanes and feed MoE Sampler Dual's lora_lanes."
    )

    def build(self, lora_name, strength_high, strength_low, curve=None,
              lanes=None):
        out = list(lanes) if lanes else []

        if not isinstance(lora_name, str) or lora_name in _NONE_VALUES:
            return (out, "no LoRA selected -- lane passed through")

        available = folder_paths.get_filename_list("loras")
        if lora_name not in available:
            raise ValueError(
                f"LoraPairLane: {lora_name!r} is not in the loras folder. "
                "Refresh the node or fix the selection.")

        lower_to_actual = {n.lower(): n for n in available}
        side, partner = _resolve_pair(lora_name, lower_to_actual)
        if partner is not None:
            if side == "high":
                file_high, file_low = lora_name, partner
            else:
                file_high, file_low = partner, lora_name
            pair_note = f"pair ({file_high} / {file_low})"
        else:
            file_high = file_low = lora_name
            pair_note = "single file -> both experts"

        name = os.path.splitext(os.path.basename(lora_name))[0]
        out.append({
            "name": name,
            "file_high": file_high,
            "file_low": file_low,
            "strength_high": float(strength_high),
            "strength_low": float(strength_low),
            "curve": curve,
        })

        curve_note = "curved" if curve is not None else "constant"
        info = (f"lane {len(out)}: {name} -- {pair_note}; "
                f"high @{float(strength_high):g}, low @{float(strength_low):g} "
                f"({curve_note})")
        return (out, info)
