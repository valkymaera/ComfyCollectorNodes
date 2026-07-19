"""
Curve To Core - Convert a CCN_CURVE into ComfyUI's native CURVE
"""

import json

from .curve_cfg_guider import hermite

# comfy_api gained curve support alongside the native CURVE type; when
# available, output a CurveInput instance to match core's own convention
# (the CurveEditor node also outputs instances rather than raw dicts)
try:
    from comfy_api.input import CurveInput
except ImportError:
    CurveInput = None


DEFAULT_CURVE = '[{"x":0,"y":0,"in":0,"out":1,"mirrored":true},{"x":1,"y":1,"in":1,"out":0,"mirrored":true}]'


class CurveToCore:
    """Convert a CCN_CURVE (Hermite keys with tangents) into ComfyUI's
    native CURVE (control points + interpolation mode).

    Core's format derives tangents from points, so arbitrary Hermite
    tangents cannot be carried over exactly; the CCN curve is sampled
    instead. Key positions are always included in the sample set, so key
    values are preserved exactly and deviation between samples shrinks as
    the sample count rises.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("INT", {
                    "default": 17, "min": 2, "max": 1025, "step": 1,
                    "tooltip": "Uniform sample count across 0..1 (key "
                               "positions are added on top). Higher is more "
                               "faithful; lower stays editable if the result "
                               "is fed into a native curve editor.",
                }),
                "interpolation": (["monotone_cubic", "linear"], {
                    "default": "monotone_cubic",
                    "tooltip": "Interpolation mode of the emitted core curve.",
                }),
                "curve_data": ("STRING", {
                    "default": DEFAULT_CURVE, "multiline": False,
                }),
            },
            "optional": {
                "curve": ("CCN_CURVE", {
                    "tooltip": "Overrides curve_data when connected.",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print conversion details to console.",
                }),
            },
        }

    RETURN_TYPES = ("CURVE",)
    RETURN_NAMES = ("core_curve",)
    FUNCTION = "convert"
    CATEGORY = "CCN"
    DESCRIPTION = ("Convert a CCN_CURVE into ComfyUI's native CURVE by "
                   "sampling the Hermite curve. Key positions are sampled "
                   "exactly.")

    def convert(self, samples, interpolation, curve_data, curve=None, debug=False):
        if curve is not None:
            keys = curve
        else:
            try:
                keys = json.loads(curve_data)
            except (json.JSONDecodeError, TypeError):
                print(
                    "[ComfyCollectorNodes] CurveToCore: invalid curve_data "
                    "JSON, using default curve"
                )
                keys = json.loads(DEFAULT_CURVE)

        keys = sorted(keys, key=lambda k: k["x"])

        n = max(2, int(samples))
        xs = {i / (n - 1) for i in range(n)}
        for k in keys:
            xs.add(min(1.0, max(0.0, float(k["x"]))))

        # Merge near-identical positions so float noise between uniform
        # samples and key positions can't create zero-width segments
        merged = []
        for x in sorted(xs):
            if merged and x - merged[-1] < 1e-9:
                continue
            merged.append(x)

        points = [[x, hermite(keys, x)] for x in merged]
        raw = {"points": points, "interpolation": interpolation}

        if CurveInput is not None:
            core_curve = CurveInput.from_raw(raw)
        else:
            print(
                "[ComfyCollectorNodes] CurveToCore: comfy_api curve support "
                "not found (older ComfyUI?), emitting raw curve dict"
            )
            core_curve = raw

        if debug:
            print(
                f"[ComfyCollectorNodes] CurveToCore: sampled {len(keys)} "
                f"Hermite key(s) to {len(points)} {interpolation} point(s)"
            )

        return (core_curve,)
