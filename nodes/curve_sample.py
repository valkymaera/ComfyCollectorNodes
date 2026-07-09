import json


class CurveSample:
    """Sample a value from an interactive bezier curve at a given position.
    
    The curve is defined in normalized (0,0)..(1,1) space with endpoints
    pinned. Evaluation uses cubic Hermite interpolation (same basis as
    Unity AnimationCurves).
    
    x_scale: incoming t is divided by this before sampling (maps 0..x_scale → 0..1)
    y_scale: sampled value is multiplied by this on output
    """

    DEFAULT_CURVE = '[{"x":0,"y":0,"in":0,"out":1,"mirrored":true},{"x":1,"y":1,"in":1,"out":0,"mirrored":true}]'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "x_scale": ("FLOAT", {"default": 1.0, "step": 0.001}),
                "y_scale": ("FLOAT", {"default": 1.0, "step": 0.001}),
                "curve_data": ("STRING", {"default": CurveSample.DEFAULT_CURVE, "multiline": False}),
            },
            "optional": {
                "curve": ("CCN_CURVE",),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "sample"
    CATEGORY = "CCN"

    @staticmethod
    def _hermite(keys, t):
        """Evaluate cubic Hermite spline at position t."""
        n = len(keys)
        if n == 0:
            return 0.0
        if n == 1:
            return keys[0]["y"]

        if t <= keys[0]["x"]:
            return keys[0]["y"]
        if t >= keys[-1]["x"]:
            return keys[-1]["y"]

        idx = 0
        for i in range(n - 1):
            if keys[i]["x"] <= t <= keys[i + 1]["x"]:
                idx = i
                break

        k0, k1 = keys[idx], keys[idx + 1]
        dt = k1["x"] - k0["x"]
        if dt < 1e-10:
            return k0["y"]

        lt = (t - k0["x"]) / dt
        lt2 = lt * lt
        lt3 = lt2 * lt

        h00 = 2.0 * lt3 - 3.0 * lt2 + 1.0
        h10 = lt3 - 2.0 * lt2 + lt
        h01 = -2.0 * lt3 + 3.0 * lt2
        h11 = lt3 - lt2

        return h00 * k0["y"] + h10 * k0["out"] * dt + h01 * k1["y"] + h11 * k1["in"] * dt

    def sample(self, t, x_scale, y_scale, curve_data, curve=None):
        if curve is not None:
            keys = curve
        else:
            try:
                keys = json.loads(curve_data)
            except (json.JSONDecodeError, TypeError):
                return (0.0,)

        keys = sorted(keys, key=lambda k: k["x"])

        # Normalize t by x_scale, then clamp to 0..1
        nt = t / x_scale if abs(x_scale) > 1e-10 else 0.0
        nt = max(0.0, min(1.0, nt))

        y = self._hermite(keys, nt)

        return (y * y_scale,)
