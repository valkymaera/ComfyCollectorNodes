"""
Curve From Core - Convert ComfyUI's native CURVE into a CCN_CURVE
"""

import math

# comfy_api gained curve support alongside the native CURVE type; older
# installs lack it, in which case only raw dict/list inputs can arrive anyway
try:
    from comfy_api.input import CurveInput
except ImportError:
    CurveInput = None

# Resolution used when an unknown CurveInput subclass (analytical, LUT-based,
# etc.) must be resampled through its public interface because its shape is
# not recoverable from control points alone
RESAMPLE_COUNT = 33

# Width used to separate duplicate-x points (step discontinuities). Larger
# than the CCN evaluator's 1e-10 zero-width segment guard, small enough to
# be invisible in practice
STEP_EPSILON = 1e-9


def _separate_duplicate_x(keys):
    """Shift earlier members of a duplicate-x run slightly left.

    Core evaluates a duplicate-x step by taking the LAST point's y at the
    shared x (searchsorted side='right'), while the CCN evaluator's scan
    would take the first. Tangents are computed on the original points
    beforehand, so only the emitted positions move; the step lands within
    STEP_EPSILON of its true position with the correct value on both sides.
    """
    i = 0
    n = len(keys)
    while i < n:
        j = i
        while j + 1 < n and keys[j + 1]["x"] == keys[i]["x"]:
            j += 1
        if j > i:
            for m in range(i, j):
                keys[m]["x"] = keys[j]["x"] - (j - m) * STEP_EPSILON
        i = j + 1
    return keys


def _fritsch_carlson_slopes(points):
    """Monotone tangents for sorted (x, y) points.

    Mirrors MonotoneCubicCurve._compute_slopes in comfy_api's curve_types so
    converted curves evaluate identically to the core implementation.
    """
    n = len(points)
    if n < 2:
        return [0.0] * n

    deltas = []
    for i in range(n - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        deltas.append(0.0 if dx == 0 else dy / dx)

    slopes = [0.0] * n
    slopes[0] = deltas[0]
    slopes[-1] = deltas[-1]
    for i in range(1, n - 1):
        if deltas[i - 1] * deltas[i] <= 0:
            slopes[i] = 0.0
        else:
            slopes[i] = (deltas[i - 1] + deltas[i]) / 2

    for i in range(n - 1):
        if deltas[i] == 0:
            slopes[i] = 0.0
            slopes[i + 1] = 0.0
        else:
            alpha = slopes[i] / deltas[i]
            beta = slopes[i + 1] / deltas[i]
            s = alpha * alpha + beta * beta
            if s > 9:
                t = 3.0 / math.sqrt(s)
                slopes[i] = t * alpha * deltas[i]
                slopes[i + 1] = t * beta * deltas[i]
    return slopes


def _keys_from_monotone(points):
    """Exact conversion: core's monotone cubic is a Hermite spline whose
    per-point tangent is the Fritsch-Carlson slope on both sides."""
    slopes = _fritsch_carlson_slopes(points)
    return [
        {"x": x, "y": y, "in": s, "out": s, "mirrored": True}
        for (x, y), s in zip(points, slopes)
    ]


def _keys_from_linear(points):
    """Exact conversion: a Hermite segment whose tangents both equal the
    chord slope degenerates to that straight line."""
    n = len(points)
    if n == 1:
        x, y = points[0]
        return [{"x": x, "y": y, "in": 0.0, "out": 0.0, "mirrored": True}]

    deltas = []
    for i in range(n - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        deltas.append(0.0 if dx == 0 else dy / dx)

    keys = []
    for i, (x, y) in enumerate(points):
        t_in = deltas[i - 1] if i > 0 else deltas[0]
        t_out = deltas[i] if i < n - 1 else deltas[-1]
        keys.append({
            "x": x, "y": y,
            "in": t_in, "out": t_out,
            "mirrored": t_in == t_out,
        })
    return keys


def _extract_points(core_curve):
    """Normalize any core curve representation to (points, interpolation).

    Accepts a CurveInput instance (core's CurveEditor node outputs these),
    a {"points", "interpolation"} dict, a bare point list, or any of those
    wrapped in the frontend's {"__value__": ...} widget envelope.
    Raises ValueError with explicit structure info on anything else.
    """
    data = core_curve
    if isinstance(data, dict) and "__value__" in data:
        data = data["__value__"]

    if CurveInput is not None and isinstance(data, CurveInput):
        cls_name = type(data).__name__
        points = [(float(x), float(y)) for x, y in data.points]
        if cls_name == "LinearCurve":
            return points, "linear"
        if cls_name == "MonotoneCubicCurve":
            return points, "monotone_cubic"
        print(
            f"[ComfyCollectorNodes] CurveFromCore: unknown curve class "
            f"'{cls_name}', resampling at {RESAMPLE_COUNT} points (approximate)"
        )
        xs = [i / (RESAMPLE_COUNT - 1) for i in range(RESAMPLE_COUNT)]
        return [(x, float(data.interp(x))) for x in xs], "monotone_cubic"

    if isinstance(data, dict):
        if "points" not in data:
            raise ValueError(
                f"CurveFromCore: dict input has no 'points' key "
                f"(keys: {list(data.keys())})"
            )
        raw_points = data["points"]
        interpolation = data.get("interpolation", "monotone_cubic")
    elif isinstance(data, (list, tuple)):
        raw_points = data
        interpolation = "monotone_cubic"
    else:
        raise ValueError(
            f"CurveFromCore: unsupported input type '{type(data).__name__}' "
            f"(expected CurveInput, dict, or point list)"
        )

    points = []
    for i, p in enumerate(raw_points):
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            raise ValueError(
                f"CurveFromCore: point {i} is not an (x, y) pair: {p!r}"
            )
        points.append((float(p[0]), float(p[1])))

    if not points:
        raise ValueError("CurveFromCore: curve contains no points")

    if interpolation not in ("monotone_cubic", "linear"):
        print(
            f"[ComfyCollectorNodes] CurveFromCore: unknown interpolation "
            f"'{interpolation}', treating as monotone_cubic"
        )
        interpolation = "monotone_cubic"

    points.sort(key=lambda p: p[0])
    return points, interpolation


class CurveFromCore:
    """Convert ComfyUI's native CURVE (control points + interpolation mode)
    into a CCN_CURVE (Hermite keys with tangents).

    monotone_cubic and linear inputs convert exactly, since both are
    special cases of the Hermite spline CCN curves use. Unknown CurveInput
    subclasses are resampled through their public evaluation interface.

    Lets curves authored with core's native editor (e.g. the CurveEditor
    node) drive CCN curve consumers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # forceInput keeps this a plain socket; without it the
                # frontend attaches its Vue-only CURVE widget, which renders
                # as a "Node 2.0 only" placeholder in legacy mode
                "core_curve": ("CURVE", {
                    "forceInput": True,
                    "tooltip": "Native ComfyUI curve, e.g. from the core "
                               "CurveEditor node.",
                }),
            },
            "optional": {
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print conversion details to console.",
                }),
            },
        }

    RETURN_TYPES = ("CCN_CURVE",)
    RETURN_NAMES = ("curve",)
    FUNCTION = "convert"
    CATEGORY = "CCN"
    DESCRIPTION = ("Convert ComfyUI's native CURVE into a CCN_CURVE. "
                   "Exact for monotone_cubic and linear curves.")

    def convert(self, core_curve, debug=False):
        points, interpolation = _extract_points(core_curve)

        if interpolation == "linear":
            keys = _keys_from_linear(points)
        else:
            keys = _keys_from_monotone(points)

        keys = _separate_duplicate_x(keys)

        if debug:
            print(
                f"[ComfyCollectorNodes] CurveFromCore: converted "
                f"{len(points)} {interpolation} point(s) to "
                f"{len(keys)} Hermite key(s)"
            )

        return (keys,)
