"""
Dimension scaling utility - operate on width/height values without requiring images.
"""


class DimensionScale:
    """
    Scale input dimensions relative to reference dimensions.
    
    Modes:
    - scale_width:  Match ref width, adjust height proportionally.
    - scale_height: Match ref height, adjust width proportionally.
    - match_exact:  Output ref dimensions directly (ignores aspect ratio).
    - smart_scale:  Pick the axis with the least percent change, scale proportionally.
    """

    CATEGORY = "ComfyCollectorNodes/Dimension"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 1, "max": 65536}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 65536}),
                "ref_width": ("INT", {"default": 1920, "min": 1, "max": 65536}),
                "ref_height": ("INT", {"default": 1080, "min": 1, "max": 65536}),
                "scale_type": (["scale_width", "scale_height", "match_exact", "smart_scale"], {"default": "smart_scale"}),
                "round_to": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "info")
    FUNCTION = "scale"

    def scale(self, width, height, ref_width, ref_height, scale_type, round_to):
        if scale_type == "scale_width":
            factor = ref_width / width
            new_w = int(width * factor)
            new_h = int(height * factor)
            info = f"scale_width factor={factor:.4f}"

        elif scale_type == "scale_height":
            factor = ref_height / height
            new_w = int(width * factor)
            new_h = int(height * factor)
            info = f"scale_height factor={factor:.4f}"

        elif scale_type == "match_exact":
            new_w = ref_width
            new_h = ref_height
            info = "match_exact"

        elif scale_type == "smart_scale":
            factor_w = ref_width / width
            factor_h = ref_height / height
            pct_w = abs(factor_w - 1.0)
            pct_h = abs(factor_h - 1.0)
            if pct_w <= pct_h:
                factor = factor_w
                axis = "width"
            else:
                factor = factor_h
                axis = "height"
            new_w = int(width * factor)
            new_h = int(height * factor)
            info = f"smart_scale axis={axis} factor={factor:.4f} Δw={pct_w*100:.1f}% Δh={pct_h*100:.1f}%"

        else:
            raise ValueError(f"Unknown scale_type: {scale_type}")

        new_w = _round(new_w, round_to)
        new_h = _round(new_h, round_to)

        print(f"[CCN DimensionScale] {width}x{height} -> {new_w}x{new_h} ({info})")
        return (new_w, new_h, info)


def _round(value, step):
    """Round to nearest multiple of step, minimum step."""
    return max(step, (value // step) * step)
