class FloatLerp:
    """Unclamped linear interpolation between two float values."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "b": ("FLOAT", {"default": 1.0, "step": 0.001}),
                "t": ("FLOAT", {"default": 0.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "lerp"
    CATEGORY = "CCN"

    def lerp(self, a, b, t):
        return (a + (b - a) * t,)
