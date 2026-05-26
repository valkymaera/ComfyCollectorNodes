import json


DEFAULT_CURVE = '[{"x":0,"y":0,"in":0,"out":1,"mirrored":true},{"x":1,"y":1,"in":1,"out":0,"mirrored":true}]'


class CurveDefinition:
    """Define a reusable curve visually. Outputs a CURVE that can be
    connected to CurveSample, CurveCFGGuider, or any other node
    accepting a CURVE input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "curve_data": ("STRING", {"default": DEFAULT_CURVE, "multiline": False}),
            }
        }

    RETURN_TYPES = ("CURVE",)
    FUNCTION = "define"
    CATEGORY = "CCN"

    def define(self, curve_data):
        try:
            keys = json.loads(curve_data)
        except (json.JSONDecodeError, TypeError):
            keys = json.loads(DEFAULT_CURVE)

        keys.sort(key=lambda k: k["x"])
        return (keys,)
