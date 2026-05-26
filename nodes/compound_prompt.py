"""
Compound Prompt mode selector.
Cycles through Off / Temporal / Split / Neutral and outputs three bools.
"""


class CompoundPrompt:

    MODES = ["Off", "Temporal", "Split", "Neutral"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (cls.MODES, {"default": "Off"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("temporal", "split", "neutral")
    FUNCTION = "execute"
    CATEGORY = "CCN"

    def execute(self, mode):
        return (mode == "Temporal", mode == "Split", mode == "Neutral")
