"""
String Splitter — Split a string by delimiter into up to 5 outputs.
"""


class StringSplitter:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
                "delimiter": ("STRING", {"default": ","}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("out_1", "out_2", "out_3", "out_4", "out_5")
    FUNCTION = "split"
    CATEGORY = "CCN"
    DESCRIPTION = (
        "Split a string by a delimiter into up to 5 outputs.  "
        "Unused outputs return empty strings."
    )

    def split(self, text, delimiter):
        parts = text.split(delimiter, maxsplit=4)
        parts = [p.strip() for p in parts]
        while len(parts) < 5:
            parts.append("")
        return tuple(parts[:5])
