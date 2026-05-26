"""
String Concatenate - Combine multiple strings with optional delimiter
"""


class StringConcatenate:
    """
    Concatenates up to 4 strings with an optional delimiter.
    Enter text directly or connect inputs to override.
    Empty strings are skipped.
    """
    
    CATEGORY = "ComfyCollectorNodes/Utils"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delimiter": ("STRING", {"default": ""}),
                "string_1": ("STRING", {"default": ""}),
                "string_2": ("STRING", {"default": ""}),
                "string_3": ("STRING", {"default": ""}),
                "string_4": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "concatenate"

    def concatenate(self, delimiter, string_1, string_2, string_3, string_4):
        # Collect non-empty strings
        parts = []
        for s in [string_1, string_2, string_3, string_4]:
            if s is not None and s != "":
                parts.append(s)
        
        # Handle escaped newlines in delimiter
        if delimiter == "\\n":
            delimiter = "\n"
        elif delimiter == "\\t":
            delimiter = "\t"
        
        result = delimiter.join(parts)
        
        return (result,)
