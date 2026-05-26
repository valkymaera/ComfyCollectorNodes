"""
String Merge Unique - Merge comma-separated strings without duplicates
"""


class StringMergeUnique:
    """
    Merges up to 3 comma-separated strings, removing duplicates.
    Preserves order of first occurrence.
    """
    
    CATEGORY = "ComfyCollectorNodes/Utils"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_1": ("STRING", {"default": ""}),
            },
            "optional": {
                "string_2": ("STRING", {"default": ""}),
                "string_3": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "merge_unique"

    def merge_unique(self, string_1, string_2="", string_3=""):
        seen = set()
        result = []
        
        for s in [string_1, string_2, string_3]:
            if not s:
                continue
            for item in s.split(","):
                item = item.strip()
                if item and item not in seen:
                    seen.add(item)
                    result.append(item)
        
        return (", ".join(result),)
