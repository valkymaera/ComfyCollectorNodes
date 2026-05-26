"""
String List Slicer node - extracts an item from a delimited string by index
"""

class StringListSlicer:
    """
    Extracts a single item from a delimited string at the specified index.
    
    Takes a string containing multiple values separated by a delimiter,
    splits it, and returns the item at the given index.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"default": "", "multiline": True}),
                "index": ("INT", {"default": 0, "min": -999999, "max": 999999, "step": 1}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": ","}),
                "strip_whitespace": ("BOOLEAN", {"default": True}),
                "default_value": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("item", "list_length")
    FUNCTION = "slice"
    CATEGORY = "CCN/String"

    def slice(self, input_string, index, delimiter=",", strip_whitespace=True, default_value=""):
        if not input_string:
            return (default_value, 0)
        
        # Split the string by delimiter
        items = input_string.split(delimiter)
        
        # Optionally strip whitespace from each item
        if strip_whitespace:
            items = [item.strip() for item in items]
        
        # Filter out empty items if stripping whitespace
        if strip_whitespace:
            items = [item for item in items if item]
        
        list_length = len(items)
        
        if list_length == 0:
            return (default_value, 0)
        
        # Handle negative indices (Python-style)
        # Also handle out-of-bounds by wrapping or returning default
        try:
            if index < 0:
                # Python-style negative indexing
                actual_index = list_length + index
                if actual_index < 0:
                    return (default_value, list_length)
            else:
                actual_index = index
            
            if actual_index >= list_length:
                return (default_value, list_length)
            
            return (items[actual_index], list_length)
        except (IndexError, ValueError):
            return (default_value, list_length)
