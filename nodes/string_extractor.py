"""
String Extractor node - extracts text segments using bookend strings
"""

class StringExtractor:
    """
    Extracts text segments from a string using two bookend strings.
    
    Splits the input into three parts:
    - Before segment: all text up to and including the start bookend
    - Middle segment: all text between the bookends (exclusive)
    - After segment: all text from the end bookend onwards (inclusive)
    
    Wide bookends mode:
    - When False: uses FIRST start bookend and LAST end bookend (middle may contain bookend strings)
    - When True: uses LAST start bookend and FIRST end bookend (middle guaranteed clean)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"default": "", "multiline": True}),
                "start_bookend": ("STRING", {"default": ""}),
                "end_bookend": ("STRING", {"default": ""}),
            },
            "optional": {
                "wide_bookends": ("BOOLEAN", {"default": False}),
                "case_sensitive": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("before", "middle", "after")
    FUNCTION = "extract"
    CATEGORY = "CCN/String"

    def extract(self, input_string, start_bookend, end_bookend, wide_bookends=False, case_sensitive=True):
        if not input_string:
            return ("", "", "")
        
        if not start_bookend and not end_bookend:
            return ("", input_string, "")
        
        search_string = input_string if case_sensitive else input_string.lower()
        search_start = start_bookend if case_sensitive else start_bookend.lower()
        search_end = end_bookend if case_sensitive else end_bookend.lower()
        
        # Find bookend positions based on wide_bookends mode
        if wide_bookends:
            # Wide mode: LAST start bookend, FIRST end bookend (squeeze the middle)
            start_idx = search_string.rfind(search_start) if start_bookend else -1
            end_idx = search_string.find(search_end) if end_bookend else -1
        else:
            # Normal mode: FIRST start bookend, LAST end bookend (expand the middle)
            start_idx = search_string.find(search_start) if start_bookend else -1
            end_idx = search_string.rfind(search_end) if end_bookend else -1
        
        # Handle cases where bookends are not found
        if start_idx == -1 and end_idx == -1:
            # Neither bookend found - return entire string as middle
            return ("", input_string, "")
        
        if start_idx == -1:
            # Only end bookend found
            before = ""
            middle = input_string[:end_idx].strip()
            after = input_string[end_idx:]
            return (before, middle, after)
        
        if end_idx == -1:
            # Only start bookend found
            before_end = start_idx + len(start_bookend)
            before = input_string[:before_end]
            middle = input_string[before_end:].strip()
            after = ""
            return (before, middle, after)
        
        # Both bookends found
        before_end = start_idx + len(start_bookend)
        
        # Calculate segments
        before = input_string[:before_end]
        after = input_string[end_idx:]
        
        # Check for overlap (end bookend starts before start bookend ends)
        if end_idx < before_end:
            # Bookends overlap - middle is empty, but bookends don't interfere
            middle = ""
        else:
            middle = input_string[before_end:end_idx].strip()
        
        return (before, middle, after)
