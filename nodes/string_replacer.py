"""
String Replacer - Find and replace multiple patterns in a string
"""


class StringReplacer:
    """
    Replace multiple patterns in a string.
    
    Format for replacements:
      "find,replace; find2,replace2; find3,replace3"
    
    Example:
      Input: "The red cat sat on the red mat"
      Replacements: "red,blue; cat,dog"
      Output: "The blue dog sat on the blue mat"
    
    Whitespace around find/replace values is trimmed.
    Empty replacements are allowed (to delete text).
    """
    
    CATEGORY = "ComfyCollectorNodes/String"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "replacements": ("STRING", {"default": "", "multiline": True}),
                "case_sensitive": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text", "replacement_count")
    FUNCTION = "replace"

    def replace(self, text, replacements, case_sensitive, debug):
        if not replacements.strip():
            return (text, 0)
        
        total_replacements = 0
        result = text
        
        # Parse replacement pairs
        pairs = replacements.split(";")
        
        for pair in pairs:
            pair = pair.strip()
            if not pair:
                continue
            
            # Split on first comma only (in case replacement contains commas)
            parts = pair.split(",", 1)
            if len(parts) != 2:
                if debug:
                    print(f"[ComfyCollectorNodes] StringReplacer: Skipping invalid pair: '{pair}'")
                continue
            
            find_str = parts[0].strip()
            replace_str = parts[1].strip()
            
            if not find_str:
                if debug:
                    print(f"[ComfyCollectorNodes] StringReplacer: Skipping empty find string")
                continue
            
            # Count and replace
            if case_sensitive:
                count = result.count(find_str)
                result = result.replace(find_str, replace_str)
            else:
                # Case-insensitive replacement
                import re
                pattern = re.compile(re.escape(find_str), re.IGNORECASE)
                count = len(pattern.findall(result))
                result = pattern.sub(replace_str, result)
            
            total_replacements += count
            
            if debug and count > 0:
                print(f"[ComfyCollectorNodes] StringReplacer: '{find_str}' -> '{replace_str}' ({count} replacements)")
        
        if debug:
            print(f"[ComfyCollectorNodes] StringReplacer: Total replacements: {total_replacements}")
        
        return (result, total_replacements)
