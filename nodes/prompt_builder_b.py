"""
Prompt Builder B - Structured prompt construction with alternate labels
"""


class PromptBuilderB:
    """
    Builds prompts from labeled sections (alternate labels).
    Empty sections are skipped.
    
    Sections:
      - Quality: technical quality descriptors
      - Style: artistic style
      - Mood: emotional tone
      - Motion: movement descriptors
      - General: catch-all
    """
    
    CATEGORY = "ComfyCollectorNodes/Prompt"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delimiter": ("STRING", {"default": "\\n\\n"}),
                "prefix_sections": ("BOOLEAN", {"default": False}),
                "quality": ("STRING", {"default": "", "multiline": True}),
                "style": ("STRING", {"default": "", "multiline": True}),
                "mood": ("STRING", {"default": "", "multiline": True}),
                "motion": ("STRING", {"default": "", "multiline": True}),
                "general": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build_prompt"

    def build_prompt(self, delimiter, prefix_sections, quality, style, mood, motion, general):
        # Handle escaped characters
        delimiter = delimiter.replace("\\n", "\n").replace("\\t", "\t")
        
        # Section names and values
        sections = [
            ("quality", quality),
            ("style", style),
            ("mood", mood),
            ("motion", motion),
            ("general", general),
        ]
        
        # Collect non-empty sections
        parts = []
        for name, value in sections:
            if value and value.strip():
                if prefix_sections:
                    parts.append(f"{name}: {value.strip()}")
                else:
                    parts.append(value.strip())
        
        result = delimiter.join(parts)
        
        return (result,)
