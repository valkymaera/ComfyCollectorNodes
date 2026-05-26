"""
Prompt Builder - Structured prompt construction with labeled sections
"""


class PromptBuilder:
    """
    Builds prompts from labeled sections.
    Empty sections are skipped.
    
    Sections:
      - Metadata: tablesetting & summary
      - Features: cast/subjects
      - Scene: camera, themes, setting
      - Details: finer movement, actions, events
      - Feedback: comment/note-style flavor
    """
    
    CATEGORY = "ComfyCollectorNodes/Prompt"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delimiter": ("STRING", {"default": "\\n\\n"}),
                "prefix_sections": ("BOOLEAN", {"default": False}),
                "metadata": ("STRING", {"default": "", "multiline": True}),
                "features": ("STRING", {"default": "", "multiline": True}),
                "scene": ("STRING", {"default": "", "multiline": True}),
                "details": ("STRING", {"default": "", "multiline": True}),
                "feedback": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build_prompt"

    def build_prompt(self, delimiter, prefix_sections, metadata, features, scene, details, feedback):
        # Handle escaped characters
        delimiter = delimiter.replace("\\n", "\n").replace("\\t", "\t")
        
        # Section names and values
        sections = [
            ("metadata", metadata),
            ("features", features),
            ("scene", scene),
            ("details", details),
            ("feedback", feedback),
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
