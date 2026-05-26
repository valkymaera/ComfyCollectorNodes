"""
Print - Print a custom string to console
"""


class Print:
    """
    Prints a custom string to the console when triggered.
    Connect any input to 'trigger' to control when it prints.
    """
    
    CATEGORY = "ComfyCollectorNodes/Utils"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Hello", "multiline": True}),
            },
            "optional": {
                "trigger": ("*",),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("trigger",)
    FUNCTION = "print_text"

    def print_text(self, text, trigger=None):
        print(f"[CCN] {text}")
        return (trigger,)

