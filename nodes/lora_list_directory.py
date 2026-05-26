"""
LoRA List Directory - List all LoRAs in a directory
"""

import os
import folder_paths


class LoraListDirectory:
    """
    Lists all LoRA files in a directory.
    Useful for seeing what's available before iterating with LoRA Loader By Index.
    """
    
    CATEGORY = "ComfyCollectorNodes/Loaders"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subdirectory": ("STRING", {"default": "", "placeholder": "e.g. new_loras or wan/characters"}),
                "recursive": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("lora_names", "total_count")
    FUNCTION = "list_loras"

    def list_loras(self, subdirectory, recursive):
        lora_extensions = ('.safetensors', '.pt', '.bin', '.ckpt')
        
        # Get base lora directory
        lora_base = folder_paths.get_folder_paths("loras")[0]
        
        # Build target directory
        if subdirectory.strip():
            lora_dir = os.path.join(lora_base, subdirectory.strip())
        else:
            lora_dir = lora_base
        
        if not os.path.isdir(lora_dir):
            raise ValueError(f"Directory not found: {lora_dir}")
        
        # Find lora files
        if recursive:
            lora_files = []
            for root, _, files in os.walk(lora_dir):
                for f in files:
                    if f.lower().endswith(lora_extensions):
                        rel_path = os.path.relpath(os.path.join(root, f), lora_dir)
                        lora_files.append(rel_path)
            lora_files = sorted(lora_files)
        else:
            lora_files = sorted([
                f for f in os.listdir(lora_dir) 
                if f.lower().endswith(lora_extensions)
            ])
        
        total_count = len(lora_files)
        lora_names = "\n".join(lora_files)
        
        print(f"[ComfyCollectorNodes] Found {total_count} LoRAs in {lora_dir}")
        
        return (lora_names, total_count)
