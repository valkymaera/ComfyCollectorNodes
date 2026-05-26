"""
LoRA Loader By Index - Load LoRAs by their position in a folder
"""

import os
import folder_paths
import comfy.sd
import comfy.utils


class LoraLoaderByIndex:
    """
    Loads a LoRA from a directory based on its index position.
    Useful for iterating through a collection of LoRAs.
    Automatically wraps index if it exceeds the number of available LoRAs.
    
    Searches all configured LoRA paths including extra_model_paths.yaml entries.
    """
    
    CATEGORY = "ComfyCollectorNodes/Loaders"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "subdirectory": ("STRING", {"default": "", "placeholder": "e.g. new_loras or wan/characters"}),
                "recursive": ("BOOLEAN", {"default": False}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("model", "clip", "lora_name", "total_loras", "actual_index", "wrapped")
    FUNCTION = "load_lora_by_index"

    def load_lora_by_index(self, model, subdirectory, recursive, index, strength_model, strength_clip, clip=None):
        lora_extensions = ('.safetensors', '.pt', '.bin', '.ckpt')
        
        # Get all lora directories (includes extra_model_paths.yaml entries)
        lora_paths = folder_paths.get_folder_paths("loras")
        
        # Find the target directory across all paths
        lora_dir = None
        lora_base = None
        
        for base_path in lora_paths:
            if subdirectory.strip():
                candidate = os.path.join(base_path, subdirectory.strip())
            else:
                candidate = base_path
            
            if os.path.isdir(candidate):
                lora_dir = candidate
                lora_base = base_path
                break
        
        if lora_dir is None:
            searched = [os.path.join(p, subdirectory.strip()) if subdirectory.strip() else p for p in lora_paths]
            raise ValueError(f"Directory not found. Searched: {searched}")
        
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
        
        if not lora_files:
            raise ValueError(f"No LoRA files found in {lora_dir}")
        
        total_loras = len(lora_files)
        original_index = index
        
        # Check if wrapping is needed
        wrapped = index >= total_loras
        actual_index = index % total_loras
        
        if wrapped:
            if actual_index == 0:
                print(f"\n{'='*60}")
                print(f"[ComfyCollectorNodes] *** LIST COMPLETE - WRAPPING TO START ***")
                print(f"[ComfyCollectorNodes] Index {original_index} exceeds {total_loras} LoRAs")
                print(f"[ComfyCollectorNodes] Wrapping to index 0")
                print(f"{'='*60}\n")
            else:
                print(f"[ComfyCollectorNodes] Index {original_index} exceeds {total_loras} LoRAs, wrapping to index {actual_index}")
        
        lora_filename = lora_files[actual_index]
        
        if subdirectory.strip():
            lora_name = os.path.join(subdirectory.strip(), lora_filename)
        else:
            lora_name = lora_filename
            
        lora_path = os.path.join(lora_base, lora_name)
        
        print(f"[ComfyCollectorNodes] Loading LoRA {actual_index + 1}/{total_loras}: {lora_name}")
        
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        # Handle optional clip
        if clip is not None:
            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model, clip, lora, strength_model, strength_clip
            )
        else:
            model_lora, _ = comfy.sd.load_lora_for_models(
                model, None, lora, strength_model, 0
            )
            clip_lora = None
        
        return (model_lora, clip_lora, lora_name, total_loras, actual_index, wrapped)
