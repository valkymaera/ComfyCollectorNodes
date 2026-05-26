"""
Multi LoRA Loader - Load multiple LoRAs with normalization options
"""

import os
import folder_paths
import comfy.sd
import comfy.utils


class MultiLoraLoaderBase:
    """
    Base class for Multi LoRA Loaders.
    """
    
    CATEGORY = "ComfyCollectorNodes/Loaders"
    MAX_LORAS = 4  # Override in subclasses
    
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        
        inputs = {
            "required": {
                "model": ("MODEL",),
                "normalize": ("BOOLEAN", {"default": False}),
                "strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            },
            "optional": {
                "clip": ("CLIP",),
            }
        }
        
        # Add lora slots
        for i in range(1, cls.MAX_LORAS + 1):
            inputs["optional"][f"lora_{i}"] = (lora_list, {"default": "None"})
            inputs["optional"][f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001})
        
        # Add debug at the end
        inputs["optional"]["debug"] = ("BOOLEAN", {"default": False})
        
        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "loaded_loras")
    FUNCTION = "load_loras"

    def load_loras(self, model, normalize, strength_multiplier, debug=False, clip=None, **kwargs):
        # Collect active loras (strength != 0 and not "None")
        active_loras = []
        for i in range(1, self.MAX_LORAS + 1):
            lora_name = kwargs.get(f"lora_{i}")
            strength = kwargs.get(f"strength_{i}")
            
            # Skip if not set
            if lora_name is None or strength is None:
                continue
            if not lora_name or lora_name == "None":
                continue
            # Skip if strength is zero (user's way to disable)
            if strength == 0:
                continue
                
            # Verify lora file exists
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None or not os.path.exists(lora_path):
                if debug:
                    print(f"[ComfyCollectorNodes] Warning: LoRA not found, skipping: {lora_name}")
                continue
                
            active_loras.append((lora_name, strength, lora_path))
        
        if not active_loras:
            if debug:
                print(f"[ComfyCollectorNodes] No LoRAs active (none selected or all have 0 strength)")
            return (model, clip, "None")
        
        # Calculate normalization factor if needed
        if normalize:
            total_strength = sum(abs(s) for _, s, _ in active_loras)
            if total_strength > 1.0:
                norm_factor = 1.0 / total_strength
                if debug:
                    print(f"[ComfyCollectorNodes] Normalizing: total strength {total_strength:.3f} -> factor {norm_factor:.3f}")
            else:
                norm_factor = 1.0
                if debug:
                    print(f"[ComfyCollectorNodes] Normalize enabled but total strength {total_strength:.3f} <= 1.0, no scaling needed")
        else:
            norm_factor = 1.0
        
        # Apply loras
        model_lora = model
        clip_lora = clip
        loaded_info = []
        
        for lora_name, strength, lora_path in active_loras:
            try:
                # Apply normalization and multiplier
                final_strength = strength * norm_factor * strength_multiplier
                
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                
                # Handle optional clip
                if clip_lora is not None:
                    model_lora, clip_lora = comfy.sd.load_lora_for_models(
                        model_lora, clip_lora, lora, final_strength, final_strength
                    )
                else:
                    model_lora, _ = comfy.sd.load_lora_for_models(
                        model_lora, None, lora, final_strength, 0
                    )
                
                loaded_info.append(f"{lora_name} @ {final_strength:.3f}")
                if debug:
                    print(f"[ComfyCollectorNodes] Loaded LoRA: {lora_name} (strength: {final_strength:.3f})")
                
            except Exception as e:
                if debug:
                    print(f"[ComfyCollectorNodes] Error loading LoRA {lora_name}: {e}")
                continue
        
        if not loaded_info:
            return (model, clip, "None (all failed to load)")
            
        return (model_lora, clip_lora, "\n".join(loaded_info))


class MultiLoraLoader4(MultiLoraLoaderBase):
    """Load up to 4 LoRAs at once."""
    MAX_LORAS = 4


class MultiLoraLoader8(MultiLoraLoaderBase):
    """Load up to 8 LoRAs at once."""
    MAX_LORAS = 8


class MultiLoraLoader12(MultiLoraLoaderBase):
    """Load up to 12 LoRAs at once."""
    MAX_LORAS = 12
