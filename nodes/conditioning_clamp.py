"""
Conditioning Clamp - Clamp conditioning tensor values to a min/max range
"""

import torch


class ConditioningClamp:
    """
    Clamps conditioning tensor values to a min/max range.
    Can help reduce extreme values that cause artifacts.
    """
    
    CATEGORY = "ComfyCollectorNodes/Conditioning"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "min_value": ("FLOAT", {"default": -4.0, "min": -20.0, "max": 0.0, "step": 0.1}),
                "max_value": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "clamp_conditioning"

    def clamp_conditioning(self, conditioning, min_value, max_value):
        out = []
        for cond_tuple in conditioning:
            cond = cond_tuple[0].clone()
            pooled = cond_tuple[1].copy() if len(cond_tuple) > 1 else {}
            
            original_min = cond.min().item()
            original_max = cond.max().item()
            
            cond = torch.clamp(cond, min_value, max_value)
            
            new_min = cond.min().item()
            new_max = cond.max().item()
            
            out.append((cond, pooled))
        
        print(f"[ComfyCollectorNodes] Conditioning clamped: [{original_min:.2f}, {original_max:.2f}] -> [{new_min:.2f}, {new_max:.2f}]")
        
        return (out,)
