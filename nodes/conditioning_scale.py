"""
Conditioning Scale - Multiply conditioning tensor to boost or reduce prompt influence
"""

import torch


class ConditioningScale:
    """
    Scales conditioning tensor by a multiplier.
    Higher values = stronger prompt influence.
    Lower values = weaker prompt influence.
    
    Unlike normalization, this simply amplifies/reduces magnitude.
    """
    
    CATEGORY = "ComfyCollectorNodes/Conditioning"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "scale_conditioning"

    def scale_conditioning(self, conditioning, scale):
        if scale == 1.0:
            return (conditioning,)
        
        out = []
        for cond_tuple in conditioning:
            cond = cond_tuple[0].clone()
            pooled = cond_tuple[1].copy() if len(cond_tuple) > 1 else {}
            
            # Scale the conditioning tensor
            cond = cond * scale
            
            out.append((cond, pooled))
        
        print(f"[ComfyCollectorNodes] Conditioning scaled by {scale}x")
        return (out,)
