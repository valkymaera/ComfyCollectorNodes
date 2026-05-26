"""
Conditioning Stats - Print statistics about conditioning tensor
"""

import torch


class ConditioningStats:
    """
    Prints statistics about a conditioning tensor.
    Useful for debugging and understanding conditioning values.
    Passes conditioning through unchanged.
    """
    
    CATEGORY = "ComfyCollectorNodes/Conditioning"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "label": ("STRING", {"default": "Conditioning"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "print_stats"

    def print_stats(self, conditioning, label):
        print(f"\n{'='*60}")
        print(f"[CCN Conditioning Stats] {label}")
        print(f"{'='*60}")
        
        for i, cond_tuple in enumerate(conditioning):
            cond = cond_tuple[0]
            pooled = cond_tuple[1] if len(cond_tuple) > 1 else {}
            
            print(f"  Cond[{i}] Shape: {list(cond.shape)}")
            print(f"  Cond[{i}] Min:   {cond.min().item():.4f}")
            print(f"  Cond[{i}] Max:   {cond.max().item():.4f}")
            print(f"  Cond[{i}] Mean:  {cond.mean().item():.4f}")
            print(f"  Cond[{i}] Std:   {cond.std().item():.4f}")
            
            if pooled:
                print(f"  Cond[{i}] Pooled keys: {list(pooled.keys())}")
        
        print(f"{'='*60}\n")
        
        return (conditioning,)
