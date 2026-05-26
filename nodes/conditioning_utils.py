"""
Conditioning utilities - Lerp, Subtract, and more
"""

import torch
import random


class ConditioningLerp:
    """
    Linear interpolation between two conditionings.
    
    blend = 0.0 → 100% conditioning_a
    blend = 0.5 → 50/50 mix
    blend = 1.0 → 100% conditioning_b
    
    Formula: result = a * (1 - blend) + b * blend
    """
    
    CATEGORY = "ComfyCollectorNodes/Conditioning"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_a": ("CONDITIONING",),
                "conditioning_b": ("CONDITIONING",),
                "blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "lerp"

    def lerp(self, conditioning_a, conditioning_b, blend):
        # Both conditionings should have same structure
        result = []
        
        for i in range(min(len(conditioning_a), len(conditioning_b))):
            cond_a = conditioning_a[i][0]
            cond_b = conditioning_b[i][0]
            pooled_a = conditioning_a[i][1].copy() if len(conditioning_a[i]) > 1 else {}
            pooled_b = conditioning_b[i][1] if len(conditioning_b[i]) > 1 else {}
            
            # Lerp the main conditioning tensor
            # Handle different sequence lengths by truncating to shorter
            min_len = min(cond_a.shape[1], cond_b.shape[1])
            cond_a_trim = cond_a[:, :min_len, :]
            cond_b_trim = cond_b[:, :min_len, :]
            
            lerped = cond_a_trim * (1.0 - blend) + cond_b_trim * blend
            
            # Lerp pooled outputs if both have them
            result_pooled = pooled_a.copy()
            if "pooled_output" in pooled_a and "pooled_output" in pooled_b:
                pooled_a_tensor = pooled_a["pooled_output"]
                pooled_b_tensor = pooled_b["pooled_output"]
                if pooled_a_tensor is not None and pooled_b_tensor is not None:
                    result_pooled["pooled_output"] = pooled_a_tensor * (1.0 - blend) + pooled_b_tensor * blend
            
            result.append([lerped, result_pooled])
        
        return (result,)


class ConditioningSubtract:
    """
    Subtract one conditioning from another.
    
    Useful for conceptual removal:
      full_scene - "snow" = scene without snow concept
    
    Formula: result = conditioning_a - conditioning_b * strength
    """
    
    CATEGORY = "ComfyCollectorNodes/Conditioning"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_a": ("CONDITIONING",),
                "conditioning_b": ("CONDITIONING",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "subtract"

    def subtract(self, conditioning_a, conditioning_b, strength):
        result = []
        
        for i in range(min(len(conditioning_a), len(conditioning_b))):
            cond_a = conditioning_a[i][0]
            cond_b = conditioning_b[i][0]
            pooled_a = conditioning_a[i][1].copy() if len(conditioning_a[i]) > 1 else {}
            pooled_b = conditioning_b[i][1] if len(conditioning_b[i]) > 1 else {}
            
            # Handle different sequence lengths
            min_len = min(cond_a.shape[1], cond_b.shape[1])
            cond_a_trim = cond_a[:, :min_len, :]
            cond_b_trim = cond_b[:, :min_len, :]
            
            subtracted = cond_a_trim - cond_b_trim * strength
            
            # Subtract pooled outputs if both have them
            result_pooled = pooled_a.copy()
            if "pooled_output" in pooled_a and "pooled_output" in pooled_b:
                pooled_a_tensor = pooled_a["pooled_output"]
                pooled_b_tensor = pooled_b["pooled_output"]
                if pooled_a_tensor is not None and pooled_b_tensor is not None:
                    result_pooled["pooled_output"] = pooled_a_tensor - pooled_b_tensor * strength
            
            result.append([subtracted, result_pooled])
        
        return (result,)


class RandomSelect:
    """
    Randomly select one of up to 5 inputs.
    
    Only connected inputs are considered.
    Re-rolls each execution unless seed is set.
    """
    
    CATEGORY = "ComfyCollectorNodes/Utils"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "input_1": ("*",),
                "input_2": ("*",),
                "input_3": ("*",),
                "input_4": ("*",),
                "input_5": ("*",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF}),
            },
        }

    RETURN_TYPES = ("*", "INT")
    RETURN_NAMES = ("output", "selected_index")
    FUNCTION = "select"

    def select(self, input_1=None, input_2=None, input_3=None, input_4=None, input_5=None, seed=-1):
        # Collect non-None inputs with their indices
        inputs = []
        for i, inp in enumerate([input_1, input_2, input_3, input_4, input_5], 1):
            if inp is not None:
                inputs.append((i, inp))
        
        if not inputs:
            print("[ComfyCollectorNodes] RandomSelect: No inputs connected!")
            return (None, 0)
        
        # Set seed if specified
        if seed >= 0:
            random.seed(seed)
        
        # Pick randomly
        selected_idx, selected_value = random.choice(inputs)
        
        print(f"[ComfyCollectorNodes] RandomSelect: Picked input_{selected_idx} (of {len(inputs)} connected)")
        
        return (selected_value, selected_idx)
