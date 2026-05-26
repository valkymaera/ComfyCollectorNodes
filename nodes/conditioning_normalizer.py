"""
Conditioning Normalizer - Apply various normalization methods to conditioning
"""

import torch


class ConditioningNormalizer:
    """
    Applies normalization methods to conditioning tensors.
    Inspired by A1111 emphasis normalization techniques.
    
    These normalizations can subtly affect image generation even without
    explicit emphasis weights in your prompt, by changing the distribution
    of values in the conditioning tensor.
    """
    
    CATEGORY = "ComfyCollectorNodes/Conditioning"
    
    NORMALIZATION_METHODS = [
        "none",
        "max_norm",
        "std_norm", 
        "std_half",
        "zscore",
        "zscore_avg",
        "zscore_half",
        "slight_z",
        "mean_restore",
        "range",
        "clamp_1",
        "clamp_1.5",
        "clamp_2",
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "method": (cls.NORMALIZATION_METHODS, {"default": "none"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.0001}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "normalize"

    def normalize(self, conditioning, method, strength):
        if method == "none" or strength == 0:
            return (conditioning,)
        
        out = []
        for cond_tuple in conditioning:
            cond = cond_tuple[0].clone()
            pooled = cond_tuple[1].copy() if len(cond_tuple) > 1 else {}
            
            cond = self._apply_normalization(cond, method, strength)
            
            out.append((cond, pooled))
        
        return (out,)
    
    def _apply_normalization(self, z: torch.Tensor, method: str, strength: float) -> torch.Tensor:
        original = z.clone()
        
        if method == "max_norm":
            # Preserve maximum magnitude
            max_val = z.abs().max()
            if max_val > 0:
                z = z / max_val
                
        elif method == "std_norm":
            # Normalize by standard deviation
            std = z.std()
            if std > 0:
                z = z / std
                
        elif method == "std_half":
            # Gentler std normalization
            std = z.std()
            if std > 0:
                z = z / (std * 2.0)
                
        elif method == "zscore":
            # Full z-score normalization
            mean = z.mean()
            std = z.std()
            if std > 0:
                z = (z - mean) / std
            else:
                z = z - mean
                
        elif method == "zscore_avg":
            # Average of z-score and max normalization
            max_val = z.abs().max()
            mean = z.mean()
            std = z.std()
            if std > 0 and max_val > 0:
                z_normed = (z - mean) / std
                max_normed = z / max_val
                z = (z_normed + max_normed) / 2.0
                
        elif method == "zscore_half":
            # Gentler z-score normalization
            mean = z.mean()
            std = z.std()
            if std > 0:
                z = (z - mean) / (std * 2.0)
            else:
                z = z - mean
                
        elif method == "slight_z":
            # 20% z-score, 80% max norm
            max_val = z.abs().max()
            mean = z.mean()
            std = z.std()
            if std > 0 and max_val > 0:
                z_normed = (z - mean) / std
                max_normed = z / max_val
                z = (z_normed + (max_normed * 4.0)) / 5.0
                
        elif method == "mean_restore":
            # Normalize but restore original mean (like A1111's Original emphasis)
            original_mean = z.mean()
            std = z.std()
            if std > 0:
                z = z / std
                new_mean = z.mean()
                if new_mean != 0:
                    z = z * (original_mean / new_mean)
                    
        elif method == "range":
            # Scale to [-1, 1] range
            min_val = z.min()
            max_val = z.max()
            range_val = max_val - min_val
            if range_val > 0:
                z = 2 * (z - min_val) / range_val - 1
                    
        elif method == "clamp_1":
            # Clamp values to [-1, 1]
            z = torch.clamp(z, -1.0, 1.0)
            
        elif method == "clamp_1.5":
            # Clamp values to [-1.5, 1.5]
            z = torch.clamp(z, -1.5, 1.5)
            
        elif method == "clamp_2":
            # Clamp values to [-2, 2]
            z = torch.clamp(z, -2.0, 2.0)
        
        # Blend with original based on strength
        z = original * (1.0 - strength) + z * strength
        
        return z
