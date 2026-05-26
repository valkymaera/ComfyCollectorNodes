"""
Latent Utils - Manipulate latent tensors (clamp, normalize, scale)
"""

import torch


class LatentClamp:
    """
    Clamps latent values to a min/max range.
    Useful for reducing 'burn' artifacts from extreme values.
    """
    
    CATEGORY = "ComfyCollectorNodes/Latent"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "min_value": ("FLOAT", {"default": -4.0, "min": -20.0, "max": 0.0, "step": 0.1}),
                "max_value": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "clamp_latent"

    def clamp_latent(self, latent, min_value, max_value):
        samples = latent["samples"].clone()
        
        original_min = samples.min().item()
        original_max = samples.max().item()
        
        samples = torch.clamp(samples, min_value, max_value)
        
        new_min = samples.min().item()
        new_max = samples.max().item()
        
        print(f"[ComfyCollectorNodes] Latent clamped: [{original_min:.2f}, {original_max:.2f}] -> [{new_min:.2f}, {new_max:.2f}]")
        
        return ({"samples": samples},)


class LatentScale:
    """
    Scales latent values by a multiplier.
    """
    
    CATEGORY = "ComfyCollectorNodes/Latent"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "scale_latent"

    def scale_latent(self, latent, scale):
        if scale == 1.0:
            return (latent,)
            
        samples = latent["samples"].clone()
        samples = samples * scale
        
        print(f"[ComfyCollectorNodes] Latent scaled by {scale}x")
        
        return ({"samples": samples},)


class LatentNormalize:
    """
    Normalizes latent values using various methods.
    Can help reduce artifacts or balance latent distributions.
    
    Methods match those in Conditioning Normalizer for consistency.
    
    per_channel: When enabled, normalizes each channel independently
    instead of across the whole tensor.
    """
    
    CATEGORY = "ComfyCollectorNodes/Latent"
    
    METHODS = [
        "none",
        "max_norm",      # Divide by max absolute value
        "std_norm",      # Divide by standard deviation
        "std_half",      # Gentler std normalization (divide by std*2)
        "zscore",        # Full z-score: subtract mean, divide by std
        "zscore_avg",    # Average of z-score and max normalization
        "zscore_half",   # Gentler z-score (divide by std*2)
        "slight_z",      # 20% z-score, 80% max norm
        "mean_restore",  # Normalize but restore original mean
        "range",         # Scale to [-1, 1] range
        "clamp_1",       # Clamp values to [-1, 1]
        "clamp_1.5",     # Clamp values to [-1.5, 1.5]
        "clamp_2",       # Clamp values to [-2, 2]
        "clamp_3",       # Clamp values to [-3, 3]
        "clamp_4",       # Clamp values to [-4, 4]
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "method": (cls.METHODS, {"default": "none"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.0001}),
                "per_channel": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "normalize_latent"

    def _normalize_tensor(self, samples, method):
        """Apply normalization to a tensor (whole or single channel)."""
        if method == "max_norm":
            max_val = samples.abs().max()
            if max_val > 0:
                samples = samples / max_val
                
        elif method == "std_norm":
            std = samples.std()
            if std > 0:
                samples = samples / std
                
        elif method == "std_half":
            std = samples.std()
            if std > 0:
                samples = samples / (std * 2.0)
                
        elif method == "zscore":
            mean = samples.mean()
            std = samples.std()
            if std > 0:
                samples = (samples - mean) / std
            else:
                samples = samples - mean
                
        elif method == "zscore_avg":
            max_val = samples.abs().max()
            mean = samples.mean()
            std = samples.std()
            if std > 0 and max_val > 0:
                z_normed = (samples - mean) / std
                max_normed = samples / max_val
                samples = (z_normed + max_normed) / 2.0
                
        elif method == "zscore_half":
            mean = samples.mean()
            std = samples.std()
            if std > 0:
                samples = (samples - mean) / (std * 2.0)
            else:
                samples = samples - mean
                
        elif method == "slight_z":
            max_val = samples.abs().max()
            mean = samples.mean()
            std = samples.std()
            if std > 0 and max_val > 0:
                z_normed = (samples - mean) / std
                max_normed = samples / max_val
                samples = (z_normed + (max_normed * 4.0)) / 5.0
                
        elif method == "mean_restore":
            original_mean = samples.mean()
            std = samples.std()
            if std > 0:
                samples = samples / std
                new_mean = samples.mean()
                if new_mean != 0:
                    samples = samples * (original_mean / new_mean)
                
        elif method == "range":
            min_val = samples.min()
            max_val = samples.max()
            range_val = max_val - min_val
            if range_val > 0:
                samples = 2 * (samples - min_val) / range_val - 1
                
        elif method == "clamp_1":
            samples = torch.clamp(samples, -1.0, 1.0)
            
        elif method == "clamp_1.5":
            samples = torch.clamp(samples, -1.5, 1.5)
            
        elif method == "clamp_2":
            samples = torch.clamp(samples, -2.0, 2.0)
            
        elif method == "clamp_3":
            samples = torch.clamp(samples, -3.0, 3.0)
            
        elif method == "clamp_4":
            samples = torch.clamp(samples, -4.0, 4.0)
        
        return samples

    def normalize_latent(self, latent, method, strength, per_channel):
        if method == "none" or strength == 0:
            return (latent,)
            
        samples = latent["samples"].clone()
        original = samples.clone()
        
        if per_channel:
            # Normalize each channel independently
            num_channels = samples.shape[1]
            for c in range(num_channels):
                samples[:, c, :, :] = self._normalize_tensor(samples[:, c, :, :], method)
        else:
            # Normalize whole tensor
            samples = self._normalize_tensor(samples, method)
        
        # Blend with original
        samples = original * (1.0 - strength) + samples * strength
        
        mode_str = "per-channel" if per_channel else "whole"
        print(f"[ComfyCollectorNodes] Latent normalized: method={method}, strength={strength}, mode={mode_str}")
        
        return ({"samples": samples},)


class LatentStats:
    """
    Prints statistics about a latent tensor.
    Useful for debugging and understanding latent values.
    Passes latent through unchanged.
    """
    
    CATEGORY = "ComfyCollectorNodes/Latent"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "label": ("STRING", {"default": "Latent"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "print_stats"

    def print_stats(self, latent, label):
        samples = latent["samples"]
        
        print(f"\n{'='*60}")
        print(f"[CCN Latent Stats] {label}")
        print(f"{'='*60}")
        print(f"  Shape: {list(samples.shape)}")
        print(f"  Min:   {samples.min().item():.4f}")
        print(f"  Max:   {samples.max().item():.4f}")
        print(f"  Mean:  {samples.mean().item():.4f}")
        print(f"  Std:   {samples.std().item():.4f}")
        print(f"{'='*60}\n")
        
        return (latent,)
