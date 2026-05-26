"""
Latent Channel Offset - Adjust individual latent channels
"""

import torch


class LatentChannelOffset:
    """
    Adjusts individual latent channels by adding offsets.
    
    SD 1.5/SDXL latents have 4 channels with rough correlations:
      - Channel 0: Cyan-Red axis, brightness/openness
      - Channel 1: Magenta-Green, structural elements
      - Channel 2: Yellow-Blue tones
      - Channel 3: Luminance/contrast
    
    Wan models use 16 channels with different (less documented) meanings.
    
    Positive values push toward one end of the spectrum,
    negative values push toward the other. Effects are abstract
    and semantic, not just color shifts.
    """
    
    CATEGORY = "ComfyCollectorNodes/Latent"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "channel_0": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "channel_1": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "channel_2": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "channel_3": ("FLOAT", {"default": 0.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "offset_channels"

    def offset_channels(self, latent, channel_0, channel_1, channel_2, channel_3):
        samples = latent["samples"].clone()
        num_channels = samples.shape[1]
        
        offsets = [channel_0, channel_1, channel_2, channel_3]
        
        # Apply offsets to available channels
        for i, offset in enumerate(offsets):
            if i < num_channels and offset != 0:
                samples[:, i, :, :] += offset
        
        applied = [f"ch{i}:{offsets[i]:+.2f}" for i in range(min(4, num_channels)) if offsets[i] != 0]
        if applied:
            print(f"[ComfyCollectorNodes] Latent channel offsets: {', '.join(applied)}")
        
        return ({"samples": samples},)


class LatentChannelOffset16:
    """
    Adjusts individual latent channels for models with 16 channels (e.g., Wan).
    
    Channel meanings in 16-channel latents are less documented.
    Experimentation encouraged!
    """
    
    CATEGORY = "ComfyCollectorNodes/Latent"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "latent": ("LATENT",),
            },
        }
        
        # Add 16 channel inputs
        for i in range(16):
            inputs["required"][f"ch_{i:02d}"] = ("FLOAT", {"default": 0.0, "step": 0.01})
        
        return inputs

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "offset_channels"

    def offset_channels(self, latent, **kwargs):
        samples = latent["samples"].clone()
        num_channels = samples.shape[1]
        
        applied = []
        for i in range(16):
            offset = kwargs.get(f"ch_{i:02d}", 0.0)
            if i < num_channels and offset != 0:
                samples[:, i, :, :] += offset
                applied.append(f"ch{i}:{offset:+.2f}")
        
        if applied:
            print(f"[ComfyCollectorNodes] Latent channel offsets: {', '.join(applied)}")
        
        return ({"samples": samples},)


class LatentChannelScale:
    """
    Scales individual latent channels by multipliers.
    
    Unlike offset (which adds), scale multiplies channel values.
    Values > 1 amplify, < 1 reduce, negative values invert.
    """
    
    CATEGORY = "ComfyCollectorNodes/Latent"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "channel_0": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "channel_1": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "channel_2": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "channel_3": ("FLOAT", {"default": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "scale_channels"

    def scale_channels(self, latent, channel_0, channel_1, channel_2, channel_3):
        samples = latent["samples"].clone()
        num_channels = samples.shape[1]
        
        scales = [channel_0, channel_1, channel_2, channel_3]
        
        # Apply scales to available channels
        for i, scale in enumerate(scales):
            if i < num_channels and scale != 1.0:
                samples[:, i, :, :] *= scale
        
        applied = [f"ch{i}:×{scales[i]:.2f}" for i in range(min(4, num_channels)) if scales[i] != 1.0]
        if applied:
            print(f"[ComfyCollectorNodes] Latent channel scales: {', '.join(applied)}")
        
        return ({"samples": samples},)


class LatentChannelScale16:
    """
    Scales individual latent channels for models with 16 channels (e.g., Wan).
    
    Unlike offset (which adds), scale multiplies channel values.
    Values > 1 amplify, < 1 reduce, negative values invert.
    """
    
    CATEGORY = "ComfyCollectorNodes/Latent"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "latent": ("LATENT",),
            },
        }
        
        # Add 16 channel inputs
        for i in range(16):
            inputs["required"][f"ch_{i:02d}"] = ("FLOAT", {"default": 1.0, "step": 0.01})
        
        return inputs

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "scale_channels"

    def scale_channels(self, latent, **kwargs):
        samples = latent["samples"].clone()
        num_channels = samples.shape[1]
        
        applied = []
        for i in range(16):
            scale = kwargs.get(f"ch_{i:02d}", 1.0)
            if i < num_channels and scale != 1.0:
                samples[:, i, :, :] *= scale
                applied.append(f"ch{i}:×{scale:.2f}")
        
        if applied:
            print(f"[ComfyCollectorNodes] Latent channel scales: {', '.join(applied)}")
        
        return ({"samples": samples},)
