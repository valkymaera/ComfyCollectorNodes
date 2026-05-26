"""
Image/Video Resize utilities - robust handling of various formats
"""

import torch
import torch.nn.functional as F


class ResizeByShorterEdge:
    """
    Resize images/video frames so the shorter edge matches target size.
    Maintains aspect ratio.
    
    Handles various input formats robustly:
    - Standard ComfyUI IMAGE (B, H, W, C) float32
    - Video frames in various formats
    - Converts uint8 to float32 if needed
    """
    
    CATEGORY = "ComfyCollectorNodes/Image"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "target_size": ("INT", {"default": 1080, "min": 64, "max": 8192, "step": 8}),
                "interpolation": (["bilinear", "bicubic", "nearest", "area"], {"default": "bilinear"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "resize"

    def resize(self, images, target_size, interpolation):
        # Debug info
        print(f"[CCN Resize] Input shape: {images.shape}, dtype: {images.dtype}")
        
        # Ensure float32
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
            print(f"[CCN Resize] Converted uint8 to float32")
        elif images.dtype != torch.float32:
            images = images.float()
        
        # Expected shape: (B, H, W, C) for ComfyUI
        # PyTorch interpolate expects: (B, C, H, W)
        
        if len(images.shape) == 4:
            b, h, w, c = images.shape
            
            # Sanity check - if h or w is very small, dimensions might be swapped
            if h <= 4 and c > 4:
                print(f"[CCN Resize] Warning: Height={h} seems wrong, attempting to fix...")
                # Might be (B, C, H, W) already
                images = images.permute(0, 2, 3, 1)
                b, h, w, c = images.shape
                print(f"[CCN Resize] Reshaped to: {images.shape}")
            
        elif len(images.shape) == 3:
            # Single image (H, W, C) - add batch dim
            images = images.unsqueeze(0)
            b, h, w, c = images.shape
        else:
            raise ValueError(f"Unexpected image shape: {images.shape}")
        
        print(f"[CCN Resize] Processing: {b} images, {h}x{w}, {c} channels")
        
        # Calculate new dimensions based on shorter edge
        if h < w:
            # Height is shorter
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            # Width is shorter (or square)
            new_w = target_size
            new_h = int(h * (target_size / w))
        
        # Make dimensions divisible by 8 (helpful for models)
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8
        
        print(f"[CCN Resize] Resizing: {h}x{w} -> {new_h}x{new_w}")
        
        # Convert to (B, C, H, W) for interpolation
        images = images.permute(0, 3, 1, 2)
        
        # Resize
        mode = interpolation
        if mode == "area" and (new_h > h or new_w > w):
            # Area mode only works for downscaling, use bilinear for upscaling
            mode = "bilinear"
        
        align_corners = None if mode in ["nearest", "area"] else False
        
        resized = F.interpolate(
            images, 
            size=(new_h, new_w), 
            mode=mode,
            align_corners=align_corners
        )
        
        # Convert back to (B, H, W, C)
        resized = resized.permute(0, 2, 3, 1)
        
        # Clamp to valid range
        resized = torch.clamp(resized, 0.0, 1.0)
        
        print(f"[CCN Resize] Output shape: {resized.shape}")
        
        return (resized,)


class ResizeToMatch:
    """
    Resize images to match the dimensions of a reference image.
    Useful for blending upscaled video with original.
    """
    
    CATEGORY = "ComfyCollectorNodes/Image"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference": ("IMAGE",),
                "interpolation": (["bilinear", "bicubic", "nearest", "area"], {"default": "bilinear"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "resize"

    def resize(self, images, reference, interpolation):
        # Get reference dimensions
        if len(reference.shape) == 4:
            _, ref_h, ref_w, _ = reference.shape
        else:
            ref_h, ref_w = reference.shape[0], reference.shape[1]
        
        print(f"[CCN ResizeToMatch] Input: {images.shape} -> Target: {ref_h}x{ref_w}")
        
        # Ensure float32
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.dtype != torch.float32:
            images = images.float()
        
        # Handle shape
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        
        b, h, w, c = images.shape
        
        # Convert to (B, C, H, W) for interpolation
        images = images.permute(0, 3, 1, 2)
        
        # Resize
        mode = interpolation
        if mode == "area" and (ref_h > h or ref_w > w):
            mode = "bilinear"
        
        align_corners = None if mode in ["nearest", "area"] else False
        
        resized = F.interpolate(
            images,
            size=(ref_h, ref_w),
            mode=mode,
            align_corners=align_corners
        )
        
        # Convert back to (B, H, W, C)
        resized = resized.permute(0, 2, 3, 1)
        resized = torch.clamp(resized, 0.0, 1.0)
        
        print(f"[CCN ResizeToMatch] Output: {resized.shape}")
        
        return (resized,)


class ImageBlend:
    """
    Blend two image sets together.
    Useful for mixing upscaled video with original for color preservation.
    
    Formula: result = image_a * (1 - blend) + image_b * blend
    """
    
    CATEGORY = "ComfyCollectorNodes/Image"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blend"

    def blend(self, image_a, image_b, blend):
        # Ensure same dtype
        if image_a.dtype != torch.float32:
            image_a = image_a.float() / 255.0 if image_a.dtype == torch.uint8 else image_a.float()
        if image_b.dtype != torch.float32:
            image_b = image_b.float() / 255.0 if image_b.dtype == torch.uint8 else image_b.float()
        
        # Handle batch size mismatch
        if image_a.shape[0] != image_b.shape[0]:
            min_batch = min(image_a.shape[0], image_b.shape[0])
            image_a = image_a[:min_batch]
            image_b = image_b[:min_batch]
            print(f"[CCN ImageBlend] Batch size mismatch, truncated to {min_batch}")
        
        # Handle resolution mismatch by resizing image_b to match image_a
        if image_a.shape[1:3] != image_b.shape[1:3]:
            print(f"[CCN ImageBlend] Resolution mismatch: {image_a.shape} vs {image_b.shape}")
            # Resize image_b to match image_a
            target_h, target_w = image_a.shape[1], image_a.shape[2]
            image_b = image_b.permute(0, 3, 1, 2)
            image_b = F.interpolate(image_b, size=(target_h, target_w), mode="bilinear", align_corners=False)
            image_b = image_b.permute(0, 2, 3, 1)
            print(f"[CCN ImageBlend] Resized image_b to {image_b.shape}")
        
        # Blend
        result = image_a * (1.0 - blend) + image_b * blend
        result = torch.clamp(result, 0.0, 1.0)
        
        return (result,)
