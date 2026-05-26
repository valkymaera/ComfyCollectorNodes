"""
Image Loader By Index - Load images by their position in a folder
"""

import os
import numpy as np
import torch
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif')


def find_files(directory, extensions, recursive):
    """Collect files matching extensions from a directory, sorted alphabetically."""
    if recursive:
        files = []
        for root, _, filenames in os.walk(directory):
            for f in filenames:
                if f.lower().endswith(extensions):
                    rel_path = os.path.relpath(os.path.join(root, f), directory)
                    files.append(rel_path)
        return sorted(files)
    else:
        return sorted([
            f for f in os.listdir(directory)
            if f.lower().endswith(extensions)
        ])


def resolve_index(index, total, label="files"):
    """Wrap index if it exceeds total. Returns (actual_index, wrapped)."""
    wrapped = index >= total
    actual_index = index % total
    return actual_index, wrapped


class ImageLoaderByIndex:
    """
    Loads an image from a directory based on its index position.
    Useful for iterating through a collection of reference images in I2V workflows.
    Automatically wraps index if it exceeds the number of available files.
    """

    CATEGORY = "ComfyCollectorNodes/Loaders"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "placeholder": "/path/to/images"}),
                "recursive": ("BOOLEAN", {"default": False}),
                "index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("image", "mask", "filename", "file_path", "total_files", "actual_index", "wrapped")
    FUNCTION = "load_image_by_index"

    def load_image_by_index(self, directory, recursive, index, debug=False):
        directory = directory.strip()
        if not directory:
            raise ValueError("No directory specified")
        if not os.path.isdir(directory):
            raise ValueError(f"Directory not found: {directory}")

        image_files = find_files(directory, IMAGE_EXTENSIONS, recursive)
        if not image_files:
            raise ValueError(f"No image files found in {directory}")

        total_files = len(image_files)
        actual_index, wrapped = resolve_index(index, total_files)

        if debug:
            if wrapped:
                print(f"[ComfyCollectorNodes] Index {index} exceeds {total_files} images, wrapping to index {actual_index}")
            print(f"[ComfyCollectorNodes] Loading image {actual_index + 1}/{total_files}: {image_files[actual_index]}")

        filename = image_files[actual_index]
        file_path = os.path.join(directory, filename)

        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img)

        # Separate alpha channel as mask if present
        if img.mode == "RGBA":
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - mask  # ComfyUI convention: 0 = keep, 1 = masked
            img = img.convert("RGB")
        else:
            img = img.convert("RGB")
            mask = np.zeros((img.height, img.width), dtype=np.float32)

        # Convert to ComfyUI IMAGE format: (B, H, W, C) float32 [0, 1]
        image_np = np.array(img).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        if debug:
            print(f"[ComfyCollectorNodes] Image loaded: {img.width}x{img.height}, output shape: {list(image_tensor.shape)}")

        return (image_tensor, mask_tensor, filename, file_path, total_files, actual_index, wrapped)

    @classmethod
    def IS_CHANGED(cls, directory, recursive, index, debug=False):
        directory = directory.strip()
        if not directory or not os.path.isdir(directory):
            return ""
        image_files = find_files(directory, IMAGE_EXTENSIONS, recursive)
        if not image_files:
            return ""
        total = len(image_files)
        actual_index = index % total
        file_path = os.path.join(directory, image_files[actual_index])
        return f"{file_path}:{os.path.getmtime(file_path)}"
