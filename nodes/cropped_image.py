"""
Cropped Image — Visual image cropping with interactive rectangle overlay.

Provides a canvas widget where users can drag corners to define a crop
region.  Outputs both a model-friendly image (dimensions floored to a
snap multiple) and the raw pixel crop.  Can also function as a standalone
image loader when no upstream image is wired.
"""

import os
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import folder_paths


class CroppedImage:
    """Visual crop node with interactive JS rectangle overlay and optional image loading."""

    CATEGORY = "ComfyCollectorNodes/Image"

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.isdir(input_dir):
            files = sorted([
                f for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f))
                and f.lower().endswith(
                    ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif')
                )
            ])

        return {
            "required": {
                "lock_ratio": ("BOOLEAN", {"default": False}),
                "snap_to": ("INT", {
                    "default": 8, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Model-friendly output dimensions are floored "
                               "to this multiple.",
                }),
                # Normalized 0-1 crop bounds — managed by the JS canvas widget
                "crop_x1": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001,
                }),
                "crop_y1": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001,
                }),
                "crop_x2": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.0001,
                }),
                "crop_y2": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.0001,
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "loaded_image": (["none"] + files, {"image_upload": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "INT", "INT", "INT", "INT", "IMAGE")
    RETURN_NAMES = (
        "image", "raw_image", "mask",
        "crop_x", "crop_y", "crop_width", "crop_height",
        "source_image",
    )
    FUNCTION = "crop"
    DESCRIPTION = (
        "Interactive visual crop.  Drag corners on the canvas preview to "
        "define a crop region.  'image' output is resized to model-friendly "
        "dimensions (floored to snap_to); 'raw_image' is the exact pixel crop; "
        "'source_image' passes through the original uncropped image."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Accept any loaded_image value, including a stale or missing one, so a
        # wired image input is never blocked by it. loaded_image is only used
        # when nothing is wired (see crop): the resolver prefers the wired image
        # and raises a clear error at run time if neither a wired image nor a
        # valid loaded file is available. Validating it here would reject a stale
        # selection even when a wired image makes it irrelevant. The **kwargs
        # signature also tells ComfyUI to skip its built-in "value not in list"
        # combo check, so a filename no longer in the input folder still passes.
        return True

    def crop(
        self, lock_ratio, snap_to,
        crop_x1, crop_y1, crop_x2, crop_y2,
        image=None, loaded_image="none", debug=False,
    ):
        # Resolve source image: wired takes priority over loaded
        if image is not None:
            src = image
            source_label = "wired source"
        elif loaded_image and loaded_image != "none":
            input_dir = folder_paths.get_input_directory()
            img_path = os.path.join(input_dir, loaded_image)
            if not os.path.exists(img_path):
                raise ValueError(f"Loaded image not found: {img_path}")
            pil_img = Image.open(img_path)
            pil_img = ImageOps.exif_transpose(pil_img)
            pil_img = pil_img.convert("RGB")
            src = torch.from_numpy(
                np.array(pil_img).astype(np.float32) / 255.0
            ).unsqueeze(0)
            source_label = f"loaded: {loaded_image}"
        else:
            raise ValueError(
                "No image provided. Wire an image input or load one manually."
            )

        if src.dtype != torch.float32:
            src = (
                src.float() / 255.0 if src.dtype == torch.uint8
                else src.float()
            )

        b, h, w, c = src.shape

        # Denormalize and order crop coordinates
        x1, x2 = sorted([int(crop_x1 * w), int(crop_x2 * w)])
        y1, y2 = sorted([int(crop_y1 * h), int(crop_y2 * h)])

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        crop_w = x2 - x1
        crop_h = y2 - y1

        # Raw crop — exact pixels the user selected
        raw_crop = src[:, y1:y2, x1:x2, :]

        # Model-friendly output — floor dimensions to snap_to multiples
        friendly_w = max(snap_to, (crop_w // snap_to) * snap_to)
        friendly_h = max(snap_to, (crop_h // snap_to) * snap_to)

        if friendly_w != crop_w or friendly_h != crop_h:
            temp = raw_crop.permute(0, 3, 1, 2)
            temp = F.interpolate(
                temp, size=(friendly_h, friendly_w),
                mode="bilinear", align_corners=False,
            )
            friendly = torch.clamp(temp.permute(0, 2, 3, 1), 0.0, 1.0)
        else:
            friendly = raw_crop.clone()

        # Mask at original image dimensions — white inside crop region
        mask = torch.zeros((b, h, w), dtype=torch.float32)
        mask[:, y1:y2, x1:x2] = 1.0

        if debug:
            print(
                f"[CCN CroppedImage] Source: {w}x{h} | "
                f"Crop: ({x1},{y1})->({x2},{y2}) = {crop_w}x{crop_h} | "
                f"Friendly: {friendly_w}x{friendly_h} (snap={snap_to})"
            )

        # Save input image as temp preview so the JS widget can display it
        preview_name = f"ccn_crop_preview_{uuid.uuid4().hex[:8]}.png"
        preview_dir = folder_paths.get_temp_directory()
        preview_np = (
            (src[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        )
        Image.fromarray(preview_np).save(
            os.path.join(preview_dir, preview_name), compress_level=4,
        )

        # Deliver the preview under a custom UI key (not "images"): ComfyUI
        # passes it straight to the JS onExecuted handler, but only the literal
        # "images" key renders a node preview and pushes to the image feed.
        # This keeps the crop canvas fed while suppressing both of those.
        return {
            "ui": {
                "ccn_crop_preview": [{
                    "filename": preview_name,
                    "subfolder": "",
                    "type": "temp",
                }],
                # Reported so the JS can label which source actually ran.
                "ccn_crop_source": [source_label],
            },
            "result": (friendly, raw_crop, mask, x1, y1, crop_w, crop_h, src),
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        loaded = kwargs.get("loaded_image", "none")
        if loaded and loaded != "none":
            input_dir = folder_paths.get_input_directory()
            path = os.path.join(input_dir, loaded)
            if os.path.exists(path):
                return f"{path}:{os.path.getmtime(path)}"
        return ""
