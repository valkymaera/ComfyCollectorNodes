"""
Image Inset — Composite up to three scaled images into a base image.

Built on the CroppedImage base-image machinery: a source is resolved
wired-first (else a loaded/uploaded file), saved to temp as a canvas
backdrop, and never modified.  Instead of one crop rectangle, three
optional IMAGE inputs (embed1/2/3 -> red/green/blue) each get a
destination rectangle on the source.  Connected embeds are scaled to
fill their rectangle and pasted in order 1->3, so blue lands on top
where rectangles overlap.

Two outputs: the compilation (a composited clone) and the base source
(passthrough).  Rectangle placements are stored as normalized 0-1
coordinates in hidden widgets managed by the JS canvas widget, so a
layout is resolution-independent.
"""

import os
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import folder_paths

from .rotate_image import _parse_hex_color


# embed index -> (color name, hidden-widget prefix). Order is also paste order.
_EMBEDS = [(1, "red"), (2, "green"), (3, "blue")]

# Staggered normalized defaults so the three rects don't start stacked. The JS
# widget re-places them aspect-correctly on connect; these are the fallback.
_DEFAULT_RECTS = {
    1: (0.05, 0.05, 0.33, 0.33),
    2: (0.36, 0.36, 0.64, 0.64),
    3: (0.67, 0.67, 0.95, 0.95),
}


def _to_rgb(t):
    """Coerce an IMAGE tensor (B, H, W, C) to float32 RGB (B, H, W, 3).

    Opaque-v1 behavior: a 4-channel (RGBA) embed has its alpha dropped
    rather than composited — masks/cutouts are intentionally out of scope.
    """
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    elif t.dtype != torch.float32:
        t = t.float()

    c = t.shape[-1]
    if c >= 3:
        return t[..., :3]
    if c == 1:
        return t.repeat(1, 1, 1, 3)
    # Unusual channel count — pad/truncate to 3 so downstream stays uniform.
    return t[..., :3] if c > 3 else torch.cat([t] * 3, dim=-1)[..., :3]


class ImageInset:
    """Composite up to three scaled embeds into a base image via draggable rects."""

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

        required = {
            "lock_ratio": ("BOOLEAN", {"default": True}),
        }
        # Normalized 0-1 destination rects, managed by the JS canvas widget.
        for idx, _ in _EMBEDS:
            dx1, dy1, dx2, dy2 = _DEFAULT_RECTS[idx]
            required[f"embed{idx}_x1"] = ("FLOAT", {"default": dx1, "min": 0.0, "max": 1.0, "step": 0.0001})
            required[f"embed{idx}_y1"] = ("FLOAT", {"default": dy1, "min": 0.0, "max": 1.0, "step": 0.0001})
            required[f"embed{idx}_x2"] = ("FLOAT", {"default": dx2, "min": 0.0, "max": 1.0, "step": 0.0001})
            required[f"embed{idx}_y2"] = ("FLOAT", {"default": dy2, "min": 0.0, "max": 1.0, "step": 0.0001})

        return {
            "required": required,
            "optional": {
                "image": ("IMAGE",),
                "embed1": ("IMAGE",),
                "embed2": ("IMAGE",),
                "embed3": ("IMAGE",),
                "loaded_image": (["none"] + files, {"image_upload": True}),
                "debug": ("BOOLEAN", {"default": False}),
                # Kept at the end of optional: widgets_values serializes
                # positionally, so appending here leaves old saves' indices
                # intact (the JS onConfigure guard handles the trailing shift).
                "border_width": ("INT", {
                    "default": 0, "min": 0, "max": 1024, "step": 1,
                    "tooltip": "Border thickness in base-image pixels, drawn "
                               "inside each rectangle (0 = none). The rectangle "
                               "stays the total footprint; the embed shrinks to "
                               "the inner box.",
                }),
                "border_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Hex border color, 3- or 6-digit, '#' optional "
                               "(like #1a2b3c).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("compilation", "base")
    FUNCTION = "compose"
    DESCRIPTION = (
        "Composite up to three scaled images into a base image.  Drag each "
        "embed's rectangle on the canvas to place it; the embed fills its "
        "rectangle.  An optional border is drawn inside each rectangle "
        "(border_width in base pixels, 0 = none), with the embed shrunk to "
        "the inner box.  'compilation' is the composited result; 'base' "
        "passes the source through.  The source is never modified."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Mirror CroppedImage: accept any loaded_image value (including a stale
        # or missing one) so a wired image is never blocked by it. The resolver
        # in compose() prefers the wired source and errors clearly at run time
        # if neither a wired source nor a valid loaded file exists. The **kwargs
        # signature also tells ComfyUI to skip its built-in combo membership
        # check, so a filename no longer in the input folder still passes.
        return True

    def _resolve_source(self, image, loaded_image):
        """Return (src_rgb float32 BHW3, provenance label). Wired wins over loaded."""
        if image is not None:
            return _to_rgb(image), "wired source"

        if loaded_image and loaded_image != "none":
            input_dir = folder_paths.get_input_directory()
            img_path = os.path.join(input_dir, loaded_image)
            if not os.path.exists(img_path):
                raise ValueError(f"Loaded image not found: {img_path}")
            pil_img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
            src = torch.from_numpy(
                np.array(pil_img).astype(np.float32) / 255.0
            ).unsqueeze(0)
            return src, f"loaded: {loaded_image}"

        raise ValueError(
            "No base image provided. Wire an image input or load one manually."
        )

    def _rect_pixels(self, rect, w, h):
        """Normalized rect -> ordered, clamped integer pixel box (x1, y1, x2, y2)."""
        rx1, ry1, rx2, ry2 = rect
        x1, x2 = sorted([int(rx1 * w), int(rx2 * w)])
        y1, y2 = sorted([int(ry1 * h), int(ry2 * h)])
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return x1, y1, x2, y2

    def _composite(self, canvas, embed_rgb, rect, debug, color, border_px, border_rgb):
        """Paste embed_rgb scaled to fill its pixel box into canvas (in place).

        With a border, the box's outer ring is filled with border_rgb and the
        embed shrinks to the inner box — the rect stays the total footprint.
        """
        b, h, w, _ = canvas.shape
        x1, y1, x2, y2 = self._rect_pixels(rect, w, h)
        box_w, box_h = x2 - x1, y2 - y1

        # Uniform ring, clamped so the inner box stays at least 1x1.
        eff = min(border_px, (box_w - 1) // 2, (box_h - 1) // 2)
        if eff > 0:
            canvas[:, y1:y2, x1:x2, :] = torch.tensor(
                border_rgb, dtype=canvas.dtype, device=canvas.device
            )
        ix1, iy1 = x1 + eff, y1 + eff
        ix2, iy2 = x2 - eff, y2 - eff
        in_w, in_h = ix2 - ix1, iy2 - iy1

        embed_rgb = embed_rgb.to(device=canvas.device, dtype=canvas.dtype)
        embed_b = embed_rgb.shape[0]

        # Scale per distinct source frame only — when embed is a single image
        # (the common case) this resolves to one resize reused across the batch.
        scaled_cache = {}
        for bi in range(b):
            ei = min(bi, embed_b - 1)
            scaled = scaled_cache.get(ei)
            if scaled is None:
                e = embed_rgb[ei:ei + 1].permute(0, 3, 1, 2)
                e = F.interpolate(
                    e, size=(in_h, in_w),
                    mode="bilinear", align_corners=False,
                )
                scaled = torch.clamp(e.permute(0, 2, 3, 1)[0], 0.0, 1.0)
                scaled_cache[ei] = scaled
            canvas[bi, iy1:iy2, ix1:ix2, :] = scaled

        if debug:
            border_note = ""
            if border_px > 0:
                border_note = f" | border {eff}px"
                if eff != border_px:
                    border_note += f" (requested {border_px})"
            print(
                f"[CCN ImageInset] {color}: box ({x1},{y1})->({x2},{y2}) "
                f"= {box_w}x{box_h} | embed {embed_b}x{embed_rgb.shape[2]}x"
                f"{embed_rgb.shape[1]}{border_note}"
            )

    def _save_temp_preview(self, frame_hwc, prefix):
        """Write one RGB frame to the temp dir; return the filename for the UI."""
        arr = (frame_hwc.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        name = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        Image.fromarray(arr).save(
            os.path.join(folder_paths.get_temp_directory(), name),
            compress_level=4,
        )
        return name

    def compose(
        self, lock_ratio,
        embed1_x1, embed1_y1, embed1_x2, embed1_y2,
        embed2_x1, embed2_y1, embed2_x2, embed2_y2,
        embed3_x1, embed3_y1, embed3_x2, embed3_y2,
        image=None, embed1=None, embed2=None, embed3=None,
        loaded_image="none", debug=False,
        border_width=0, border_color="#000000",
    ):
        src, source_label = self._resolve_source(image, loaded_image)
        b, h, w, _ = src.shape

        # Self-clamp: VALIDATE_INPUTS returning True skips ComfyUI's built-in
        # min/max checks, so an API prompt can deliver out-of-range values.
        # The color only matters (and is only validated) when a border shows.
        border_px = max(0, int(border_width))
        border_rgb = (
            _parse_hex_color(border_color, label="border_color")
            if border_px > 0 else None
        )

        rects = {
            1: (embed1_x1, embed1_y1, embed1_x2, embed1_y2),
            2: (embed2_x1, embed2_y1, embed2_x2, embed2_y2),
            3: (embed3_x1, embed3_y1, embed3_x2, embed3_y2),
        }
        embeds = {1: embed1, 2: embed2, 3: embed3}

        # Composite into a clone so the passthrough base stays untouched.
        compilation = src.clone()
        for idx, color in _EMBEDS:
            embed = embeds[idx]
            if embed is None:
                continue
            self._composite(
                compilation, _to_rgb(embed), rects[idx], debug, color,
                border_px, border_rgb,
            )
        compilation = torch.clamp(compilation, 0.0, 1.0)

        if debug:
            active = [c for i, c in _EMBEDS if embeds[i] is not None]
            print(
                f"[CCN ImageInset] Base: {w}x{h} ({source_label}) | "
                f"active embeds: {active or 'none'}"
            )

        # Backdrop + per-embed thumbnails for the JS canvas, under custom UI
        # keys (not "images"): ComfyUI hands them to onExecuted but only the
        # literal "images" key renders a node preview / pushes to the feed.
        ui = {
            "ccn_inset_preview": [{
                "filename": self._save_temp_preview(src[0], "ccn_inset_preview"),
                "subfolder": "",
                "type": "temp",
            }],
            "ccn_inset_source": [source_label],
        }
        for idx, _ in _EMBEDS:
            embed = embeds[idx]
            if embed is None:
                continue
            ui[f"ccn_inset_embed{idx}"] = [{
                "filename": self._save_temp_preview(
                    _to_rgb(embed)[0], f"ccn_inset_embed{idx}"
                ),
                "subfolder": "",
                "type": "temp",
            }]

        return {"ui": ui, "result": (compilation, src)}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Re-run when the loaded file changes on disk even if the dropdown value
        # is unchanged. Widget (rect) and wired-input changes are tracked by
        # ComfyUI independently, so this only adds the file-mtime signal.
        loaded = kwargs.get("loaded_image", "none")
        if loaded and loaded != "none":
            input_dir = folder_paths.get_input_directory()
            path = os.path.join(input_dir, loaded)
            if os.path.exists(path):
                return f"{path}:{os.path.getmtime(path)}"
        return ""
