"""
Rotate Image — Visual image rotation around a draggable pivot point.

Provides a canvas widget where users drag to rotate the image live and
drag a crosshair to place the pivot.  Fixed mode keeps the input frame
(content rotates around the pivot, uncovered areas are filled); expand
mode grows the output to the rotated bounding box.  A MASK output marks
where the fill shows, for inpainting the uncovered regions.  Can also
function as a standalone image loader when no upstream image is wired.
"""

import math
import os
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import folder_paths


def _to_rgb(t):
    """Coerce an IMAGE tensor (B, H, W, C) to float32 RGB (B, H, W, 3).

    A 4-channel (RGBA) input has its alpha dropped rather than composited.
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
    return t[..., :3] if c > 3 else torch.cat([t] * 3, dim=-1)[..., :3]


def _parse_hex_color(value):
    """Parse a 3- or 6-digit hex color ('#' optional) to float RGB 0-1."""
    s = (value or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) == 6:
        try:
            return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except ValueError:
            pass
    raise ValueError(
        f"fill_color must be a 3- or 6-digit hex color like #1a2b3c, got: {value!r}"
    )


class RotateImage:
    """Visual rotate node with interactive JS pivot/angle overlay and optional image loading."""

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
                "angle": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1,
                    "tooltip": "Rotation in degrees, positive = clockwise. "
                               "Drag on the canvas to rotate; Shift snaps to 15°.",
                }),
                "expand": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Off: output keeps the input dimensions (corners "
                               "crop, uncovered areas fill). On: output grows to "
                               "the rotated bounding box; pivot has no effect.",
                }),
                "fill_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Hex color painted where the rotated image no "
                               "longer covers the frame.",
                }),
                "interpolation": (["bilinear", "nearest", "bicubic"],),
                # Normalized 0-1 pivot — managed by the JS canvas widget
                "pivot_x": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.0001,
                }),
                "pivot_y": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.0001,
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "loaded_image": (["none"] + files, {"image_upload": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "source_image")
    FUNCTION = "rotate"
    DESCRIPTION = (
        "Interactive visual rotation.  Drag on the canvas to rotate around a "
        "draggable pivot (center by default).  With expand off the output keeps "
        "the input frame and uncovered areas take fill_color; with expand on "
        "the output grows to the rotated bounding box.  'mask' is white where "
        "the fill shows (including bounding-box corners in expand mode) for "
        "inpainting; 'source_image' passes through the unrotated image."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Accept any loaded_image value, including a stale or missing one, so a
        # wired image input is never blocked by it. loaded_image is only used
        # when nothing is wired: the resolver prefers the wired image and raises
        # a clear error at run time if neither a wired image nor a valid loaded
        # file is available. The **kwargs signature also tells ComfyUI to skip
        # its built-in "value not in list" combo check, so a filename no longer
        # in the input folder still passes.
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
            "No image provided. Wire an image input or load one manually."
        )

    def _build_grid(
        self, angle_deg, w_in, h_in, w_out, h_out,
        cx_in, cy_in, cx_out, cy_out, device, dtype,
    ):
        """Inverse-map sampling grid for grid_sample (align_corners=False).

        Positive angle is clockwise in screen space (y-down), matching canvas
        ctx.rotate() in the JS widget, so each output pixel center is pulled
        through the transposed rotation back to its input location.
        """
        t = math.radians(angle_deg)
        cos_t, sin_t = math.cos(t), math.sin(t)
        xs = torch.arange(w_out, device=device, dtype=dtype) + 0.5
        ys = torch.arange(h_out, device=device, dtype=dtype) + 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dx = xx - cx_out
        dy = yy - cy_out
        x_in = cx_in + cos_t * dx + sin_t * dy
        y_in = cy_in - sin_t * dx + cos_t * dy
        u = 2.0 * x_in / w_in - 1.0
        v = 2.0 * y_in / h_in - 1.0
        return torch.stack((u, v), dim=-1).unsqueeze(0)

    def _save_temp_preview(self, frame_hwc, prefix):
        """Write one RGB frame to the temp dir; return the filename for the UI."""
        arr = (frame_hwc.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        name = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        Image.fromarray(arr).save(
            os.path.join(folder_paths.get_temp_directory(), name),
            compress_level=4,
        )
        return name

    def rotate(
        self, angle, expand, fill_color, interpolation, pivot_x, pivot_y,
        image=None, loaded_image="none", debug=False,
    ):
        src, source_label = self._resolve_source(image, loaded_image)
        b, h, w, _ = src.shape

        # Validate up front so a bad hex fails consistently regardless of angle.
        fill_rgb = _parse_hex_color(fill_color)

        angle = math.fmod(angle, 360.0)
        k = round(angle / 90.0)

        if abs(angle) < 1e-5:
            out = src
            mask = torch.zeros((b, h, w), dtype=torch.float32)
        elif expand and abs(angle - k * 90.0) < 1e-9:
            # Exact quarter turns: bit-exact and guaranteed exact dims.
            # rot90 is counter-clockwise while positive angle is clockwise,
            # hence the negated k.
            out = torch.rot90(src, k=(-k) % 4, dims=(1, 2)).contiguous()
            mask = torch.zeros(out.shape[:3], dtype=torch.float32)
        else:
            if expand:
                t = math.radians(angle)
                cos_a, sin_a = abs(math.cos(t)), abs(math.sin(t))
                # round, not ceil: at exact 90s float dust would add a row.
                w_out = max(1, round(w * cos_a + h * sin_a))
                h_out = max(1, round(w * sin_a + h * cos_a))
                cx_in, cy_in = w / 2.0, h / 2.0
                cx_out, cy_out = w_out / 2.0, h_out / 2.0
            else:
                w_out, h_out = w, h
                cx_in = cx_out = pivot_x * w
                cy_in = cy_out = pivot_y * h

            grid = self._build_grid(
                angle, w, h, w_out, h_out,
                cx_in, cy_in, cx_out, cy_out, src.device, src.dtype,
            )
            src_nchw = src.permute(0, 3, 1, 2)
            sampled = F.grid_sample(
                src_nchw, grid.expand(b, -1, -1, -1), mode=interpolation,
                padding_mode="zeros", align_corners=False,
            )

            # Coverage of the source over each output pixel. Bilinear even for
            # bicubic images (bicubic rings at edges; the image clamp handles
            # its overshoot on the color side).
            cov_mode = "nearest" if interpolation == "nearest" else "bilinear"
            ones = torch.ones((1, 1, h, w), device=src.device, dtype=src.dtype)
            cov = F.grid_sample(
                ones, grid, mode=cov_mode,
                padding_mode="zeros", align_corners=False,
            ).clamp(0.0, 1.0)

            # Zeros padding already attenuates sampled edge pixels by coverage,
            # so the over-composite is sampled + fill*(1-cov); multiplying
            # sampled by cov again would double-attenuate into a dark fringe.
            fill = torch.tensor(
                fill_rgb, device=src.device, dtype=src.dtype
            ).view(1, 3, 1, 1)
            out = (sampled + fill * (1.0 - cov)).clamp(0.0, 1.0)
            out = out.permute(0, 2, 3, 1).contiguous()
            mask = (1.0 - cov)[0, 0].unsqueeze(0).repeat(b, 1, 1)

        if debug:
            print(
                f"[CCN RotateImage] Source: {w}x{h} ({source_label}) | "
                f"angle={angle:.2f} expand={expand} "
                f"pivot=({pivot_x:.3f},{pivot_y:.3f}) | "
                f"out: {out.shape[2]}x{out.shape[1]} | {interpolation}"
            )

        # Save the unrotated source as a temp preview so the JS widget can use
        # it as the backdrop (rotation is applied live on the canvas).
        # Delivered under a custom UI key (not "images"): ComfyUI passes it to
        # onExecuted but only the literal "images" key renders a node preview
        # and pushes to the image feed.
        return {
            "ui": {
                "ccn_rotate_preview": [{
                    "filename": self._save_temp_preview(src[0], "ccn_rotate_preview"),
                    "subfolder": "",
                    "type": "temp",
                }],
                "ccn_rotate_source": [source_label],
            },
            "result": (out, mask, src),
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
