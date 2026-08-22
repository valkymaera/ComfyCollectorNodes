"""
Image Stitch — Concatenate up to three images horizontally or vertically.

Connected slots stitch in order 1 -> 2 -> 3, skipping unconnected ones.
The first connected image defines the shared edge (height for horizontal,
width for vertical); the others are scaled to match it with aspect ratio
preserved.  An optional solid divider of border_width pixels in
border_color is drawn between adjacent images (never as an outer frame).

The JS canvas widget composes the same layout client-side from upstream
previews, so 'Load Preview' shows the stitch without executing the graph.
Per-slot temp previews are saved on each run as its fallback source.
"""

import os
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import folder_paths

from .rotate_image import _parse_hex_color, _to_rgb


class ImageStitch:
    """Stitch up to three images side by side or stacked, with an optional divider."""

    CATEGORY = "ComfyCollectorNodes/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "direction": (["horizontal", "vertical"], {
                    "tooltip": "horizontal: left-to-right. vertical: "
                               "top-to-bottom. Connected inputs stitch in "
                               "slot order 1 -> 2 -> 3.",
                }),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "border_width": ("INT", {
                    "default": 0, "min": 0, "max": 1024, "step": 1,
                    "tooltip": "Divider thickness in pixels drawn between "
                               "adjacent images only (0 = none). No outer "
                               "frame.",
                }),
                "border_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Hex divider color, 3- or 6-digit, '#' "
                               "optional (like #1a2b3c).",
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch"
    DESCRIPTION = (
        "Stitch up to three images together horizontally or vertically.  "
        "Connected inputs join in slot order 1 -> 2 -> 3; the first one sets "
        "the shared edge and the others scale to match it with aspect ratio "
        "preserved.  An optional border_width divider in border_color is "
        "drawn between adjacent images."
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

    def stitch(
        self, direction, image1=None, image2=None, image3=None,
        border_width=0, border_color="#000000", debug=False,
    ):
        imgs = [
            (i, _to_rgb(t))
            for i, t in ((1, image1), (2, image2), (3, image3))
            if t is not None
        ]
        if not imgs:
            raise ValueError(
                "No images connected. Wire at least one of image1, image2, "
                "or image3."
            )

        # Self-clamp for raw API prompts; the color only matters (and is only
        # validated) when a divider will actually show — border > 0 and at
        # least two images to draw it between.
        border_px = max(0, int(border_width))
        border_rgb = (
            _parse_hex_color(border_color, label="border_color")
            if border_px > 0 and len(imgs) > 1 else None
        )

        first = imgs[0][1]
        dev = first.device
        target_h, target_w = first.shape[1], first.shape[2]
        out_b = max(rgb.shape[0] for _, rgb in imgs)

        # Scale each image so its shared edge matches the first (aspect
        # preserved), then broadcast shorter batches by repeating their last
        # frame so every segment concatenates cleanly.
        segments = []
        for _, rgb in imgs:
            rgb = rgb.to(device=dev, dtype=torch.float32)
            b, h, w, _ = rgb.shape
            # Half-up rounding (not banker's) so the JS preview's Math.round
            # computes identical segment sizes.
            if direction == "horizontal":
                nh, nw = target_h, max(1, int(w * target_h / h + 0.5))
            else:
                nh, nw = max(1, int(h * target_w / w + 0.5)), target_w
            if (nh, nw) != (h, w):
                x = rgb.permute(0, 3, 1, 2)
                x = F.interpolate(
                    x, size=(nh, nw), mode="bilinear", align_corners=False,
                )
                rgb = torch.clamp(x.permute(0, 2, 3, 1), 0.0, 1.0)
            if b < out_b:
                idx = torch.arange(out_b, device=dev).clamp(max=b - 1)
                rgb = rgb[idx]
            segments.append(rgb)

        strip = None
        if border_px > 0 and len(segments) > 1:
            shape = (
                (out_b, target_h, border_px, 3) if direction == "horizontal"
                else (out_b, border_px, target_w, 3)
            )
            strip = torch.tensor(
                border_rgb, dtype=torch.float32, device=dev,
            ).view(1, 1, 1, 3).expand(shape)

        parts = []
        for k, seg in enumerate(segments):
            if k and strip is not None:
                parts.append(strip)
            parts.append(seg)
        cat_dim = 2 if direction == "horizontal" else 1
        out = torch.clamp(torch.cat(parts, dim=cat_dim), 0.0, 1.0)

        if debug:
            sizes = ", ".join(
                f"image{i} {rgb.shape[2]}x{rgb.shape[1]} (b{rgb.shape[0]})"
                for i, rgb in imgs
            )
            print(
                f"[CCN ImageStitch] {direction} | {sizes} | "
                f"border {border_px}px | "
                f"out {out.shape[2]}x{out.shape[1]} (b{out_b})"
            )

        # Per-slot frame-0 previews for the JS canvas, under custom UI keys
        # (not "images"): ComfyUI hands them to onExecuted but only the
        # literal "images" key renders a node preview / pushes to the feed.
        # The JS composes the stitch client-side so direction/border widget
        # changes stay live without a re-run.
        ui = {"ccn_stitch_source": [", ".join(f"image{i}" for i, _ in imgs)]}
        for i, rgb in imgs:
            ui[f"ccn_stitch_image{i}"] = [{
                "filename": self._save_temp_preview(
                    rgb[0], f"ccn_stitch_image{i}"
                ),
                "subfolder": "",
                "type": "temp",
            }]

        return {"ui": ui, "result": (out,)}
