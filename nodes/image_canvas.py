"""
Image Canvas — Generate a solid-color image of a given size.

A minimal source node for compositions: outputs one flat-color frame to
serve as a base for the interactive canvas tools (Cropped Image, Image
Inset, Image Stitch, Rotate Image).  Its JS side exposes the CCN
live-preview hook (ccn_image), rendering the canvas client-side from the
current widget values, so a downstream node's Load Preview shows it
without executing the graph.
"""

import torch

from .rotate_image import _parse_hex_color


class ImageCanvas:
    """Solid-color image generator with a live client-side preview hook."""

    CATEGORY = "ComfyCollectorNodes/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 1024, "min": 1, "max": 8192, "step": 1,
                }),
                "height": ("INT", {
                    "default": 1024, "min": 1, "max": 8192, "step": 1,
                }),
                "color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Hex fill color, 3- or 6-digit, '#' optional "
                               "(like #1a2b3c).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    DESCRIPTION = (
        "Output a solid-color image of the given width, height, and hex "
        "color — a blank canvas base for the interactive image tools.  "
        "Downstream CCN canvas nodes can Load Preview it without executing "
        "the graph."
    )

    def generate(self, width, height, color):
        rgb = _parse_hex_color(color, label="color")
        # Self-clamp for raw API prompts that bypass widget validation.
        w = max(1, min(8192, int(width)))
        h = max(1, min(8192, int(height)))
        image = torch.tensor(rgb, dtype=torch.float32).view(1, 1, 1, 3)
        return (image.expand(1, h, w, 3).contiguous(),)
