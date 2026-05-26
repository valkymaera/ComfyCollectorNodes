"""
Scale and re-save a LoRA file with adjusted alpha or weight values.

Modes:
  - scale_alpha:   Multiply existing alpha values by a scalar.
                   If alphas don't exist, creates them as (rank * scale).
  - set_alpha:     Set all alpha values to a specific float.
  - scale_weights: Multiply all lora_up and lora_down weight tensors by a scalar.
"""

import os
import torch
import folder_paths
from safetensors.torch import load_file, save_file


def _safe_mul(tensor, scalar):
    """Multiply a tensor by a scalar, handling fp8 dtypes that lack arithmetic ops."""
    orig_dtype = tensor.dtype
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return (tensor.to(torch.float32) * scalar).to(orig_dtype)
    return tensor * scalar


class LoraScaleSave:
    """Rescale a LoRA's effective strength and save to a new file."""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "Source LoRA file to scale.",
                }),
                "filename_prefix": ("STRING", {
                    "default": "loras/CCN_scaled_lora",
                    "tooltip": "Output path and filename prefix (relative to output directory).",
                }),
                "mode": (["scale_alpha", "set_alpha", "scale_weights"], {
                    "default": "scale_alpha",
                    "tooltip": (
                        "scale_alpha: Multiply alpha values by the scale factor "
                        "(creates alphas at rank * scale if missing). "
                        "set_alpha: Set all alpha values to the exact value given. "
                        "scale_weights: Multiply all LoRA weight tensors directly."
                    ),
                }),
                "value": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": (
                        "For scale_alpha: multiplier applied to alpha values. "
                        "For set_alpha: the exact alpha value to write. "
                        "For scale_weights: multiplier applied to weight tensors."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "CCN/lora"

    def execute(self, lora_name, filename_prefix, mode, value):
        # --- Load source LoRA ---
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise FileNotFoundError(f"LoRA not found: {lora_name}")

        tensors = load_file(lora_path)
        output = {}

        if mode == "scale_alpha":
            for k, v in tensors.items():
                output[k] = v

            # Scale existing alphas; create missing ones from rank
            alpha_keys_found = {k for k in tensors if ".alpha" in k}

            for k, v in tensors.items():
                if k.endswith(".lora_down.weight"):
                    alpha_key = k.replace(".lora_down.weight", ".alpha")
                    rank = v.shape[0]

                    if alpha_key in alpha_keys_found:
                        output[alpha_key] = _safe_mul(tensors[alpha_key].clone(), value)
                    else:
                        output[alpha_key] = torch.tensor(rank * value)

        elif mode == "set_alpha":
            for k, v in tensors.items():
                output[k] = v

            # Remove any existing alphas, write new ones from down weights
            for k, v in tensors.items():
                if k.endswith(".lora_down.weight"):
                    alpha_key = k.replace(".lora_down.weight", ".alpha")
                    output[alpha_key] = torch.tensor(float(value))

        elif mode == "scale_weights":
            for k, v in tensors.items():
                if k.endswith(".lora_up.weight"):
                    output[k] = _safe_mul(v, value)
                elif k.endswith(".diff") or k.endswith(".diff_b"):
                    output[k] = _safe_mul(v, value)
                else:
                    output[k] = v

        # --- Build output path ---
        full_output_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        os.makedirs(full_output_folder, exist_ok=True)
        filepath = os.path.join(full_output_folder, f"{filename}_{counter:05d}.safetensors")

        save_file(output, filepath)

        print(f"[CCN] LoRA saved: {filepath}  (mode={mode}, value={value})")
        return (filepath,)
