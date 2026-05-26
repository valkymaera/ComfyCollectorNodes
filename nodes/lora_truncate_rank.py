"""
Fast LoRA rank reduction by directly truncating SVD-ordered components.

Unlike full re-decomposition (which reconstructs the delta and runs SVD again),
this simply slices lora_up and lora_down to keep only the top N components.
This is valid because SVD extraction stores singular values in descending order,
so the first N components are guaranteed to be the N most significant.

IMPORTANT: This only produces correct results for LoRAs created via SVD
extraction (e.g. LoraExtractKJ, kohya extract). LoRAs from training do NOT
have ordered rank dimensions — truncating them would discard arbitrary
components rather than the least significant ones.
"""

import os
import torch
import numpy as np
import folder_paths
import comfy.utils
from tqdm import tqdm

# Recognized LoRA weight key formats
LORA_FORMATS = [
    ("lora_down", "lora_up"),       # sd-scripts / kohya
    ("lora_A", "lora_B"),           # PEFT
    ("down", "up"),                 # ControlLoRA
]


def _find_lora_format(keys):
    """Detect which LoRA key format is used in this file."""
    key_str = " ".join(keys)
    for down_name, up_name in LORA_FORMATS:
        if f".{down_name}." in key_str or f".{down_name}" in key_str:
            return down_name, up_name
    return None, None


def _get_partner_key(key, down_name, up_name):
    """Given a lora_down key, return the corresponding lora_up key and vice versa."""
    if f".{down_name}." in key:
        return key.replace(f".{down_name}.", f".{up_name}."), "down"
    elif key.endswith(f".{down_name}"):
        return key[: -len(down_name)] + up_name, "down"
    elif f".{up_name}." in key:
        return key.replace(f".{up_name}.", f".{down_name}."), "up"
    elif key.endswith(f".{up_name}"):
        return key[: -len(up_name)] + down_name, "up"
    return None, None


def _get_alpha_key(down_key, down_name):
    """Derive the alpha key from a lora_down key."""
    # e.g. "model.layer.lora_down.weight" -> "model.layer.alpha"
    if f".{down_name}.weight" in down_key:
        return down_key.replace(f".{down_name}.weight", ".alpha")
    elif f".{down_name}" in down_key:
        idx = down_key.rfind(f".{down_name}")
        return down_key[:idx] + ".alpha"
    return None


class LoraTruncateRank:
    """Fast LoRA rank reduction by slicing SVD-ordered components.
    
    Only valid for SVD-extracted LoRAs, not trained LoRAs."""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "Source LoRA file. Must be from SVD extraction, not training.",
                }),
                "new_rank": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": (
                        "Target rank. Must be less than the source LoRA's rank. "
                        "Layers already at or below this rank are left unchanged."
                    ),
                }),
                "filename_prefix": ("STRING", {
                    "default": "loras/CCN_truncated_lora",
                    "tooltip": "Output path and filename prefix (relative to output directory).",
                }),
                "output_dtype": (["match_original", "fp16", "bf16", "fp32", "fp8_e4m3", "fp8_e5m2"], {
                    "default": "match_original",
                    "tooltip": "Data type for saved tensors. fp8_e4m3 halves size vs fp16 with minimal quality loss.",
                }),
                "verbose": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print per-layer rank changes to console.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "CCN/lora"
    DESCRIPTION = (
        "Fast rank reduction for SVD-extracted LoRAs by directly truncating "
        "weight matrices. Runs in seconds vs. minutes for full re-decomposition. "
        "NOT valid for trained LoRAs — their rank dimensions are not ordered "
        "by importance."
    )

    def execute(self, lora_name, new_rank, filename_prefix, output_dtype, verbose):
        # --- Load ---
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise FileNotFoundError(f"LoRA not found: {lora_name}")

        lora_sd, metadata = comfy.utils.load_torch_file(lora_path, return_metadata=True)

        # Detect format
        down_name, up_name = _find_lora_format(lora_sd.keys())
        if down_name is None:
            raise ValueError(
                "Could not detect LoRA format. Expected keys containing "
                "'lora_down', 'lora_A', or 'down'."
            )

        # Determine save dtype
        if output_dtype == "match_original":
            first_weight = next(
                (v for k, v in lora_sd.items()
                 if k.endswith(".weight") and isinstance(v, torch.Tensor)),
                None,
            )
            save_dtype = first_weight.dtype if first_weight is not None else torch.float16
        else:
            save_dtype = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
                "fp8_e4m3": torch.float8_e4m3fn,
                "fp8_e5m2": torch.float8_e5m2,
            }[output_dtype]

        # --- Identify down/up pairs ---
        processed = set()
        output_sd = {}
        old_ranks = []
        new_ranks = []

        # Collect all down keys
        down_keys = []
        for k in lora_sd:
            partner, role = _get_partner_key(k, down_name, up_name)
            if role == "down" and partner in lora_sd:
                down_keys.append(k)

        pbar = tqdm(total=len(down_keys), desc="Truncating LoRA rank")
        comfy_pbar = comfy.utils.ProgressBar(len(down_keys))

        for down_key in down_keys:
            up_key, _ = _get_partner_key(down_key, down_name, up_name)
            alpha_key = _get_alpha_key(down_key, down_name)

            down_weight = lora_sd[down_key]
            up_weight = lora_sd[up_key]

            # Determine current rank from the down weight
            old_rank = down_weight.shape[0]
            truncated_rank = min(new_rank, old_rank)

            old_ranks.append(old_rank)
            new_ranks.append(truncated_rank)

            if truncated_rank < old_rank:
                # Truncate: keep top N components
                # down is (rank, in_dim, ...) — slice first dim
                # up is (out_dim, rank, ...) — slice second dim
                new_down = down_weight[:truncated_rank].contiguous().to(save_dtype)
                new_up = up_weight[:, :truncated_rank].contiguous().to(save_dtype)

                if verbose:
                    tqdm.write(f"  {down_key}: {old_rank} -> {truncated_rank}")
            else:
                new_down = down_weight.contiguous().to(save_dtype)
                new_up = up_weight.contiguous().to(save_dtype)

                if verbose:
                    tqdm.write(f"  {down_key}: {old_rank} (unchanged)")

            output_sd[down_key] = new_down
            output_sd[up_key] = new_up

            # Adjust alpha to preserve effective scale
            # Formula: delta = (alpha / rank) * (up @ down)
            # To keep the same scale: new_alpha = old_alpha * (new_rank / old_rank)
            if alpha_key is not None:
                if alpha_key in lora_sd:
                    old_alpha = float(lora_sd[alpha_key])
                    new_alpha = old_alpha * (truncated_rank / old_rank)
                else:
                    # No alpha means implicit alpha = rank, so new alpha = new_rank
                    new_alpha = float(truncated_rank)
                output_sd[alpha_key] = torch.tensor(new_alpha).to(save_dtype)

            processed.add(down_key)
            processed.add(up_key)
            if alpha_key:
                processed.add(alpha_key)

            pbar.update(1)
            comfy_pbar.update(1)

        pbar.close()

        # Pass through any remaining keys (bias diffs, full diffs, etc.)
        for k, v in lora_sd.items():
            if k not in processed:
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    output_sd[k] = v.to(save_dtype)
                else:
                    output_sd[k] = v

        # --- Summary ---
        if verbose and old_ranks:
            max_old = max(old_ranks)
            avg_new = np.mean(new_ranks)
            changed = sum(1 for o, n in zip(old_ranks, new_ranks) if n < o)
            unchanged = len(old_ranks) - changed
            print(f"\n[CCN] Truncate summary: {changed} layers reduced, "
                  f"{unchanged} unchanged. Max old rank: {max_old}, "
                  f"avg new rank: {avg_new:.1f}")

        # --- Update metadata ---
        if metadata is None:
            metadata = {}
        metadata["ss_training_comment"] = (
            f"Truncated from rank {max(old_ranks) if old_ranks else '?'} "
            f"to {new_rank};"
            + metadata.get("ss_training_comment", "")
        )
        metadata["ss_network_dim"] = str(new_rank)

        # --- Save ---
        full_output_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        os.makedirs(full_output_folder, exist_ok=True)

        max_old_rank = max(old_ranks) if old_ranks else 0
        dtype_suffix = f"_{output_dtype}" if output_dtype != "match_original" else ""
        output_file = (
            f"{filename}_trunc{max_old_rank}to{new_rank}"
            f"{dtype_suffix}_{counter:05d}.safetensors"
        )
        filepath = os.path.join(full_output_folder, output_file)

        comfy.utils.save_torch_file(output_sd, filepath, metadata=metadata)

        size_before = os.path.getsize(lora_path) / (1024 * 1024)
        size_after = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[CCN] Truncated LoRA saved: {filepath}")
        print(f"[CCN] Size: {size_before:.1f}MB -> {size_after:.1f}MB ({size_after/size_before*100:.0f}%)")

        return (filepath,)
