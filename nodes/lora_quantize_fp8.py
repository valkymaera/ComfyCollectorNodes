"""
LoRA Quantize FP8 - Convert LoRA weights to float8_e4m3fn and save.
Optionally prune small singular values before quantizing.
"""

import os
import torch
import folder_paths
from safetensors.torch import load_file, save_file


class LoraQuantizeFP8:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "output_filename": ("STRING", {"default": "", "tooltip": "Leave blank to auto-name with _fp8 suffix"}),
                "prune_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05,
                                          "tooltip": "Fraction of smallest singular values to zero out before quantizing. 0 = no pruning."}),
                "dtype": (["float8_e4m3fn", "float8_e5m2"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "quantize"
    CATEGORY = "CCN"
    OUTPUT_NODE = True

    def quantize(self, lora_name, output_filename, prune_ratio, dtype):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        state_dict = load_file(lora_path)

        target_dtype = getattr(torch, dtype)

        if prune_ratio > 0:
            state_dict = self._prune(state_dict, prune_ratio)

        quantized = {}
        for key, tensor in state_dict.items():
            # Only quantize float tensors; leave indices/metadata alone
            if tensor.is_floating_point():
                quantized[key] = tensor.to(torch.float32).to(target_dtype)
            else:
                quantized[key] = tensor

        # Build output path
        if not output_filename:
            base = os.path.splitext(lora_name)[0]
            output_filename = f"{base}_fp8"

        if not output_filename.endswith(".safetensors"):
            output_filename += ".safetensors"

        lora_dir = os.path.dirname(lora_path)
        out_path = os.path.join(lora_dir, output_filename)

        # Copy over metadata from original if present
        save_file(quantized, out_path)

        size_before = os.path.getsize(lora_path) / (1024 * 1024)
        size_after = os.path.getsize(out_path) / (1024 * 1024)
        print(f"[CCN] LoRA quantized: {size_before:.1f}MB -> {size_after:.1f}MB ({size_after/size_before*100:.0f}%) -> {out_path}")

        return (out_path,)

    def _prune(self, state_dict, ratio):
        """
        Find matched lora_up/lora_down pairs, SVD the product,
        zero the smallest singular values, reconstruct.
        """
        # Group by prefix
        up_keys = {k: k for k in state_dict if "lora_up" in k and "weight" in k}
        processed = set()
        result = dict(state_dict)  # shallow copy

        for up_key in up_keys:
            down_key = up_key.replace("lora_up", "lora_down")
            if down_key not in state_dict:
                continue

            up_w = state_dict[up_key].to(torch.float32)
            down_w = state_dict[down_key].to(torch.float32)

            # up is (out_features, rank), down is (rank, in_features)
            # Product is (out_features, in_features)
            product = up_w @ down_w

            try:
                U, S, Vh = torch.linalg.svd(product, full_matrices=False)
            except Exception:
                continue

            # Zero out bottom fraction of singular values
            cutoff = int(len(S) * ratio)
            if cutoff > 0:
                S[-cutoff:] = 0

            # Reconstruct back into up/down form at original rank
            rank = down_w.shape[0]
            S_sqrt = torch.sqrt(S[:rank])
            new_up = U[:, :rank] * S_sqrt.unsqueeze(0)   # (out, rank)
            new_down = Vh[:rank, :] * S_sqrt.unsqueeze(1)  # (rank, in)

            result[up_key] = new_up.to(state_dict[up_key].dtype)
            result[down_key] = new_down.to(state_dict[down_key].dtype)
            processed.add(up_key)
            processed.add(down_key)

        if processed:
            print(f"[CCN] Pruned {len(processed)//2} lora_up/down pairs (ratio={ratio})")

        return result
