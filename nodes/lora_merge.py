"""
Model-free LoRA merging.

A single node (LoraMergeSave) with a dynamic row list like the LoRA Multi
Loader: rows expand as LoRAs are added, each with its own strength, and the
node merges every enabled row into one LoRA file. The dynamic rows are
provided by js/lora_merge.js and arrive as widget values named "lora_<n>",
each a dict: {"on": bool, "lora": str, "strength": float, "id": int}.

Math
----
Each LoRA i contributes a delta:  dW_i = s_i * (alpha_i / rank_i) * up_i @ down_i
The merged delta  dW = sum_i dW_i  is represented *exactly* by:

    down' = concat_rows( c_i * down_i )     where c_i = s_i * alpha_i / rank_i
    up'   = concat_cols( up_i )
    alpha' = rank' = sum_i rank_i           (so the loader's alpha/rank factor is 1)

No base model is involved, so no merge-into-checkpoint + extract round trip.
Full-rank ".diff" / ".diff_b" patches are merged by weighted summation
(also exact). DoRA adapters cannot be merged this way and abort with a
clear error. Optional SVD re-ranking compresses the concatenated result to
a target rank (the only lossy step, off by default).

Key conventions understood (auto-detected, mixed inputs allowed):
    {base}.lora_up.weight / {base}.lora_down.weight   (+ optional {base}.alpha)
    {base}.lora.up.weight / {base}.lora.down.weight
    {base}.lora_B[.infix].weight / {base}.lora_A[.infix].weight   (peft)
The merged file re-uses the key naming of the first LoRA that provided each
layer, so it stays compatible with whatever loader handled the inputs.
"""

import os
import re
import json
import time

import torch
import folder_paths
import comfy.utils
import comfy.model_management
from tqdm import tqdm

from .lora_multi_loader import AnyType, FlexibleOptionalInputType, _NONE_VALUES

_ANY = AnyType("*")


# --------------------------------------------------------------------------
# Key parsing (pure python, no torch)
# --------------------------------------------------------------------------

# Order matters: the specific dotted forms must match before the peft forms.
_KEY_PATTERNS = (
    (re.compile(r"^(?P<base>.+)\.lora_up\.weight$"),                     "up"),
    (re.compile(r"^(?P<base>.+)\.lora_down\.weight$"),                   "down"),
    (re.compile(r"^(?P<base>.+)\.lora\.up\.weight$"),                    "up"),
    (re.compile(r"^(?P<base>.+)\.lora\.down\.weight$"),                  "down"),
    (re.compile(r"^(?P<base>.+)\.lora_B(?:\.[^.]+)?\.weight$"),          "up"),
    (re.compile(r"^(?P<base>.+)\.lora_A(?:\.[^.]+)?\.weight$"),          "down"),
    (re.compile(r"^(?P<base>.+)\.alpha$"),                               "alpha"),
    (re.compile(r"^(?P<base>.+)\.diff$"),                                "diff"),
    (re.compile(r"^(?P<base>.+)\.diff_b$"),                              "diff_b"),
    (re.compile(r"^(?P<base>.+)\.dora_scale$"),                          "dora"),
)


def parse_lora_keys(keys):
    """Group state-dict keys by base layer name.

    Returns (groups, unknown) where groups maps
        base -> {"up": key, "down": key, "alpha": key,
                 "diff": key, "diff_b": key, "dora": key}
    (only the roles actually present) and unknown is a list of keys that
    matched no known LoRA convention.
    """
    groups = {}
    unknown = []
    for k in keys:
        for pattern, role in _KEY_PATTERNS:
            m = pattern.match(k)
            if m:
                entry = groups.setdefault(m.group("base"), {})
                if role in entry:
                    # Same role twice for one base (e.g. both lora_up and
                    # lora_B) would be ambiguous — treat as unknown rather
                    # than silently picking one.
                    unknown.append(k)
                else:
                    entry[role] = k
                break
        else:
            unknown.append(k)
    return groups, unknown


# --------------------------------------------------------------------------
# Merge core (torch)
# --------------------------------------------------------------------------

class LoraMergeError(Exception):
    pass


def _as_2d(t):
    """Flatten a (possibly conv) LoRA factor to 2D for matmul/SVD."""
    return t.reshape(t.shape[0], -1)


def _accumulate(acc, sd, groups, strength, source_name):
    """Fold one parsed LoRA state dict into the accumulator.

    acc maps base -> {
        "ups": [t...], "downs": [t...],         # scale folded into downs
        "up_key": str, "down_key": str,          # first-seen output naming
        "emit_alpha": bool,
        "up_shape": tuple, "down_kv_shape": tuple,  # consistency checks
        "diff": tensor or None, "diff_key": str,
        "diff_b": tensor or None, "diff_b_key": str,
    }
    """
    for base, entry in groups.items():
        if "dora" in entry:
            raise LoraMergeError(
                f"'{source_name}' contains DoRA weights ({entry['dora']}). "
                "DoRA adapters rescale weight magnitudes and cannot be merged "
                "by concatenation — merge aborted."
            )

        has_pair = "up" in entry and "down" in entry
        has_diff = "diff" in entry or "diff_b" in entry
        if not has_pair and not has_diff:
            # e.g. an orphaned alpha with no up/down — nothing to merge.
            continue

        slot = acc.setdefault(base, {
            "ups": [], "downs": [],
            "up_key": None, "down_key": None, "emit_alpha": False,
            "up_shape": None, "down_kv_shape": None,
            "diff": None, "diff_key": None,
            "diff_b": None, "diff_b_key": None,
        })

        if has_pair:
            up = sd[entry["up"]].to(torch.float32)
            down = sd[entry["down"]].to(torch.float32)
            rank = down.shape[0]
            if up.shape[1] != rank:
                raise LoraMergeError(
                    f"'{source_name}' layer '{base}': up dim 1 "
                    f"({up.shape[1]}) does not match down rank ({rank})."
                )
            alpha_t = sd.get(entry["alpha"]) if "alpha" in entry else None
            alpha = float(alpha_t.item()) if alpha_t is not None else None
            scale = strength * ((alpha / rank) if alpha is not None else 1.0)

            up_shape = (up.shape[0],) + tuple(up.shape[2:])
            down_kv_shape = tuple(down.shape[1:])
            if slot["up_key"] is None:
                slot["up_key"] = entry["up"]
                slot["down_key"] = entry["down"]
                slot["emit_alpha"] = "alpha" in entry
                slot["up_shape"] = up_shape
                slot["down_kv_shape"] = down_kv_shape
            else:
                if slot["up_shape"] != up_shape or slot["down_kv_shape"] != down_kv_shape:
                    raise LoraMergeError(
                        f"Layer '{base}' has incompatible shapes across inputs: "
                        f"up {slot['up_shape']} vs {up_shape}, "
                        f"down-kv {slot['down_kv_shape']} vs {down_kv_shape}."
                    )
            slot["ups"].append(up)
            slot["downs"].append(down * scale)

        if "diff" in entry:
            d = sd[entry["diff"]].to(torch.float32) * strength
            slot["diff"] = d if slot["diff"] is None else slot["diff"] + d
            slot["diff_key"] = slot["diff_key"] or entry["diff"]
        if "diff_b" in entry:
            d = sd[entry["diff_b"]].to(torch.float32) * strength
            slot["diff_b"] = d if slot["diff_b"] is None else slot["diff_b"] + d
            slot["diff_b_key"] = slot["diff_b_key"] or entry["diff_b"]


def _svd_rerank(up_cat, down_cat, target_rank):
    """Compress a concatenated (up, down) pair to target_rank via SVD.

    Works for linear (2D) and conv (4D down / 4D-or-2D up) factors; returns
    tensors shaped like the inputs but with the new rank.
    """
    if up_cat.dim() > 2:
        # conv up factors are (out, r, 1, 1); anything else is nonstandard
        trailing = 1
        for d in up_cat.shape[2:]:
            trailing *= d
        if trailing != 1:
            raise LoraMergeError(
                f"Cannot SVD re-rank: up factor has nonstandard shape "
                f"{tuple(up_cat.shape)} (expected trailing dims of 1)."
            )
        up2 = up_cat.reshape(up_cat.shape[0], up_cat.shape[1])
    else:
        up2 = up_cat
    down2 = _as_2d(down_cat)

    delta = up2 @ down2  # (out, in*k*k)
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    r = min(target_rank, S.shape[0])
    sqrt_s = torch.sqrt(S[:r])
    new_up2 = U[:, :r] * sqrt_s.unsqueeze(0)
    new_down2 = sqrt_s.unsqueeze(1) * Vh[:r]

    if down_cat.dim() > 2:
        new_down = new_down2.reshape((r,) + tuple(down_cat.shape[1:]))
    else:
        new_down = new_down2
    if up_cat.dim() > 2:
        new_up = new_up2.reshape((up_cat.shape[0], r) + (1,) * (up_cat.dim() - 2))
    else:
        new_up = new_up2
    return new_up, new_down


def merge_stack(entries, new_rank=0, out_dtype=None, load_fn=None,
                svd_device=None):
    """Merge a list of (path, strength, display_name) into one state dict.

    Returns (state_dict, report_lines). load_fn(path) -> state dict is
    injectable for testing; defaults to comfy.utils.load_torch_file.
    svd_device: torch device for the per-layer SVD when new_rank > 0;
    results always come back to CPU for saving.
    """
    if load_fn is None:
        load_fn = lambda p: comfy.utils.load_torch_file(p, safe_load=True)
    if out_dtype is None:
        out_dtype = torch.float16
    svd_device = torch.device(svd_device) if svd_device is not None \
        else torch.device("cpu")

    acc = {}
    report = []
    styles_seen = set()
    used = 0

    for path, strength, name in tqdm(entries, desc="Loading LoRAs"):
        if strength == 0.0:
            report.append(f"skipped (strength 0): {name}")
            continue
        sd = load_fn(path)
        groups, unknown = parse_lora_keys(sd.keys())
        if not groups:
            raise LoraMergeError(
                f"'{name}' contains no recognizable LoRA keys. "
                f"First keys: {list(sd.keys())[:8]}"
            )
        if unknown:
            report.append(
                f"WARNING: {len(unknown)} unrecognized key(s) in '{name}' "
                f"were ignored, e.g. {unknown[:5]}"
            )
        # crude style fingerprint for the mixed-convention warning
        sample = next(iter(groups.values()))
        if "up" in sample:
            styles_seen.add(sample["up"].rsplit(".", 2)[-2])
        _accumulate(acc, sd, groups, strength, name)
        report.append(f"merged: {name} @ strength {strength} ({len(groups)} layers)")
        used += 1
        del sd

    if used == 0:
        raise LoraMergeError("Nothing to merge — all rows had strength 0.")
    if len(styles_seen) > 1:
        report.append(
            "WARNING: inputs use different key conventions "
            f"({sorted(styles_seen)}). ComfyUI's loader handles this, but "
            "strict third-party loaders may only apply part of the merge."
        )

    out = {}
    max_rank = 0
    svd_layers = 0
    t0 = time.time()
    ui_pbar = comfy.utils.ProgressBar(len(acc))
    pbar = tqdm(total=len(acc), desc="Merging layers")
    for base, slot in acc.items():
        comfy.model_management.throw_exception_if_processing_interrupted()
        if slot["ups"]:
            up_cat = torch.cat(slot["ups"], dim=1)
            down_cat = torch.cat(slot["downs"], dim=0)
            rank = down_cat.shape[0]
            if new_rank > 0:
                # Clamped inside _svd_rerank: new_rank >= concat rank yields a
                # lossless but singular-value-ordered result, so the output
                # can be cleanly truncated along the rank axis later.
                try:
                    up_dev, down_dev = _svd_rerank(
                        up_cat.to(svd_device), down_cat.to(svd_device),
                        new_rank)
                except RuntimeError as exc:
                    if (svd_device.type == "cpu"
                            or "out of memory" not in str(exc).lower()):
                        raise
                    report.append(
                        f"WARNING: GPU ran out of memory at layer '{base}' — "
                        "falling back to CPU for the remaining layers.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    svd_device = torch.device("cpu")
                    up_dev, down_dev = _svd_rerank(up_cat, down_cat, new_rank)
                up_cat = up_dev.to("cpu")
                down_cat = down_dev.to("cpu")
                del up_dev, down_dev
                rank = down_cat.shape[0]
                svd_layers += 1
            max_rank = max(max_rank, rank)
            out[slot["up_key"]] = up_cat.contiguous().to(out_dtype)
            out[slot["down_key"]] = down_cat.contiguous().to(out_dtype)
            if slot["emit_alpha"]:
                # scale is already folded into down; alpha = rank keeps the
                # loader's alpha/rank factor at exactly 1.
                out[f"{base}.alpha"] = torch.tensor(float(rank), dtype=out_dtype)
        if slot["diff"] is not None:
            out[slot["diff_key"]] = slot["diff"].contiguous().to(out_dtype)
        if slot["diff_b"] is not None:
            out[slot["diff_b_key"]] = slot["diff_b"].contiguous().to(out_dtype)
        ui_pbar.update(1)
        pbar.update(1)
    pbar.close()

    mode = (f"svd on {svd_device.type} ({svd_layers} layers)"
            if new_rank > 0 else "exact concat")
    report.append(f"output: {len(out)} tensors, max rank {max_rank}, "
                  f"{mode}, {time.time() - t0:.1f}s")
    return out, report


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class LoraMergeSave:
    """Merge a dynamic list of LoRAs into a single LoRA file — no base model
    needed. Rows expand as LoRAs are added (like the LoRA Multi Loader)."""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {
                    "default": "loras/CCN_merged_lora",
                    "tooltip": "Output path and filename prefix (relative to "
                               "output directory)."}),
                "new_rank": ("INT", {
                    "default": 0, "min": 0, "max": 4096,
                    "tooltip": "0 = exact concatenation (rank = sum of input "
                               "ranks, lossless, NOT truncation-friendly). "
                               ">0 = SVD each layer to min(this, sum of "
                               "ranks): lossy if below the sum, lossless but "
                               "singular-value-ordered (cleanly truncatable "
                               "later) if at/above it."}),
                "output_dtype": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                "svd_device": (["auto", "gpu", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device for the per-layer SVD when new_rank > "
                               "0. GPU is far faster for large models; auto "
                               "uses the GPU when available. Ignored when "
                               "new_rank = 0."}),
            },
            "optional": FlexibleOptionalInputType(_ANY),
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("filepath", "report")
    FUNCTION = "merge_and_save"
    OUTPUT_NODE = True
    CATEGORY = "CCN/lora"
    DESCRIPTION = ("Merges a dynamic list of LoRAs (each at its own strength) "
                   "into one LoRA via exact rank concatenation — no base "
                   "model needed — optionally SVD re-ranked, and saves it as "
                   "a .safetensors file.")

    def merge_and_save(self, filename_prefix, new_rank, output_dtype,
                       svd_device, **kwargs):
        rows = [(key, value) for key, value in kwargs.items()
                if isinstance(value, dict) and "lora" in value]
        rows.sort(key=self._row_key)

        available = set(folder_paths.get_filename_list("loras"))
        entries = []
        for row_number, (key, value) in enumerate(rows, start=1):
            if not value.get("on", True):
                continue
            lora_name = value.get("lora")
            if not isinstance(lora_name, str) or lora_name in _NONE_VALUES:
                continue
            try:
                strength = float(value.get("strength", 1.0))
            except (TypeError, ValueError):
                raise LoraMergeError(
                    f"Row {row_number} ({lora_name!r}) has a non-numeric "
                    f"strength: {value.get('strength')!r}"
                )
            if lora_name not in available:
                raise LoraMergeError(
                    f"Row {row_number} references {lora_name!r}, which is "
                    "not in the loras folder. Refresh the node or fix the "
                    "row; refusing to run with a silently missing LoRA."
                )
            path = folder_paths.get_full_path_or_raise("loras", lora_name)
            entries.append((path, strength, lora_name))

        if not entries:
            raise LoraMergeError(
                "No LoRAs to merge — add at least one enabled row with a "
                "LoRA selected."
            )

        dtype = {"fp16": torch.float16,
                 "bf16": torch.bfloat16,
                 "fp32": torch.float32}[output_dtype]

        torch_device = comfy.model_management.get_torch_device()
        if svd_device == "gpu" and torch_device.type == "cpu":
            raise LoraMergeError(
                "svd_device is 'gpu' but ComfyUI is running without a GPU "
                "device.")
        device = torch.device("cpu") if svd_device == "cpu" else torch_device

        out_sd, report = merge_stack(entries, new_rank=new_rank,
                                     out_dtype=dtype, svd_device=device)

        full_output_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        os.makedirs(full_output_folder, exist_ok=True)
        filepath = os.path.join(
            full_output_folder, f"{filename}_{counter:05d}.safetensors")

        metadata = {
            "ccn_lora_merge": json.dumps({
                "inputs": [{"name": n, "strength": s} for _, s, n in entries],
                "new_rank": new_rank,
                "dtype": output_dtype,
            }),
        }
        comfy.utils.save_torch_file(out_sd, filepath, metadata=metadata)
        report.append(f"saved: {filepath}")
        text = "\n".join(report)
        print(f"[CCN] LoRA Merge & Save:\n{text}")
        return {"ui": {"text": [text]}, "result": (filepath, text)}

    # Numeric sort on the "lora_<n>" suffix keeps row 10 after row 2;
    # non-numeric names sort after, lexicographically.
    @staticmethod
    def _row_key(item):
        suffix = item[0].partition("_")[2]
        return (0, int(suffix), "") if suffix.isdigit() else (1, 0, item[0])
