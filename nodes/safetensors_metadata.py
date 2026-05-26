"""
Safetensors Metadata Reader - reads metadata and tensor info from any safetensors file
Works with checkpoints, VAEs, LoRAs, text encoders, etc.
"""

import json
import os
import folder_paths
import safetensors
from .detect_architecture import detect_architecture, format_detection


# Build list of available model directories from folder_paths
def get_model_types():
    """Returns available folder types that might contain safetensors files."""
    # Core types that are almost always present
    types = ["loras", "checkpoints", "vae", "clip", "unet",
             "diffusion_models", "text_encoders", "controlnet",
             "style_models", "hypernetworks", "upscale_models"]
    # Only include types that folder_paths actually knows about
    available = []
    for t in types:
        try:
            folder_paths.get_filename_list(t)
            available.append(t)
        except Exception:
            pass
    return available if available else ["loras", "checkpoints"]


class SafetensorsMetadata:
    """Reads metadata and tensor structure from any safetensors file.
    
    Works across all model types - checkpoints, VAEs, LoRAs, CLIP models,
    text encoders, ControlNets, etc. Outputs a human-readable summary
    of embedded metadata plus a structural overview of tensor shapes
    and dtypes for model identification.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (get_model_types(),),
                "filename": ("STRING", {"default": ""}),
            },
            "optional": {
                "show_tensors": ("BOOLEAN", {"default": True}),
                "max_tensors": ("INT", {"default": 50, "min": 1, "max": 9999}),
                "debug_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("summary", "full_metadata",)
    OUTPUT_NODE = True
    FUNCTION = "read_metadata"
    CATEGORY = "CCN"

    def read_metadata(self, model_type, filename, show_tensors=True, max_tensors=50, debug_mode=False):
        # Resolve the file - support both dropdown selection and typed paths
        path = None
        try:
            file_list = folder_paths.get_filename_list(model_type)
            # Exact match first
            if filename in file_list:
                path = folder_paths.get_full_path(model_type, filename)
            else:
                # Partial/substring match
                matches = [f for f in file_list if filename.lower() in f.lower()]
                if len(matches) == 1:
                    path = folder_paths.get_full_path(model_type, matches[0])
                    filename = matches[0]
                elif len(matches) > 1:
                    match_list = "\n".join(f"  {m}" for m in matches[:20])
                    msg = f"Multiple matches for '{filename}':\n{match_list}"
                    return (msg, msg,)
        except Exception:
            pass

        if not path or not os.path.exists(path):
            # Try as absolute path
            if os.path.exists(filename):
                path = filename
            else:
                msg = f"File not found: '{filename}' in {model_type}"
                return (msg, msg,)

        if not path.endswith(".safetensors"):
            msg = f"Not a safetensors file: {filename}"
            return (msg, msg,)

        try:
            f = safetensors.safe_open(path, framework="pt")
            meta = f.metadata()
            tensor_keys = f.keys()
        except Exception as e:
            msg = f"Error reading file: {e}"
            return (msg, msg,)

        # --- Summary ---
        lines = []
        lines.append(f"=== Safetensors Info ===")
        lines.append(f"File: {filename}")
        lines.append(f"Tensor count: {len(tensor_keys)}")
        lines.append("")

        # --- Architecture detection ---
        tensor_key_list = list(tensor_keys)

        def get_shape(k):
            try:
                return list(f.get_slice(k).get_shape())
            except Exception:
                return None

        arch = detect_architecture(tensor_key_list, get_shape, metadata=meta, file_path=path)
        lines.append(format_detection(arch))
        lines.append("")

        # Metadata section
        if meta:
            lines.append("--- Metadata ---")
            # Check for kohya ss_ keys
            ss_keys = {k: v for k, v in meta.items() if k.startswith("ss_")}
            other_keys = {k: v for k, v in meta.items() if not k.startswith("ss_")}

            # Show non-ss keys first (format, description, etc.)
            for key in sorted(other_keys.keys()):
                if key == "__metadata__":
                    continue
                val = str(other_keys[key])
                if len(val) > 200:
                    val = val[:200] + "..."
                lines.append(f"  {key}: {val}")

            # Summarize kohya keys if present
            if ss_keys:
                lines.append("")
                lines.append("--- Training Info (kohya) ---")
                priority_keys = [
                    "ss_output_name", "ss_base_model_version",
                    "ss_network_module", "ss_network_dim", "ss_network_alpha",
                    "ss_learning_rate", "ss_unet_lr", "ss_text_encoder_lr",
                    "ss_num_train_images", "ss_num_epochs", "ss_steps",
                    "ss_caption_text", "ss_instance_prompt",
                    "ss_resolution", "ss_mixed_precision", "ss_seed",
                    "ss_training_comment",
                ]
                for key in priority_keys:
                    if key in ss_keys:
                        label = key.replace("ss_", "").replace("_", " ").title()
                        lines.append(f"  {label}: {ss_keys[key]}")

                # Tag frequency
                if "ss_tag_frequency" in ss_keys:
                    lines.append("")
                    lines.append("--- Tag Frequency (Top 20) ---")
                    try:
                        tag_freq = json.loads(ss_keys["ss_tag_frequency"])
                        merged = {}
                        for folder, tags in tag_freq.items():
                            for tag, count in tags.items():
                                merged[tag] = merged.get(tag, 0) + count
                        sorted_tags = sorted(merged.items(), key=lambda x: x[1], reverse=True)
                        for tag, count in sorted_tags[:20]:
                            lines.append(f"  {count:>4}x  {tag}")
                        if sorted_tags:
                            lines.append("")
                            lines.append(f"  ** Most likely trigger: \"{sorted_tags[0][0]}\" **")
                    except (json.JSONDecodeError, AttributeError):
                        lines.append("  (could not parse)")
        else:
            lines.append("No metadata embedded.")

        # Tensor structure for model identification
        if show_tensors:
            lines.append("")
            lines.append("--- Tensor Structure ---")

            # Collect dtype stats
            dtypes = {}
            shapes_by_prefix = {}
            all_tensors = []

            for key in tensor_keys:
                tensor = f.get_slice(key)
                shape = list(tensor.get_shape())
                dtype = str(tensor.dtype)

                dtypes[dtype] = dtypes.get(dtype, 0) + 1
                all_tensors.append((key, shape, dtype))

                # Group by top-level prefix for structure overview
                prefix = key.split(".")[0]
                if prefix not in shapes_by_prefix:
                    shapes_by_prefix[prefix] = []
                shapes_by_prefix[prefix].append((key, shape, dtype))

            # Dtype summary
            dtype_str = ", ".join(f"{d}: {c}" for d, c in sorted(dtypes.items()))
            lines.append(f"  Dtypes: {dtype_str}")
            lines.append("")

            # Show structure grouped by prefix
            lines.append(f"  Top-level modules:")
            for prefix in sorted(shapes_by_prefix.keys()):
                count = len(shapes_by_prefix[prefix])
                lines.append(f"    {prefix}: {count} tensors")

            # Show individual tensors up to limit
            lines.append("")
            display_count = min(len(all_tensors), max_tensors)
            lines.append(f"  Tensors ({display_count}/{len(all_tensors)}):")
            for key, shape, dtype in all_tensors[:max_tensors]:
                lines.append(f"    {key}: {shape} ({dtype})")
            if len(all_tensors) > max_tensors:
                lines.append(f"    ... and {len(all_tensors) - max_tensors} more")

        summary = "\n".join(lines)

        # --- Full metadata ---
        full_lines = [f"=== Full Metadata: {filename} ===", ""]
        if meta:
            for key in sorted(meta.keys()):
                if key == "__metadata__":
                    continue
                val_str = str(meta[key])
                if len(val_str) > 1000:
                    val_str = val_str[:1000] + "... (truncated)"
                full_lines.append(f"{key}: {val_str}")
        else:
            full_lines.append("No metadata.")

        full_metadata = "\n".join(full_lines)

        if debug_mode:
            print(summary)

        return (summary, full_metadata,)

    @staticmethod
    def _format_size(size_bytes):
        """Format bytes to human-readable size."""
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"
