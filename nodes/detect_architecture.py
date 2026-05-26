"""
Model Architecture Detection - identifies model type from safetensors tensor structure
Examines key names, shapes, structural patterns, and optional metadata to fingerprint
the architecture.
"""

import os


def detect_architecture(keys, get_shape=None, metadata=None, file_path=None):
    """Detect model architecture from tensor key names, shapes, and metadata.

    Args:
        keys: list of tensor key names
        get_shape: optional callable(key) -> list[int] to get tensor shapes
        metadata: optional dict of safetensors metadata (ss_base_model_version, etc.)
        file_path: optional file path for size reporting

    Returns:
        dict with:
            - model_type: str (e.g. "SDXL LoRA", "Wan 14B DiT", "SD 1.5 VAE")
            - details: list[str] of detected properties
            - confidence: str ("high", "medium", "low")
    """
    key_set = set(keys)
    key_list = list(keys)
    details = []

    # --- File size ---
    if file_path and os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        details.append(f"File size: {_format_size(file_size)}")

    # --- Strip common wrapper prefixes for structural analysis ---
    WRAPPER_PREFIXES = [
        "model.diffusion_model.",
        "diffusion_model.",
        "base_model.model.",
    ]
    wrapper_prefix = None
    if key_list:
        for wp in WRAPPER_PREFIXES:
            if key_list[0].startswith(wp):
                wrapper_prefix = wp
                break

    if wrapper_prefix:
        details.append(f"Key prefix: {wrapper_prefix[:-1]}")
        stripped_keys = [k[len(wrapper_prefix):] for k in key_list]
        stripped_to_original = {k[len(wrapper_prefix):]: k for k in key_list}
        original_get_shape = get_shape
        if get_shape:
            def get_shape(k):
                orig_key = stripped_to_original.get(k, k)
                return original_get_shape(orig_key)
        key_list = stripped_keys
        key_set = set(stripped_keys)

    # --- Check metadata for base model version ---
    meta_base_model = None
    if metadata:
        meta_base_model = (
            metadata.get("ss_base_model_version", "") or
            metadata.get("modelspec.architecture", "")
        ).strip().lower()

    # --- Detect if this is a LoRA ---
    is_lora = any(
        frag in k for k in key_list[:20]
        for frag in ("lora_A", "lora_B", "lora_down", "lora_up",
                      "lora.down", "lora.up", "to_k_lora", "to_q_lora")
    )

    # --- Detect LoRA format ---
    lora_format = None
    if is_lora:
        sample = key_list[0] if key_list else ""
        if "lora_A.default" in sample or "lora_B.default" in sample:
            lora_format = "diffusers/PEFT"
        elif "lora_down" in sample or "lora_up" in sample:
            lora_format = "kohya"
        elif "to_k_lora.down" in sample or "to_k_lora.up" in sample:
            lora_format = "diffusers (old)"
        elif "lora_A" in sample:
            lora_format = "diffusers/PEFT"

        if lora_format:
            details.append(f"LoRA format: {lora_format}")

    # --- Detect rank from LoRA shapes ---
    if is_lora and get_shape:
        for k in key_list[:10]:
            if any(frag in k for frag in ("lora_A", "lora_down", "lora.down", "to_k_lora.down")):
                shape = get_shape(k)
                if shape and len(shape) == 2:
                    rank = min(shape)
                    details.append(f"Rank: {rank}")
                    break

    # --- Collect hidden dims from shapes ---
    hidden_dims = set()
    cross_attn_dim = None
    if get_shape:
        for k in key_list:
            shape = get_shape(k)
            if shape and len(shape) == 2:
                for dim in shape:
                    if dim in (640, 768, 1024, 1280, 1536, 2048, 3072, 3840):
                        hidden_dims.add(dim)
                # Cross-attention key dim (text encoder output)
                if cross_attn_dim is None:
                    if ("attn2" in k or "cross" in k) and ("to_k" in k or "k_proj" in k):
                        cross_attn_dim = max(shape)
            # Early out once we have enough info
            if cross_attn_dim and len(hidden_dims) >= 3:
                break

    # --- Architecture detection by key patterns ---

    # Check for top-level prefixes
    prefixes = set()
    for k in key_list:
        prefixes.add(k.split(".")[0])

    has_down_blocks = "down_blocks" in prefixes
    has_up_blocks = "up_blocks" in prefixes
    has_mid_block = "mid_block" in prefixes
    has_encoder = "encoder" in prefixes
    has_decoder = "decoder" in prefixes
    has_layers = "layers" in prefixes
    has_context_refiner = "context_refiner" in prefixes
    has_noise_refiner = "noise_refiner" in prefixes
    has_cap_embedder = "cap_embedder" in prefixes
    has_t_embedder = "t_embedder" in prefixes
    has_x_embedder = "x_embedder" in prefixes
    has_transformer_blocks = any("transformer_blocks" in k for k in key_list)
    has_single_transformer = any("single_transformer_blocks" in k for k in key_list)
    has_adaln = any("adaLN_modulation" in k for k in key_list)
    has_feed_forward_w3 = any("feed_forward.w3" in k for k in key_list)
    has_temporal = any("temporal" in k for k in key_list)
    has_quant_conv = "quant_conv" in prefixes or "post_quant_conv" in prefixes

    # Count layers
    layer_indices = set()
    for k in key_list:
        if k.startswith("layers."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_indices.add(int(parts[1]))

    # Count transformer blocks depth
    max_tb = -1
    for k in key_list:
        if "transformer_blocks." in k:
            idx = k.find("transformer_blocks.")
            if idx >= 0:
                rest = k[idx + len("transformer_blocks."):]
                num = rest.split(".")[0]
                if num.isdigit():
                    max_tb = max(max_tb, int(num))

    # --- Identify architecture ---

    # === VAE ===
    if has_encoder and has_decoder and has_quant_conv:
        latent_channels = None
        if get_shape:
            for k in key_list:
                if "post_quant_conv.weight" in k:
                    shape = get_shape(k)
                    if shape:
                        latent_channels = shape[0]
                        break

        # Check for 3D convolutions (causal VAE)
        has_3d_conv = False
        if get_shape:
            for k in key_list[:100]:
                if "conv" in k and "weight" in k:
                    shape = get_shape(k)
                    if shape and len(shape) == 5:
                        has_3d_conv = True
                        break

        if has_3d_conv:
            details.append("3D Causal VAE")
            if latent_channels == 16:
                return _result("Wan 3D-VAE", details, "high")
            return _result("3D Causal VAE (unknown)", details, "medium")

        channel_prog = _get_channel_progression(key_list, get_shape)
        if channel_prog:
            details.append(f"Channel progression: {channel_prog}")

        if latent_channels:
            details.append(f"Latent channels: {latent_channels}")

        if latent_channels == 4:
            return _result("SD 1.5 / SDXL VAE", details, "high")
        elif latent_channels == 16:
            return _result("Flux / SD3 VAE", details, "high")

        return _result("VAE (unknown variant)", details, "medium")

    # === Wan (with unique modules) ===
    if has_context_refiner or has_noise_refiner or has_cap_embedder:
        num_layers = len(layer_indices) if layer_indices else 0
        details.append(f"DiT layers: {num_layers}")

        if has_feed_forward_w3:
            details.append("SwiGLU FFN (w1/w2/w3)")
        if has_context_refiner:
            details.append("Context refiner blocks")
        if has_noise_refiner:
            details.append("Noise refiner blocks")

        if 3840 in hidden_dims:
            variant = "Wan 14B"
            details.append("Hidden dim: 3840")
        elif 1536 in hidden_dims:
            variant = "Wan 1.3B"
            details.append("Hidden dim: 1536")
        else:
            variant = "Wan (unknown size)"

        suffix = " LoRA" if is_lora else " DiT"
        return _result(f"{variant}{suffix}", details, "high")

    # === Flux / SD3 ===
    if has_single_transformer and has_transformer_blocks:
        details.append("Dual-stream + single-stream blocks (Flux)")
        if 3072 in hidden_dims:
            details.append("Hidden dim: 3072")
        suffix = " LoRA" if is_lora else ""
        return _result(f"Flux{suffix}", details, "high")

    if has_transformer_blocks and has_adaln and not has_down_blocks:
        if 3072 in hidden_dims:
            details.append("Hidden dim: 3072")
            if has_temporal:
                suffix = " LoRA" if is_lora else ""
                return _result(f"HunyuanVideo{suffix}", details, "medium")
            suffix = " LoRA" if is_lora else ""
            return _result(f"SD3 / Flux-like DiT{suffix}", details, "medium")

    # === DiT with layers.* — Wan 14B / Z-Image / other DiTs ===
    if has_layers and has_adaln and not has_down_blocks:
        num_layers = len(layer_indices) if layer_indices else 0
        details.append(f"DiT layers: {num_layers}")
        if has_feed_forward_w3:
            details.append("SwiGLU FFN (w1/w2/w3)")

        if 3840 in hidden_dims:
            details.append("Hidden dim: 3840")

            # Try metadata to disambiguate
            if meta_base_model:
                model_name = _identify_from_metadata(meta_base_model)
                if model_name:
                    suffix = " LoRA" if is_lora else ""
                    return _result(f"{model_name}{suffix}", details, "high")

            # Without metadata, these architectures are structurally ambiguous
            suffix = " LoRA" if is_lora else " DiT"
            return _result(f"Wan 14B / Z-Image{suffix}", details, "medium")

        elif 1536 in hidden_dims:
            details.append("Hidden dim: 1536")
            suffix = " LoRA" if is_lora else " DiT"
            return _result(f"Wan 1.3B{suffix}", details, "high")

        else:
            for dim in sorted(hidden_dims, reverse=True):
                details.append(f"Hidden dim: {dim}")
                break
            suffix = " LoRA" if is_lora else " DiT"
            return _result(f"DiT (unknown){suffix}", details, "low")

    # === SDXL / SD 1.5 / SD 2.x UNet ===
    if has_down_blocks and has_up_blocks and has_mid_block:
        if cross_attn_dim == 2048:
            details.append("Cross-attn dim: 2048 (dual CLIP)")
            if max_tb >= 9:
                details.append(f"Max transformer depth: {max_tb + 1}")
            suffix = " LoRA" if is_lora else " UNet"
            return _result(f"SDXL{suffix}", details, "high")

        if cross_attn_dim == 768:
            details.append("Cross-attn dim: 768 (CLIP ViT-L)")
            suffix = " LoRA" if is_lora else " UNet"
            return _result(f"SD 1.5{suffix}", details, "high")

        if cross_attn_dim == 1024:
            details.append("Cross-attn dim: 1024 (OpenCLIP ViT-H)")
            suffix = " LoRA" if is_lora else " UNet"
            return _result(f"SD 2.x{suffix}", details, "high")

        # Fallback: use hidden dims
        if 1280 in hidden_dims and 640 in hidden_dims:
            if 2048 in hidden_dims:
                suffix = " LoRA" if is_lora else " UNet"
                return _result(f"SDXL{suffix}", details, "medium")
            suffix = " LoRA" if is_lora else " UNet"
            return _result(f"SD 1.5 / SD 2.x{suffix}", details, "medium")

        suffix = " LoRA" if is_lora else " UNet"
        return _result(f"UNet (unknown){suffix}", details, "low")

    # === LTX-Video ===
    if has_temporal and has_transformer_blocks and has_down_blocks:
        details.append("Temporal transformer blocks in UNet")
        suffix = " LoRA" if is_lora else ""
        return _result(f"LTX-Video{suffix}", details, "medium")

    # === Text encoder ===
    has_text_model = "text_model" in prefixes
    has_embeddings = any("embeddings" in k for k in key_list)
    if has_text_model and has_embeddings:
        if any("encoder.block" in k for k in key_list):
            suffix = " LoRA" if is_lora else ""
            return _result(f"T5 Text Encoder{suffix}", details, "medium")
        suffix = " LoRA" if is_lora else ""
        return _result(f"CLIP Text Encoder{suffix}", details, "medium")

    # === Fallback ===
    if hidden_dims:
        dim_str = ", ".join(str(d) for d in sorted(hidden_dims, reverse=True))
        details.append(f"Hidden dims: {dim_str}")

    # Last resort: check metadata
    if meta_base_model:
        model_name = _identify_from_metadata(meta_base_model)
        if model_name:
            suffix = " LoRA" if is_lora else ""
            return _result(f"{model_name}{suffix}", details, "medium")

    suffix = " LoRA" if is_lora else ""
    return _result(f"Unknown architecture{suffix}", details, "low")


# --- Known base model version strings from training tools ---
KNOWN_BASE_MODELS = {
    # Wan
    "wan": "Wan",
    "wan2.1": "Wan 2.1",
    "wan2.2": "Wan 2.2",
    "wan_video": "Wan Video",
    "wan_image": "Wan Image",
    # Z-Image
    "zimage": "Z-Image",
    "z-image": "Z-Image",
    "zimage_turbo": "Z-Image Turbo",
    "z-image-turbo": "Z-Image Turbo",
    # SDXL
    "sdxl": "SDXL",
    "sdxl_base": "SDXL",
    "sdxl_base_v1-0": "SDXL 1.0",
    # SD 1.x
    "sd1": "SD 1.5",
    "sd15": "SD 1.5",
    "sd_v1": "SD 1.5",
    "stable_diffusion_v1": "SD 1.5",
    # SD 2.x
    "sd2": "SD 2.x",
    "sd_v2": "SD 2.x",
    # SD 3
    "sd3": "SD3",
    "sd3_medium": "SD3 Medium",
    # Flux
    "flux": "Flux",
    "flux1": "Flux.1",
    "flux_dev": "Flux.1 Dev",
    "flux_schnell": "Flux.1 Schnell",
    # HunyuanVideo
    "hunyuan_video": "HunyuanVideo",
    "hunyuanvideo": "HunyuanVideo",
    # LTX
    "ltx_video": "LTX-Video",
    "ltxvideo": "LTX-Video",
}


def _identify_from_metadata(base_model_str):
    """Try to identify model from ss_base_model_version or modelspec.architecture."""
    if not base_model_str:
        return None

    normalized = base_model_str.lower().strip()

    # Exact match first
    if normalized in KNOWN_BASE_MODELS:
        return KNOWN_BASE_MODELS[normalized]

    # Substring match
    for key, name in KNOWN_BASE_MODELS.items():
        if key in normalized or normalized in key:
            return name

    # Return the raw string capitalized as a fallback
    return base_model_str.strip()


def _result(model_type, details, confidence):
    return {
        "model_type": model_type,
        "details": details,
        "confidence": confidence,
    }


def _get_channel_progression(key_list, get_shape):
    """Extract channel sizes from down_blocks to identify VAE variant."""
    if not get_shape:
        return None
    channels = []
    for k in key_list:
        if "down_blocks" in k and "conv1.weight" in k and "resnets.0" in k:
            shape = get_shape(k)
            if shape and len(shape) >= 2:
                channels.append(shape[0])
    if channels:
        return " → ".join(str(c) for c in channels)
    return None


def _format_size(size_bytes):
    """Format bytes to human-readable size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def format_detection(result):
    """Format detection result as a readable string block."""
    conf_icon = {"high": "●", "medium": "◐", "low": "○"}.get(result["confidence"], "?")
    lines = []
    lines.append(f"*** Detected: {result['model_type']} [{conf_icon} {result['confidence']} confidence] ***")
    for d in result["details"]:
        lines.append(f"  {d}")
    return "\n".join(lines)
