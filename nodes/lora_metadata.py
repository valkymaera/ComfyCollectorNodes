"""
LoRA Metadata Reader - extracts metadata from safetensors LoRA files
Surfaces training parameters, trigger words, and tag frequencies
across multiple training tools (kohya, OneTrainer, EveryDream2,
ai-toolkit/Ostris, SimpleTuner, diffusers, modelspec, etc.)
"""

import json
import folder_paths
import safetensors
from .detect_architecture import detect_architecture, format_detection


class LoraMetadata:
    """Reads and displays metadata embedded in safetensors LoRA files.

    Scans for metadata conventions from all major training tools and
    surfaces trigger words, training parameters, and tag frequencies.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
            },
            "optional": {
                "debug_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("summary", "full_metadata",)
    OUTPUT_NODE = True
    FUNCTION = "read_metadata"
    CATEGORY = "CCN"

    # Organized by trainer/convention, each entry is (key, display_label)
    # Order within each group reflects importance
    KNOWN_KEYS = {
        "modelspec": [
            ("modelspec.title", "Title"),
            ("modelspec.architecture", "Architecture"),
            ("modelspec.trigger_phrase", "Trigger Phrase"),
            ("modelspec.resolution", "Resolution"),
            ("modelspec.description", "Description"),
            ("modelspec.author", "Author"),
            ("modelspec.tags", "Tags"),
            ("modelspec.date", "Date"),
            ("modelspec.version", "Version"),
            ("modelspec.prediction_type", "Prediction Type"),
            ("modelspec.implementation", "Implementation"),
            ("modelspec.hash_sha256", "SHA256"),
        ],
        "kohya": [
            ("ss_output_name", "Output Name"),
            ("ss_base_model_version", "Base Model"),
            ("ss_network_module", "Network Module"),
            ("ss_network_dim", "Network Dim (Rank)"),
            ("ss_network_alpha", "Network Alpha"),
            ("ss_learning_rate", "Learning Rate"),
            ("ss_unet_lr", "UNet LR"),
            ("ss_text_encoder_lr", "Text Encoder LR"),
            ("ss_num_train_images", "Train Images"),
            ("ss_num_epochs", "Epochs"),
            ("ss_steps", "Steps"),
            ("ss_max_train_steps", "Max Train Steps"),
            ("ss_caption_text", "Caption Text"),
            ("ss_instance_prompt", "Instance Prompt"),
            ("ss_resolution", "Resolution"),
            ("ss_mixed_precision", "Mixed Precision"),
            ("ss_optimizer", "Optimizer"),
            ("ss_lr_scheduler", "LR Scheduler"),
            ("ss_seed", "Seed"),
            ("ss_training_comment", "Training Comment"),
            ("ss_sd_model_name", "SD Model Name"),
            ("ss_clip_skip", "CLIP Skip"),
            ("ss_noise_offset", "Noise Offset"),
            ("ss_prior_loss_weight", "Prior Loss Weight"),
            ("ss_v2", "SD v2"),
            ("ss_v_parameterization", "V-Parameterization"),
            ("ss_min_snr_gamma", "Min SNR Gamma"),
            ("ss_network_args", "Network Args"),
            ("ss_reg_dataset_dirs", "Regularization Dirs"),
        ],
        "everydream2": [
            ("ed2_trainer_version", "Trainer Version"),
            ("ed2_learning_rate", "Learning Rate"),
            ("ed2_resolution", "Resolution"),
            ("ed2_batch_size", "Batch Size"),
            ("ed2_optimizer", "Optimizer"),
            ("ed2_steps", "Steps"),
            ("ed2_seed", "Seed"),
        ],
        "ai-toolkit": [
            ("toolkit_type", "Toolkit Type"),
            ("toolkit_config", "Config"),
            ("toolkit_name", "Name"),
            ("toolkit_version", "Version"),
        ],
        "simpletuner": [
            ("simpletuner_version", "Version"),
            ("simpletuner_model_type", "Model Type"),
            ("simpletuner_trigger_phrase", "Trigger Phrase"),
        ],
        "diffusers": [
            ("format", "Format"),
            ("framework", "Framework"),
            ("pipeline_tag", "Pipeline Tag"),
        ],
    }

    # Keys across any trainer that might contain trigger words
    TRIGGER_KEYS = [
        "modelspec.trigger_phrase",
        "simpletuner_trigger_phrase",
        "ss_instance_prompt",
        "ss_caption_text",
        "ss_training_comment",
        "trigger_word",
        "trigger_phrase",
        "trigger",
        "activation_text",
    ]

    def read_metadata(self, lora_name, debug_mode=False):
        path = folder_paths.get_full_path("loras", lora_name)

        try:
            f = safetensors.safe_open(path, framework="pt")
            meta = f.metadata()
            tensor_keys = list(f.keys())
        except Exception as e:
            msg = f"Error reading metadata: {e}"
            return (msg, msg,)

        lines = []
        lines.append("=== LoRA Metadata Summary ===")
        lines.append(f"File: {lora_name}")
        lines.append("")

        # --- Architecture detection (always available) ---
        def get_shape(k):
            try:
                return list(f.get_slice(k).get_shape())
            except Exception:
                return None

        arch = detect_architecture(tensor_keys, get_shape, metadata=meta, file_path=path)
        lines.append(format_detection(arch))
        lines.append("")

        if not meta:
            lines.append("No training metadata embedded.")
            summary = "\n".join(lines)
            if debug_mode:
                print(summary)
            return (summary, summary,)

        # --- Trigger word detection (top priority) ---
        triggers = self._find_triggers(meta)
        if triggers:
            lines.append("*** TRIGGER WORDS ***")
            for source, value in triggers:
                lines.append(f"  {value}  (from {source})")
            lines.append("")

        # --- Scan each trainer convention ---
        found_any = False
        for trainer, keys in self.KNOWN_KEYS.items():
            matched = []
            for key, label in keys:
                if key in meta:
                    val = str(meta[key])
                    if len(val) > 300:
                        val = val[:300] + "..."
                    matched.append(f"  {label}: {val}")

            if matched:
                found_any = True
                lines.append(f"--- {trainer} ---")
                lines.extend(matched)
                lines.append("")

        # --- Tag frequency (kohya) ---
        if "ss_tag_frequency" in meta:
            self._append_tag_frequency(lines, meta)

        # --- Dataset directories (kohya) ---
        if "ss_dataset_dirs" in meta:
            self._append_dataset_dirs(lines, meta)

        # --- Catch any unrecognized keys ---
        known_prefixes = ("ss_", "ed2_", "toolkit_", "simpletuner_", "modelspec.")
        known_exact = set()
        for keys in self.KNOWN_KEYS.values():
            for key, _ in keys:
                known_exact.add(key)
        known_exact.update(self.TRIGGER_KEYS)
        known_exact.update(["__metadata__", "ss_tag_frequency", "ss_dataset_dirs"])

        unknown = {}
        for key in meta:
            if key in known_exact:
                continue
            if any(key.startswith(p) for p in known_prefixes):
                continue
            unknown[key] = meta[key]

        if unknown:
            lines.append("--- Other ---")
            for key in sorted(unknown.keys()):
                val = str(unknown[key])
                if len(val) > 300:
                    val = val[:300] + "..."
                lines.append(f"  {key}: {val}")
            lines.append("")

        if not found_any and not triggers:
            lines.append("No recognized training metadata found.")
            lines.append("Check full_metadata output for raw key/value pairs.")

        summary = "\n".join(lines)

        # --- Full metadata ---
        full_lines = ["=== Full LoRA Metadata ===", f"File: {lora_name}", ""]
        for key in sorted(meta.keys()):
            val_str = str(meta[key])
            if len(val_str) > 500:
                val_str = val_str[:500] + "... (truncated)"
            full_lines.append(f"{key}: {val_str}")

        full_metadata = "\n".join(full_lines)

        if debug_mode:
            print(summary)

        return (summary, full_metadata,)

    def _find_triggers(self, meta):
        """Search all known trigger word locations across trainers."""
        triggers = []
        seen = set()

        # Check explicit trigger keys
        for key in self.TRIGGER_KEYS:
            if key in meta and meta[key].strip():
                val = meta[key].strip()
                if val not in seen:
                    seen.add(val)
                    triggers.append((key, val))

        # Check kohya dataset dir names (often "N_triggername" format)
        if "ss_dataset_dirs" in meta:
            try:
                dirs = json.loads(meta["ss_dataset_dirs"])
                for dirname in dirs:
                    # kohya convention: "num_repeats_conceptname"
                    parts = dirname.split("_", 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        concept = parts[1]
                        if concept not in seen:
                            seen.add(concept)
                            triggers.append(("dataset_dir", concept))
            except (json.JSONDecodeError, AttributeError):
                pass

        # Check tag frequency for most common tag
        if "ss_tag_frequency" in meta:
            try:
                tag_freq = json.loads(meta["ss_tag_frequency"])
                merged = {}
                for folder, tags in tag_freq.items():
                    for tag, count in tags.items():
                        merged[tag] = merged.get(tag, 0) + count
                if merged:
                    top_tag = max(merged, key=merged.get)
                    if top_tag not in seen:
                        seen.add(top_tag)
                        triggers.append(("top_tag", top_tag))
            except (json.JSONDecodeError, AttributeError):
                pass

        return triggers

    def _append_tag_frequency(self, lines, meta):
        """Parse and format kohya tag frequency data."""
        lines.append("--- Tag Frequency (Top 20) ---")
        try:
            tag_freq = json.loads(meta["ss_tag_frequency"])
            merged = {}
            for folder, tags in tag_freq.items():
                lines.append(f"  Dataset: {folder}")
                for tag, count in tags.items():
                    merged[tag] = merged.get(tag, 0) + count

            sorted_tags = sorted(merged.items(), key=lambda x: x[1], reverse=True)
            for tag, count in sorted_tags[:20]:
                lines.append(f"    {count:>4}x  {tag}")
        except (json.JSONDecodeError, AttributeError):
            lines.append("  (could not parse tag frequency)")
        lines.append("")

    def _append_dataset_dirs(self, lines, meta):
        """Parse and format kohya dataset directory info."""
        try:
            dataset_dirs = json.loads(meta["ss_dataset_dirs"])
            lines.append("--- Dataset Directories ---")
            for dirname, info in dataset_dirs.items():
                lines.append(f"  {dirname}")
                if isinstance(info, dict):
                    for k, v in info.items():
                        lines.append(f"    {k}: {v}")
            lines.append("")
        except (json.JSONDecodeError, AttributeError):
            pass
