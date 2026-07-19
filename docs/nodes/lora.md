# LoRA

Nodes for loading LoRAs — by index or filtered — and for
working with LoRA files themselves: rescaling, shrinking, and inspecting them.

## At a glance

| Node | Summary |
|------|---------|
| [LoRA Loader By Index](#lora-loader-by-index) | Loads a LoRA by its numeric position in a sorted directory listing — built for sweeping a collection across runs. |
| [LoRA Loader Filtered](#lora-loader-filtered) | The built-in LoRA loader with a dropdown sortable by date, name, or size (newest-first by default). |
| [LoRA List Directory](#lora-list-directory) | Lists the LoRA files in a folder plus a count — a companion to index-based loading. |
| [LoRA Scale & Save](#lora-scale-save) | Bakes a strength change into a LoRA (via alpha or weights) and saves it as a new file. |
| [LoRA Truncate Rank](#lora-truncate-rank) | Shrinks an SVD-extracted LoRA by slicing it to a lower rank — fast, no re-decomposition. |
| [LoRA Metadata](#lora-metadata) | Reads a LoRA's embedded training metadata: trigger words, base model, rank, tags, and more. |
| [Safetensors Metadata](#safetensors-metadata) | Generic safetensors inspector for any model type — metadata plus tensor structure. |

---

## Loading

### LoRA Loader By Index

**Loads a LoRA chosen by its numeric position in an alphabetically sorted
directory listing, wrapping past the end — built for sweeping a whole
collection across queued runs.**

Point it at the `loras` folder (or a subfolder) and increment `index` each run
to iterate every LoRA in turn. It searches every configured LoRA path,
including `extra_model_paths.yaml` entries, and wraps the index modulo the
file count instead of erroring — with a console banner when a full pass
completes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | — | Base model to apply the LoRA to. |
| `subdirectory` | STRING | `""` | Subfolder relative to a LoRA base path (can be nested, e.g. `wan/characters`). Blank = the base folder itself. |
| `recursive` | BOOLEAN | false | Include nested subfolders (files listed by relative path). |
| `index` | INT | 0 | Zero-based position in the sorted list. Wraps modulo the file count. |
| `strength_model` | FLOAT | 1.0 | LoRA strength on the model weights (`-20`–`20`). |
| `strength_clip` | FLOAT | 1.0 | LoRA strength on CLIP; only used when `clip` is connected. |
| `clip` | CLIP | *(optional)* | Optional CLIP to also patch. |

**Outputs:** `model`, `clip` (or `None` if not connected), `lora_name`,
`total_loras`, `actual_index` (post-wrap), `wrapped` (true if the index
wrapped).

**File types:** `.safetensors`, `.pt`, `.bin`, `.ckpt`. Raises an error if the
directory can't be found or contains no LoRA files.

<!-- TODO: screenshot — LoRA Loader By Index wired with an incrementing counter -->

### LoRA Loader Filtered

**A faithful clone of the built-in Load LoRA node whose dropdown can be sorted
by modification date, name, or size — newest-first by default.**

The sort is applied client-side in the picker, so grabbing the LoRA you just
downloaded or trained is one click. Text search is the combo widget's native
type-to-filter. Loading behavior matches the built-in node exactly, including
the tensor cache and the zero-strength short-circuit.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | — | Model to patch. |
| `clip` | CLIP | — | CLIP to patch. |
| `lora_name` | choice | — | The LoRA file; the dropdown is reordered per the sort controls. |
| `strength_model` | FLOAT | 1.0 | Model strength. |
| `strength_clip` | FLOAT | 1.0 | CLIP strength. |
| `sort_by` | choice | `date_modified` | `date_modified`, `name`, or `size` — display order only. |
| `sort_order` | choice | `descending` | Sort direction — display order only. |

**Outputs:** `model`, `clip`.

If both strengths are `0.0` the inputs pass through untouched with no file
load. File dates/sizes come from a small server endpoint and are cached in the
browser, refreshed only when new files appear.

### LoRA List Directory

**Lists the LoRA files in a directory as a newline-separated string, plus a
count.**

A companion to [LoRA Loader By Index](#lora-loader-by-index): check what's in
a folder and how many items there are before setting up index-based iteration.
Only the first configured LoRA path is searched.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subdirectory` | STRING | `""` | Subfolder relative to the LoRA base directory. |
| `recursive` | BOOLEAN | false | Walk nested subfolders (relative paths). |

**Outputs:** `lora_names` (newline-joined), `total_count`.

---

## File tools

These read or write LoRA files on disk. The save nodes always write a *new*
file to the ComfyUI output directory with an auto-incrementing counter —
existing files are never overwritten and the source LoRA is never modified.

### LoRA Scale & Save

**Bakes a strength change into a LoRA file — by scaling its alphas, setting
them outright, or scaling the weights — and saves the result as a new
safetensors file.**

A LoRA's contribution is `(alpha / rank) · (up @ down)`, so its effective
strength can be pre-baked either through the alpha scalars or the weight
tensors. Use this to permanently tone a LoRA up or down, or to add alpha
values to a LoRA that lacks them.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lora_name` | choice | — | Source LoRA from the `loras` folder. |
| `filename_prefix` | STRING | `loras/CCN_scaled_lora` | Output path + prefix, relative to the output directory. |
| `mode` | choice | `scale_alpha` | See modes below. |
| `value` | FLOAT | 1.0 | Meaning depends on mode (`-10`–`10`). |

**Modes:**

- `scale_alpha` — multiply every layer's alpha by `value`. Layers with *no*
  alpha get one created as `rank × value`, so this mode can add alphas to a
  LoRA that had none.
- `set_alpha` — write `value` as the exact alpha for every layer, replacing
  existing alphas.
- `scale_weights` — multiply the `lora_up` weights (and any `diff`/`diff_b`
  tensors) by `value`, leaving alphas alone.

**Outputs:** `filepath` (the absolute path of the new file).

fp8 tensors are upcast for the multiply and cast back, preserving their dtype.
Training metadata is **not** carried over to the new file.

### LoRA Truncate Rank

**Reduces a LoRA's rank by slicing off the least-significant components — fast
(seconds, no SVD re-decomposition) and produces a proportionally smaller
file.**

!!! warning "SVD-extracted LoRAs only"
    Slicing is only mathematically valid for **SVD-extracted** LoRAs (e.g.
    from a checkpoint-diff extract), where components are stored in descending
    order of importance. *Trained* LoRAs don't order their rank dimensions, so
    truncating one discards arbitrary components and degrades quality
    unpredictably. The node does not check this for you.

Alphas are rescaled automatically (`alpha × new_rank / old_rank`) so the
LoRA's effective strength is preserved. Layers already at or below the target
rank pass through unchanged. Training metadata is preserved and annotated with
the truncation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lora_name` | choice | — | Source LoRA (kohya, PEFT, and ControlLoRA key formats supported). |
| `new_rank` | INT | 32 | Target rank (`1`–`4096`); per-layer the kept rank is `min(new_rank, old_rank)`. |
| `filename_prefix` | STRING | `loras/CCN_truncated_lora` | Output path + prefix relative to the output directory. |
| `output_dtype` | choice | `match_original` | `match_original`, `fp16`, `bf16`, `fp32`, `fp8_e4m3`, `fp8_e5m2`. fp8 roughly halves size vs fp16. |
| `verbose` | BOOLEAN | true | Print per-layer rank changes and a size summary. |

**Outputs:** `filepath`. The filename encodes the change, e.g.
`CCN_truncated_lora_trunc128to32_fp8_e4m3_00001.safetensors`, and the console
reports the before/after file size.

### LoRA Metadata

**Reads a LoRA file's embedded training metadata and surfaces it in a
human-readable report — trigger words, base model, rank/alpha, learning rates,
tag frequencies, and a detected architecture line.**

It understands the metadata conventions of many trainers (kohya `ss_*`,
modelspec, EveryDream2, ai-toolkit/Ostris, SimpleTuner, diffusers) and
highlights trigger words gathered from several locations, including dataset
folder names and the most frequent training tag. Use it to figure out how to
actually use a downloaded LoRA. Read-only — nothing is modified.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lora_name` | choice | — | LoRA file to inspect. |
| `debug_mode` | BOOLEAN | false | Also print the summary to the console. |

**Outputs:** `summary` (the curated, sectioned report), `full_metadata` (a raw
alphabetical dump of every metadata key).

<!-- TODO: screenshot — LoRA Metadata summary output showing trigger words and kohya section -->

### Safetensors Metadata

**A generic safetensors inspector for any model type — checkpoint, VAE, LoRA,
CLIP, ControlNet, UNet, upscaler — reporting metadata plus the tensor
structure.**

Where [LoRA Metadata](#lora-metadata) curates training metadata for LoRAs
specifically, this node works on any safetensors file and adds a structural
overview: tensor count, dtype histogram, top-level module grouping, and
individual tensor shapes. It runs the same architecture fingerprinting
(detecting SD1.5/SDXL/Flux/SD3/Wan/HunyuanVideo/LTX families, LoRA formats,
VAE types, text encoders) — handy for identifying an unknown file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | choice | — | Which ComfyUI model folder to look in (loras, checkpoints, vae, clip, unet, controlnet, …). |
| `filename` | STRING | `""` | The file. Exact match, unique case-insensitive substring match, or an absolute path all work. |
| `show_tensors` | BOOLEAN | true | Include the tensor-structure section. |
| `max_tensors` | INT | 50 | Cap on individually listed tensors. |
| `debug_mode` | BOOLEAN | false | Also print the summary to the console. |

**Outputs:** `summary`, `full_metadata`.
