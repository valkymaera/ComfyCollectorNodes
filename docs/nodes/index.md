# Nodes

Reference documentation for every node in the pack, grouped by category. Each
category page opens with an at-a-glance table — one or two lines per node —
linking to the full entries.

All nodes appear in the ComfyUI menu with a `(CCN)` suffix.

## Categories

### [Conditioning](conditioning.md)

Creating, shaping, editing, combining, and inspecting `CONDITIONING` — encode
replacements (Token Remap, Emphasis Encode), embedding-space concept editing
(Concept Remap, Hyper Remap, Projection Removal), tensor math (Scale,
Normalizer, Clamp), combining (Lerp, Subtract), and inspection (Stats, Token
Inspector).

### [Sampling & Guidance](sampling.md)

Changing how the sampler is guided: the visual curve editor and
curve-scheduled CFG (Curve, Curve Sample, Curve CFG Guider), CFG-Zero*
Scaled, and the Neutral Prompt family for merging auxiliary prompts into the
CFG step.

### [LoRA](lora.md)

Loading LoRAs — by index or sorted/filtered — and file tools:
LoRA Scale & Save, LoRA Truncate Rank, and the LoRA / Safetensors metadata
inspectors.

### [Image & Video](image.md)

Resize, blend, and dimension helpers; the interactive Cropped Image and Image
Inset canvas tools; and the image/video loaders including the Video Scrubber.

### [Latent](latent.md)

Latent value shaping (Clamp, Scale, Normalize, Stats) and per-channel
offset/scale nodes for 4- and 16-channel latents.

### [Prompt & Text](prompt.md)

Prompt Builders, the persistent Prompt Store family, Compound Prompt mode
switching, and the string manipulation nodes.

### [Utilities](utilities.md)

Workflow plumbing: Better Int and Gated Increment counters, Property
variables, Random Select, Float Lerp, timers, Print, Inspect Tensor, JSON
loaders, and token counters.
