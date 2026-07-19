# ComfyCollectorNodes

A set of nodes I needed and that you can also have, too, as well.

**📖 Full documentation: [valkymaera.github.io/ComfyCollectorNodes](https://valkymaera.github.io/ComfyCollectorNodes/)**

All nodes appear in the ComfyUI menu with a `(CCN)` suffix.

## What's inside

- **[Conditioning](https://valkymaera.github.io/ComfyCollectorNodes/nodes/conditioning/)** — edit prompts in embedding space: Token/Concept/Hyper Remap for blending words toward other meanings, Projection Removal as a negative tinker node for flow models, plus scale/normalize/clamp/lerp/subtract and inspection tools.
- **[Sampling & Guidance](https://valkymaera.github.io/ComfyCollectorNodes/nodes/sampling/)** — a visual curve editor driving CFG across sampling steps (Curve CFG Guider), CFG-Zero* Scaled, and the Neutral Prompt family for merging auxiliary prompts without fighting the main one.
- **[LoRA](https://valkymaera.github.io/ComfyCollectorNodes/nodes/lora/)** — load by index (batch sweeps), sorted-dropdown loading, multi-LoRA stacking with normalization, and file tools: bake in strength, truncate rank, inspect metadata/trigger words.
- **[Image & Video](https://valkymaera.github.io/ComfyCollectorNodes/nodes/image/)** — interactive crop and inset-compositing canvases, resize/blend helpers, index-based image/video loaders, and a Video Scrubber that picks an exact frame with an in-node preview.
- **[Latent](https://valkymaera.github.io/ComfyCollectorNodes/nodes/latent/)** — clamp/scale/normalize latents and adjust individual channels (4- and 16-channel variants).
- **[Prompt & Text](https://valkymaera.github.io/ComfyCollectorNodes/nodes/prompt/)** — structured prompt builders, persistent prompt stores that accumulate across runs, and string utilities.
- **[Utilities](https://valkymaera.github.io/ComfyCollectorNodes/nodes/utilities/)** — self-incrementing ints, rate-gated counters, named Property variables, timers, token counters, and tensor inspection.

## Install

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI. See
[Getting Started](https://valkymaera.github.io/ComfyCollectorNodes/getting-started/) for details.
