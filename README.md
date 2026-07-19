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
- **[Utilities](https://valkymaera.github.io/ComfyCollectorNodes/nodes/utilities/)** — self-incrementing ints, rate-gated counters, named Property variables, timers, token counters, tensor inspection, and a Python Exec escape hatch.

## Install

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI. See
[Getting Started](https://valkymaera.github.io/ComfyCollectorNodes/getting-started/) for details.


# CCN's Signature nodes
These are some nodes I've gotten the most mileage out of, that might be of particular interest.

## Video Scrubber
The [video Scrubber](https://valkymaera.github.io/ComfyCollectorNodes/nodes/image/#video-scrubber) uploads or selects a video to input as normal, but allows you to scrub through to seek a specific single frame. 

<img width="797" height="744" alt="image" src="https://github.com/user-attachments/assets/88693843-9676-4817-a911-2775a5baffa1" />

This is for image extraction from a video, not video clipping. The other outputs are the single frame, the index of that frame, and the total frames in the video.
You can scrub in the timeline or step through with the arrows or seek directly by frame input.
There is a step value input that changes how many frames are skipped when you step manually.

The frame is an estimation (which is almost always going to be good enough), but if you need exactly the precise frame at the precise index, you can fetch it with the Load Exact Frame button,
which decodes the video up to that point to calculate it and caches it in your Input/Video Scrubber Frames folder.

## Cropped Image
The [Cropped Image](https://valkymaera.github.io/ComfyCollectorNodes/nodes/image/#cropped-image) node is a standard image input node that lets you visually define the cropped area.

<img width="435" height="695" alt="image" src="https://github.com/user-attachments/assets/029c3954-008a-4dc8-881e-cdbd5b7a1a0b" />

you can lock the ratio of the crop, drag in the center to move it, drag the corners to resize. It outputs the cropped image, or the raw_image (uncropped), or some details about the crop.
If you are wiring an image in, you can click "Load Preview" to 'pull' the image from upstream without having to execute the workflow, (if it exists).

This node prioritizes wired input, overriding any loaded image set in the widget. But if nothing is wired, it acts like a normal image input.
Note that for wired input from a video scrubber the pixel size preview at the bottom of the node will give you a smaller pixel size than the actual output.
This is because it uses the html preview of the connected node rather than interacting with the video. The output will still be the correct size.
If you load the exact frame in the video (which then gets cached), loading the preview will give you accurate size again. This info bug will not affect output.

## Image Inset
The [Image Inset](https://valkymaera.github.io/ComfyCollectorNodes/nodes/image/#image-inset) node places up to three images in the canvas of a base image.
You can rescale them and move them. By default the ratio is locked to the incoming image ratio. Each image gets its own rect.

<img width="442" height="675" alt="image" src="https://github.com/user-attachments/assets/593f119d-ee0c-44b9-b780-5643301c8aba" />

Like cropped image, you can treat this as a normal image node for the base image, or accept it via wire, and can load the preview with a button to 'pull' from upstream.
Inputs for embedded images that are disconnected are ignored and will not get a placement rectangle.

## All Together

<img width="2214" height="1198" alt="image" src="https://github.com/user-attachments/assets/9c067a1a-2a34-4130-89f6-2a6df5c35d92" />

Here's an example chaining the above together; scrubbing a video for the perfect frame to crop and inset the replacement image for (using another cropped image as the inset). 
The prompt asks to replace the pegasus with the spaceship.

<img width="1230" height="757" alt="image" src="https://github.com/user-attachments/assets/b1590006-7e8a-4a3a-9dfe-bdfd068bc957" />





