# ComfyCollectorNodes

A set of nodes I needed and that you can also have, too, as well.

Most of these nodes are tinker-related; normalization, scaling, latent channel adjustment, some custom loaders with QoL features.
There are a few that hit a good niche I think was missing, and I've detailed a few of those below.

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


Below are some nodes I've gotten the most mileage out of, that might be of particular interest.


# Signature Video/Image nodes

## Video Scrubber
The [video Scrubber](https://valkymaera.github.io/ComfyCollectorNodes/nodes/image/#video-scrubber) uploads or selects a video to input as normal, but allows you to scrub through to seek a specific single frame instead of a video clip. 

<img width="797" height="744" alt="image" src="https://github.com/user-attachments/assets/88693843-9676-4817-a911-2775a5baffa1" />

This is for image extraction from a video, not video clipping. The outputs are the single frame, the index of that frame, and the total frames in the video.
You can scrub in the timeline or step through with the arrows or seek directly by frame input.
There is a step value input that changes how many frames are skipped when you step manually.

The frame is an estimation (which is almost always going to be good enough), but if you need exactly the precise frame at the precise index, you can fetch it with the Load Exact Frame button,
which decodes the video up to that point to calculate it and caches it in your Input/Video Scrubber Frames folder.

## Cropped Image
The [Cropped Image](https://valkymaera.github.io/ComfyCollectorNodes/nodes/image/#cropped-image) node is like the standard image input node, but it lets you visually define the cropped area.
*Important: cropped images are stored in your comfyui temp folder* for use in execution, which is cleared the next time you start comfyui.

<img width="435" height="695" alt="image" src="https://github.com/user-attachments/assets/029c3954-008a-4dc8-881e-cdbd5b7a1a0b" />

you can lock the ratio of the crop, drag in the center to move it, drag the corners to resize. It outputs the cropped image, or the raw_image (cropped but not resampled), or the source image, or some details about the crop.
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

# Signature Prompt nodes

This comes with three base "prompt stores". Each is primarily a helper that outputs a structured prompt from text input placed in categories. 
By default it automatically adds the category names to the prompt ahead of the text, but this can be disabled. For example, whatever you put in the 'metadata' section will be prepended by "metadata: " followed by your text.

These nodes provide extra value in that they store your prompt in session memory (not a file), associated with the category and the store name.
So if you have a store named "action_shots" and you set the mood category to "dark, gritty, and chaotic", the next time you use a prompt store of the same name it will use that value.
If the mode is set to override, then putting something else in the mood category will overwrite it. If it's set to merge, then it will split your input at the separator (comma by default), 
and append it minus anything that already exists. And append simply adds it to what exists.

<img width="1747" height="1158" alt="image" src="https://github.com/user-attachments/assets/3fb859d3-76ab-4bca-bb56-45ed97cdcdd8" />

The storage is per session, not per workflow, so you can retrieve it across many workflows until comfy is restarted or you clear it yourself.

At any time you can access any category you stored by using a PromptStoreGet node.
If you want to load all the categories you stored for a store name, you can just use an empty prompt store node's outpout, with the appropriate name (just don't set it to clear).

There is a simpler variant called "PromptBuilder" that does not store anything to memory, just provides an easy way to block out prompts into categories.

# Signature Curve
This package has curves, curve evaluation, and curve guiders that work for nodes 1.0.
I recognize that there are now curves native in nodes 2.0. But there weren't when I started this. So... now there are more options. For QoL there is a curve converter between comfy's and CCN curves.

<img width="830" height="933" alt="image" src="https://github.com/user-attachments/assets/67509de4-69bf-4adc-92a8-91d4ab27420e" />

There are three curve nodes apart from the converters.
One is just a curve itself, which can be handed off to the other two.
Another is a sampler which just samples a normalized point along the curve to provide a corresponding float value.
The last is a curve-integrated CFG guider to use with custom samplers. Each has its own curve widget or can take a wired-in one.


# Special Condition Tinkering

## Neutral Prompt Nodes
These nodes are a conceptual port of the "Neutral Prompt" mechanism from Ijleb under the MIT License: https://github.com/ljleb/sd-webui-neutral-prompt which I used a ton in Automatic1111.
The nodes are model agnostic but some will respond better than others at various weights.

This allows powerful orthagonal prompt/conditioning combination instead of a basic merge. 
Somewhat oversimplifying but basically: 
Perpendicular mode zeroes the dot product of the auxiliary prompt, basically removing overlap or conflict. 
Salient mode gives priority for elements to the prompt that seems to care the most about it, using the weight to determine how much the aux is applied where it wins.
Top-K mode selects only the strongest activations of the auxiliary conditioning to merge into the main on top (not replacing). 

The results grant special tinker-level ability to blend concepts and inject details.
This package suite comes with single node application of a neutral prompt strategy (which can be chained) as well as a 'Neutral Prompt Entry' where an auxiliary conditioning
can be added to a growing queue of strategic applications, and a neutral prompt guider that applies these directly to provide a guider and sigmas for custom samplers (with curve sampling).

## Hyper-remap
A multifunctional prompt and condition tinkering node. It has up to four layers of modification with different abstraction from the original prompt. This is a tinkering node, the results
will vary depending on the model. It is primarily used as an experimenting surface, since it modifies concepts and tokens which can vary in results from model to model.
Note that except for string replacement, all of these require re-encoding conditioning in multiple passes. For most things this is pretty fast, but for some vision-encoding models this may add 
noticeable seconds to your workflow execution time.

All the entries in the hyper-remap are separated by semicolons or newlines, and there are some per-entry means of controling the blend strength, threshold, and sharpness, including intrinsic values for delta remapping, described in the [actual documentation](https://valkymaera.github.io/ComfyCollectorNodes/nodes/conditioning/#hyper-remap).

### String replace
The first, simplest use is string replace. Comma separated values swap the first value with the second in the prompt directly. "red, blue" replaces "red" with "blue". Note this is plain substring replacement, so partial words match too ("red, blue" will happily turn "hatred" into "hatblue"). A case sensitivity toggle is available for this phase. This one is consistent across models, naturally.

<img width="680" height="220" alt="string_replace" src="https://github.com/user-attachments/assets/aeb56f08-74f5-4096-b84c-850b99cbd4ef" />

### Token Remap 
Weighted token remapping uses arrow pairs (source -> target) to blend embeddings at the changed positions. The prompt is encoded once as written, then once more per remap pair with the swap applied, and the results are blended in embedding space. The text itself is NOT modified. This lets you land partway between red and blue rather than swapping one word out entirely. Unlike string replace, this matches whole words only. Because it operates on tokens rather than words, some use cases may not have the intended effect, and some models or CLIP formats may be resilient to it.

<img width="680" height="290" alt="token_remap" src="https://github.com/user-attachments/assets/0ff7b019-742e-4192-aca3-25493caac9c3" />

### Concept Remap
Fat-arrow pairs (source => target) nudge the conditioning along a concept direction (the vector from the source concept toward the target concept). To figure out where to apply that nudge, the node works in one of two ways. If the source word actually appears in your prompt, it measures the influence directly: the prompt is encoded with and without the word, and wherever the encoding changed, that's where the concept lives, including all the contextual bleed from attention, like reflections, lighting, or palette. If the source word isn't in your prompt, it falls back to an approximation, using similarity between each position of the conditioning and the source concept. That fallback is looser, but it means you can remap concepts that are only implied, like shifting "gloomy => cheerful" on a prompt that never says gloomy. Either way, because this remaps the concept rather than individual tokens, it can affect related and adjacent elements of the scene along the way, like mood, composition, and setting details.

<img width="680" height="320" alt="concept_remap" src="https://github.com/user-attachments/assets/bffdb6e3-4f68-4caa-bf95-e19cac34557b" />

### Delta Remap
Two tildes specify a special delta remap as A\~\~B. This takes two arbitrary prompts A and B on either side of the tildes (they don't have to be related to your main prompt at all). Both are encoded, and their difference becomes a delta: roughly, "A without B". That delta is then added into your prompt's conditioning to nudge it in that abstract direction. For example, "beach\~\~bright tropical summer" produces the beach-ness left over once the bright tropical summer component is stripped out, and adds that to the outbound conditioning.

<img width="680" height="432" alt="delta_remap" src="https://github.com/user-attachments/assets/5ae4901b-7ffd-4d7e-b3c4-69a4d2df937d" />


The delta isn't dumped in uniformly. By default it's normalized so the blend strength behaves consistently no matter how different A and B are, and it's weighted two ways: toward the positions where A and B differ most (so the nudge focuses on what actually distinguishes them), and toward the parts of your prompt's conditioning that relate to A (so it lands where it's relevant). The sharpness and threshold controls govern that second layer; the first has its own per-pair overrides if you want to tune or disable it. This aspect is highly experimental. The original purpose was to experiment with bringing out details that a model may have knowledge of without being given token data for.

For example, imagine a model that was never trained on flowers and plants or any words related to flowers and plants, but it was trained on many images of bees in the wild.
We know that the flowers themselves do exist in the data, just in the context of photos of bees only, not appropriately labeled as plants.
If you wanted to generate an image that was simply a flower, how would you go about doing that? 
This delta remapping is an early experimental exploration in just that: can we take the vector difference of "macrophotography of a bee in the wild", and subtract "Bee, insect", 
apply that delta to the conditioning and increase the weight of the flower that would remain in the image?

The answer is: Sometimes. Kinda. It works well for some models and not so well for others. 
The manifold of where training has reliable results from tensor values can be sensitive, and sometimes applying a delta can push the context into a less defined space.

I am still exploring the space, but overall the concept and delta remaps have provided a soft helper for models that don't support actual negative conditioning.








