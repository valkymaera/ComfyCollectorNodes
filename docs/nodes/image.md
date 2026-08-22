# Image & Video

Nodes for resizing, blending, cropping, rotating, stitching, and compositing
images, computing dimensions, and loading images and video frames.

## At a glance

| Node | Summary |
|------|---------|
| [Resize By Shorter Edge](#resize-by-shorter-edge) | Resizes so the shorter edge hits a target size, keeping aspect and snapping to multiples of 8. |
| [Resize To Match](#resize-to-match) | Resizes images to exactly match a reference image's dimensions. |
| [Image Blend](#image-blend) | Linearly blends two image batches, auto-resizing the second to match. |
| [Cropped Image](#cropped-image) | Interactive crop tool — drag a rectangle on a canvas in the node, get the crop, mask, and geometry. |
| [Image Inset](#image-inset) | Composites up to three images onto a base via draggable placement rectangles. |
| [Image Stitch](#image-stitch) | Stitches up to three images side by side or stacked, scaling each to match the first, with an optional divider. |
| [Rotate Image](#rotate-image) | Interactive rotation — drag on a canvas to rotate around a draggable pivot, with a mask of the uncovered areas. |
| [Image Canvas](#image-canvas) | Outputs a solid-color image of a given size — a blank base the canvas tools can preview without executing. |
| [Dimension Scale](#dimension-scale) | Computes scaled width/height numbers relative to a reference resolution — no image needed. |
| [Image Loader By Index](#image-loader-by-index) | Loads one image from a folder by sorted index, wrapping past the end. |
| [Video Loader By Index](#video-loader-by-index) | Loads a video from a folder by index and decodes its frames to an image batch. |
| [Video Scrubber](#video-scrubber) | Scrub through a video visually in the node and output a single chosen frame. |

---

## Resizing & blending

### Resize By Shorter Edge

**Resizes images or video frames so the shorter edge hits a target size,
preserving aspect ratio and flooring both dimensions to a multiple of 8.**

Robust to the assorted tensor layouts and dtypes that come out of video
sources (uint8 is converted, channel-first tensors are detected and fixed).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | IMAGE | — | Batch to resize. |
| `target_size` | INT | 1080 | Target length of the shorter edge (`64`–`8192`), before the ÷8 floor. |
| `interpolation` | choice | `bilinear` | `bilinear`, `bicubic`, `nearest`, or `area`. `area` only downscales; it falls back to bilinear when upscaling. |

**Outputs:** `images` (resized, clamped to `[0,1]`).

### Resize To Match

**Resizes images to exactly the height and width of a reference image.**

Handy for aligning an upscaled or processed result back to the original
resolution before blending. No ÷8 snapping — the output matches the reference
exactly; only the reference's dimensions are used, not its content.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | IMAGE | — | Images to resize. |
| `reference` | IMAGE | — | Image whose dimensions become the target. |
| `interpolation` | choice | `bilinear` | Resampling mode (same options and `area` caveat as above). |

**Outputs:** `images`.

### Image Blend

**Linearly blends two image batches: `a·(1 − blend) + b·blend`.**

Typical use is mixing a processed result with the original to restore color or
detail. If resolutions differ, `image_b` is resized to match `image_a`; if
batch sizes differ, both are truncated to the smaller.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_a` | IMAGE | — | The base; also defines the output resolution. |
| `image_b` | IMAGE | — | Blended in; auto-resized to A if needed. |
| `blend` | FLOAT | 0.5 | `0.0` = all A, `1.0` = all B. |

**Outputs:** `image`.

---

## Interactive tools

### Cropped Image

**An interactive visual crop tool: drag a rectangle on a canvas inside the
node, and get back a model-friendly crop, the exact raw crop, a mask, and the
crop geometry.**

The node shows the source image on a canvas with a dimmed overlay outside the
crop. Drag the four corner handles to resize (the opposite corner stays
fixed), or drag inside the rectangle to move it. A live readout shows the
crop's pixel size and aspect ratio. With `lock_ratio` on, corner drags keep a
fixed aspect ratio, and the crop is rebuilt at that ratio when a
different-sized image loads.

It can also act as a standalone image loader: when nothing is wired to
`image`, use **Upload Image** or the `loaded_image` picker to crop a file from
the input directory. **Load Preview** fetches a backdrop from the wired
upstream's preview or the last run; the backdrop never changes just because a
job was queued, so your crop stays put. When wired to another CCN node, the
backdrop reflects the specific output the wire comes from — `source_image`
yields the uncropped source, `image`/`raw_image` the crop. Crop coordinates are stored as
normalized 0–1 values and serialize with the workflow.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lock_ratio` | BOOLEAN | false | Lock the crop's aspect ratio while resizing and across image swaps. |
| `snap_to` | INT | 8 | The model-friendly output's dimensions are floored to this multiple (`1`–`64`). |
| `image` | IMAGE | *(optional)* | Wired source; takes priority over `loaded_image`. |
| `loaded_image` | choice | `none` | File from the input directory to use when no image is wired (supports upload). |
| `debug` | BOOLEAN | false | Print crop diagnostics. |

*(Four hidden `crop_x1/y1/x2/y2` widgets hold the normalized rectangle; the
canvas manages them.)*

**Outputs:**

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | Model-friendly crop, resized so width/height are multiples of `snap_to`. |
| `raw_image` | IMAGE | The exact pixel crop, no resampling. |
| `mask` | MASK | Full source-size mask: `1.0` inside the crop, `0.0` outside. |
| `crop_x`, `crop_y` | INT | Top-left corner of the crop in source pixels. |
| `crop_width`, `crop_height` | INT | Raw crop size in pixels. |
| `source_image` | IMAGE | The original uncropped source image (passthrough). |

<!-- TODO: screenshot — Cropped Image node with the canvas, corner handles, and dimmed overlay -->

### Image Inset

**Composites up to three images onto a base image, each placed via its own
draggable colored rectangle on an interactive canvas.**

The three optional `embed` inputs map to colored rectangles — embed1 red,
embed2 green, embed3 blue. Each active embed's thumbnail is drawn stretched
into its rectangle, so the canvas is a true preview of the final composite.
Embeds are pasted in order 1 → 2 → 3, so blue lands on top where rectangles
overlap. Corner handles resize, interior drags move, and with `lock_ratio` on
(the default) each rectangle keeps its embed's true aspect so nothing
distorts. Newly connected embeds drop in at staggered default positions;
**Reset Placements** re-staggers them.

An optional border (`border_width` / `border_color`) is drawn inside each
rectangle: the border occupies the rectangle's outer N pixels and the embed
scales to fill the inner box, so the footprint never changes and the whole
embed stays visible. Oversized borders clamp so at least 1 px of embed
remains; with `lock_ratio` on, a fat border on a small rectangle distorts the
embed marginally (the inner box's aspect differs slightly from the
rectangle's).

Like [Cropped Image](#cropped-image), the base can be wired in or loaded from
the input directory, the backdrop only changes on explicit action, and all
placements are stored as normalized coordinates that serialize with the
workflow.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lock_ratio` | BOOLEAN | true | Constrain each rectangle to its embed's pixel aspect while resizing. |
| `image` | IMAGE | *(optional)* | Base image; wired takes priority over `loaded_image`. |
| `embed1`–`embed3` | IMAGE | *(optional)* | Images composited into the red/green/blue rectangles. |
| `loaded_image` | choice | `none` | Base file from the input directory when nothing is wired (supports upload). |
| `debug` | BOOLEAN | false | Print base/embed diagnostics. |
| `border_width` | INT | 0 | Border thickness in base-image pixels, drawn inside each rectangle; one setting shared by all embeds. 0 = none. |
| `border_color` | STRING | `#000000` | Hex border color (`#` optional, 3- or 6-digit). |

*(Twelve hidden `embedN_x1/y1/x2/y2` widgets hold the normalized rectangles.)*

**Outputs:** `compilation` (the composite), `base` (the resolved base image,
untouched).

Embeds are treated as opaque — an RGBA embed's alpha channel is dropped, not
composited. For a batched base, a single-image embed is reused across every
frame.

<!-- TODO: screenshot — Image Inset with three colored placement rectangles over a base image -->

### Image Stitch

**Stitches up to three images together side by side or stacked, scaling each
to match the first, with an optional colored divider between them.**

Connected inputs join in slot order 1 → 2 → 3; unconnected slots are simply
skipped. The first connected image sets the shared edge — heights match for
`horizontal`, widths for `vertical` — and the others are scaled to it with
aspect ratio preserved. The divider (`border_width` / `border_color`) is
drawn between adjacent images only, never as an outer frame; with a single
image connected the node is a passthrough.

The canvas shows a live composed preview: **Load Preview** pulls each wired
upstream's preview (or the thumbnail the node saved on its last run) and
stitches them client-side, so direction and divider changes are visible
without executing the graph. A wired slot with no preview available yet is
skipped in the composition and named in the status line. Upstream previews
may be downscaled thumbnails, so the status line's absolute pixel counts can
undershoot the real output — the composed geometry stays aspect-correct
regardless.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `direction` | choice | `horizontal` | `horizontal` joins left-to-right; `vertical` stacks top-to-bottom. |
| `image1`–`image3` | IMAGE | *(optional)* | Images to stitch, in slot order. At least one must be wired. |
| `border_width` | INT | 0 | Divider thickness in pixels between adjacent images (`0`–`1024`). 0 = none. |
| `border_color` | STRING | `#000000` | Hex divider color (`#` optional, 3- or 6-digit). |
| `debug` | BOOLEAN | false | Print size/border/output diagnostics. |

*(No hidden widgets — the canvas is display-only.)*

**Outputs:** `image` (the stitched result).

Images are treated as opaque — an RGBA input's alpha channel is dropped. The
output batch matches the largest connected batch; shorter batches repeat
their last frame.

<!-- TODO: screenshot — Image Stitch canvas showing three images side by side with a divider -->

### Rotate Image

**An interactive visual rotation tool: drag on a canvas inside the node to
rotate the image live around a draggable pivot, and get back the rotated
image plus a mask of the uncovered areas for inpainting.**

The node shows the image on a canvas. Drag anywhere to rotate (hold Shift to
snap to 15° steps) and drag the crosshair to place the pivot. With `expand`
off the output keeps the input frame — content rotates around the pivot,
corners crop away, and areas the image no longer covers are painted
`fill_color`. With `expand` on the output grows to the rotated bounding box
and the pivot has no effect; exact quarter turns in this mode are bit-exact
(no resampling). The `mask` output is white wherever the fill shows —
including the bounding-box corners in expand mode — ready to drive
inpainting of the uncovered regions.

Like [Cropped Image](#cropped-image), it can act as a standalone image
loader: when nothing is wired to `image`, upload a file or pick one from the
input directory via `loaded_image`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `angle` | FLOAT | 0.0 | Rotation in degrees, positive = clockwise (`-360`–`360`). |
| `expand` | BOOLEAN | false | Off: keep the input dimensions. On: grow to the rotated bounding box. |
| `fill_color` | STRING | `#000000` | 3- or 6-digit hex color painted where the image no longer covers the frame. |
| `interpolation` | choice | `bilinear` | `bilinear`, `nearest`, or `bicubic`. |
| `image` | IMAGE | *(optional)* | Wired source; takes priority over `loaded_image`. |
| `loaded_image` | choice | `none` | File from the input directory when no image is wired (supports upload). |
| `debug` | BOOLEAN | false | Print source/angle/output diagnostics. |

*(Two hidden `pivot_x`/`pivot_y` widgets hold the normalized pivot; the
canvas manages them.)*

**Outputs:** `image` (rotated), `mask` (white where the fill shows), `source_image`
(the unrotated input, passthrough).

<!-- TODO: screenshot — Rotate Image canvas mid-drag with the pivot crosshair visible -->

### Image Canvas

**Outputs a solid-color image of the given width, height, and hex color — a
blank canvas base for the interactive image tools.**

The node has no canvas of its own, but it participates in the CCN preview
convention: when wired into Cropped Image, Image Inset, Image Stitch, or
Rotate Image, their **Load Preview** button renders the flat color
client-side from the current widget values — no execution needed, and the
preview always matches what the next run will output.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width`, `height` | INT | 1024 | Output dimensions in pixels (`1`–`8192`). |
| `color` | STRING | `#000000` | Hex fill color (`#` optional, 3- or 6-digit). |

**Outputs:** `image` (a single solid-color frame).

### Dimension Scale

**Computes scaled width/height numbers from input dimensions relative to a
reference resolution — a pure number utility, no image tensor required.**

Use it to drive latent or resize dimensions in a graph. It derives a scale
factor from the input vs. reference dimensions per the chosen mode, applies
it, and floors both results to a multiple of `round_to`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width`, `height` | INT | 1024 | Input dimensions to scale. |
| `ref_width`, `ref_height` | INT | 1920 × 1080 | Reference resolution. |
| `scale_type` | choice | `smart_scale` | See modes below. |
| `round_to` | INT | 8 | Floor outputs to this multiple (`1`–`64`). |

**Modes:**

- `scale_width` — match the reference width; height follows proportionally.
- `scale_height` — match the reference height; width follows.
- `match_exact` — output exactly the reference resolution (may distort aspect).
- `smart_scale` — scale by whichever axis needs the *least* change, keeping
  proportions.

**Outputs:** `width`, `height`, `info` (a description of what the mode did,
including the factor and percent deltas).

---

## Loading

### Image Loader By Index

**Loads one image from a directory by its position in a sorted listing,
wrapping past the end — built for iterating a folder of reference images
across runs.**

Also extracts the alpha channel as a mask (inverted to ComfyUI's convention)
and reports filename, path, and count metadata. Re-executes automatically when
the selected file changes on disk.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directory` | STRING | `""` | Absolute path to the image folder. |
| `recursive` | BOOLEAN | false | Walk nested subfolders. |
| `index` | INT | 0 | Zero-based position; wraps modulo the file count. |
| `debug` | BOOLEAN | false | Print wrap/loading info. |

**Outputs:** `image`, `mask` (from alpha, or all zeros), `filename`,
`file_path`, `total_files`, `actual_index`, `wrapped`.

**File types:** png, jpg, jpeg, webp, bmp, tiff, tif.

### Video Loader By Index

**Loads a video from a directory by sorted index and decodes its frames into a
single batched IMAGE tensor.**

Supports frame skipping and a max-frame cap to control how much is extracted.
Requires OpenCV (`opencv-python-headless`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directory` | STRING | `""` | Path to the video folder. |
| `recursive` | BOOLEAN | false | Walk nested subfolders. |
| `index` | INT | 0 | Zero-based position; wraps modulo the file count. |
| `frame_skip` | INT | 0 | Skip N frames between extracted frames (`0` = every frame, `1` = every other…). |
| `max_frames` | INT | 0 | Cap on extracted frames; `0` = all. |
| `debug` | BOOLEAN | false | Print loading and shape info. |

**Outputs:** `frames` (batched IMAGE), `filename`, `file_path`, `frame_count`
(after skip/cap), `fps`, `total_files`, `actual_index`, `wrapped`.

**File types:** mp4, avi, mov, mkv, webm, gif, m4v, wmv, flv.

### Video Scrubber

**Scrub through a video with an in-node slider and preview, then output the
single frame you land on — for example to branch a new generation from a
chosen split point.**

Videos live in ComfyUI's `input` directory (the node has its own **Upload
Video** button that handles large files the stock uploader rejects). Picking
a video that has already been uploaded — even from its original folder —
retargets the node to the existing copy instantly instead of re-uploading it;
nothing in the input directory is ever overwritten. The widget shows a live
preview that updates as you drag the slider or step with the ◀ ▶ buttons
(step size is configurable), with a frame counter and dimensions readout.

Scrubbing uses fast keyframe seeks, which are quick but codec-approximate.
**Load Exact Frame** does a frame-accurate decode of the current position and
caches it as a full-resolution PNG — once cached, execution outputs those
exact pixels. **Clear Frame Cache** removes the cached PNGs. The scrub
position and preview are restored when a workflow is reloaded.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | choice | — | A video from the input directory (mp4, avi, mov, mkv, webm, gif). |
| `scrub_frame` | INT | 0 | The frame to output; clamped to the video's range. |
| `frame_step` | INT | 1 | How many frames the ◀ ▶ buttons jump (UI only). |

**Outputs:** `image` (the single chosen frame), `frame_index` (the clamped
index actually loaded), `total_frames`.

At queue time the node loads the cached exact-frame PNG when one is valid for
the current frame, and otherwise falls back to the same fast seek the preview
uses. Cached frames invalidate automatically if the source video is replaced.
Requires OpenCV.

<!-- TODO: screenshot — Video Scrubber with slider, preview, and Load Exact Frame button -->
