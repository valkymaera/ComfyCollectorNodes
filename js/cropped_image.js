import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/*
 * Interactive crop overlay for CCN_CroppedImage.
 *
 * Renders the source image on a canvas with a draggable rectangle.
 * Crop coordinates are stored as normalized 0-1 values in hidden
 * widgets that the Python node reads at execution time.
 *
 * When a new image with different dimensions loads, the crop rectangle
 * preserves its pixel-space aspect ratio, its size relative to the
 * image's longest edge, and its normalized center position.
 *
 * Image sources (in priority order):
 *   1. onExecuted callback — preview saved by the Python node
 *   2. "Load Preview" button — fetches upstream node's cached output
 *   3. "Upload Image" button — browser file picker, copies to input dir
 *   4. loaded_image combo — auto-loads when selection changes
 */

const CORNER_HIT_RADIUS = 12;
const HANDLE_SIZE = 8;
const CANVAS_HEIGHT = 350;

app.registerExtension({
    name: "CCN.CroppedImage",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "CCN_CroppedImage") return;

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origOnExecuted?.apply(this, arguments);
            if (this._ccnCrop && message?.images?.[0]) {
                const img = message.images[0];
                this._ccnCrop.loadImage(
                    api.apiURL(
                        `/view?filename=${encodeURIComponent(img.filename)}` +
                        `&type=${img.type}&subfolder=${img.subfolder || ""}`
                    )
                );
            }
        };
    },

    async nodeCreated(node) {
        if (node.comfyClass !== "CCN_CroppedImage") return;

        // ----------------------------------------------------------------
        //  Hidden crop widgets — still serialize, but invisible in the UI
        // ----------------------------------------------------------------

        const HIDDEN = ["crop_x1", "crop_y1", "crop_x2", "crop_y2"];
        const cw = {};
        for (const w of node.widgets) {
            if (HIDDEN.includes(w.name)) {
                cw[w.name] = w;
                w.type = "hidden";
                w.computeSize = () => [0, -4];
            }
        }

        function getCrop() {
            return {
                x1: cw.crop_x1?.value ?? 0,
                y1: cw.crop_y1?.value ?? 0,
                x2: cw.crop_x2?.value ?? 1,
                y2: cw.crop_y2?.value ?? 1,
            };
        }

        function setCrop(x1, y1, x2, y2) {
            const cl = (v) => Math.max(0, Math.min(1, v));
            if (cw.crop_x1) cw.crop_x1.value = cl(x1);
            if (cw.crop_y1) cw.crop_y1.value = cl(y1);
            if (cw.crop_x2) cw.crop_x2.value = cl(x2);
            if (cw.crop_y2) cw.crop_y2.value = cl(y2);
        }

        // ----------------------------------------------------------------
        //  State
        // ----------------------------------------------------------------

        let previewImg = null;
        let imgW = 0;
        let imgH = 0;
        let dragState = null;
        let lockedRatio = null;

        // ----------------------------------------------------------------
        //  Crop stabilization across image dimension changes.
        //
        //  When a new image loads with different dimensions we convert the
        //  crop rectangle from normalized coords back to pixel-space using
        //  the OLD dimensions, then re-normalize into the NEW dimensions
        //  while preserving the crop's aspect ratio, its size relative to
        //  the longest image edge, and its center position.
        // ----------------------------------------------------------------

        function adaptCropToNewDimensions(oldW, oldH, newW, newH) {
            // First load or same dimensions — nothing to adapt
            if (!oldW || !oldH || (oldW === newW && oldH === newH)) return;

            const c = getCrop();

            // Pixel-space crop dimensions on the old image
            const pxW = (c.x2 - c.x1) * oldW;
            const pxH = (c.y2 - c.y1) * oldH;
            const cropRatio = pxW / Math.max(pxH, 1e-6);

            // Size as fraction of the old image's longest edge
            const oldLong = Math.max(oldW, oldH);
            const newLong = Math.max(newW, newH);
            const cropLongest = Math.max(pxW, pxH);
            const sizeFrac = cropLongest / oldLong;

            // Rebuild crop dimensions in the new image's pixel space
            let newPxLongest = sizeFrac * newLong;
            let newPxW, newPxH;
            if (pxW >= pxH) {
                newPxW = newPxLongest;
                newPxH = newPxW / cropRatio;
            } else {
                newPxH = newPxLongest;
                newPxW = newPxH * cropRatio;
            }

            // Clamp to new image bounds
            newPxW = Math.min(newPxW, newW);
            newPxH = Math.min(newPxH, newH);

            // Preserve normalized center, clamp so rectangle stays in bounds
            const cx = (c.x1 + c.x2) / 2;
            const cy = (c.y1 + c.y2) / 2;
            const halfNW = (newPxW / newW) / 2;
            const halfNH = (newPxH / newH) / 2;

            const clampedCx = Math.max(halfNW, Math.min(1 - halfNW, cx));
            const clampedCy = Math.max(halfNH, Math.min(1 - halfNH, cy));

            setCrop(
                clampedCx - halfNW,
                clampedCy - halfNH,
                clampedCx + halfNW,
                clampedCy + halfNH,
            );
        }

        // ----------------------------------------------------------------
        //  Coordinate conversion
        // ----------------------------------------------------------------

        const cvs = document.createElement("canvas");

        // Rectangle the image occupies inside the canvas, preserving aspect ratio
        function displayRect() {
            const W = cvs.width, H = cvs.height;
            if (!imgW || !imgH) return { x: 0, y: 0, w: W, h: H };
            const iA = imgW / imgH, cA = W / H;
            let dw, dh, dx, dy;
            if (iA > cA) { dw = W; dh = W / iA; dx = 0; dy = (H - dh) / 2; }
            else          { dh = H; dw = H * iA; dx = (W - dw) / 2; dy = 0; }
            return { x: dx, y: dy, w: dw, h: dh };
        }

        function toNorm(cx, cy) {
            const r = displayRect();
            return {
                x: Math.max(0, Math.min(1, (cx - r.x) / r.w)),
                y: Math.max(0, Math.min(1, (cy - r.y) / r.h)),
            };
        }

        function toCanvas(nx, ny) {
            const r = displayRect();
            return { x: r.x + nx * r.w, y: r.y + ny * r.h };
        }

        // ----------------------------------------------------------------
        //  Drawing
        // ----------------------------------------------------------------

        function draw() {
            const ctx = cvs.getContext("2d");
            const W = cvs.width, H = cvs.height;
            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = "#1a1a1a";
            ctx.fillRect(0, 0, W, H);

            if (!previewImg) {
                ctx.fillStyle = "#666";
                ctx.font = "13px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(
                    "No preview \u2014 Load Preview, Upload Image, or queue once",
                    W / 2, H / 2,
                );
                return;
            }

            const r = displayRect();
            ctx.drawImage(previewImg, r.x, r.y, r.w, r.h);

            const crop = getCrop();
            const tl = toCanvas(crop.x1, crop.y1);
            const br = toCanvas(crop.x2, crop.y2);
            const cW = br.x - tl.x;
            const cH = br.y - tl.y;

            // Dark overlay outside the crop region
            ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
            ctx.fillRect(r.x, r.y,   r.w, tl.y - r.y);          // top
            ctx.fillRect(r.x, br.y,  r.w, r.y + r.h - br.y);    // bottom
            ctx.fillRect(r.x, tl.y,  tl.x - r.x, cH);           // left
            ctx.fillRect(br.x, tl.y, r.x + r.w - br.x, cH);     // right

            // Border
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 1.5;
            ctx.strokeRect(tl.x, tl.y, cW, cH);

            // Corner handles
            ctx.fillStyle = "#fff";
            const hs = HANDLE_SIZE;
            for (const pt of [tl, {x: br.x, y: tl.y}, {x: tl.x, y: br.y}, br]) {
                ctx.fillRect(pt.x - hs / 2, pt.y - hs / 2, hs, hs);
            }

            // Pixel-dimension label
            if (imgW && imgH) {
                const pw = Math.round((crop.x2 - crop.x1) * imgW);
                const ph = Math.round((crop.y2 - crop.y1) * imgH);
                const label = `${pw} \u00d7 ${ph}`;
                ctx.font = "11px monospace";
                const tm = ctx.measureText(label);
                const lx = (tl.x + br.x) / 2;
                const ly = br.y + 16;
                ctx.fillStyle = "rgba(0,0,0,0.7)";
                ctx.fillRect(lx - tm.width / 2 - 4, ly - 11, tm.width + 8, 15);
                ctx.fillStyle = "#fff";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(label, lx, ly - 3);
            }
        }

        // ----------------------------------------------------------------
        //  Hit detection
        // ----------------------------------------------------------------

        function cornerPositions() {
            const c = getCrop();
            return {
                tl: toCanvas(c.x1, c.y1),
                tr: toCanvas(c.x2, c.y1),
                bl: toCanvas(c.x1, c.y2),
                br: toCanvas(c.x2, c.y2),
            };
        }

        function hitCorner(cx, cy) {
            const pts = cornerPositions();
            for (const name of ["tl", "tr", "bl", "br"]) {
                const p = pts[name];
                if (Math.hypot(cx - p.x, cy - p.y) <= CORNER_HIT_RADIUS) return name;
            }
            return null;
        }

        function hitInterior(cx, cy) {
            const c = getCrop();
            const n = toNorm(cx, cy);
            return n.x > c.x1 && n.x < c.x2 && n.y > c.y1 && n.y < c.y2;
        }

        function cursorFor(corner) {
            return (corner === "tl" || corner === "br")
                ? "nwse-resize"
                : "nesw-resize";
        }

        // ----------------------------------------------------------------
        //  Mouse coordinates
        // ----------------------------------------------------------------

        function mousePos(e) {
            const rect = cvs.getBoundingClientRect();
            return {
                x: (e.clientX - rect.left) * (cvs.width / rect.width),
                y: (e.clientY - rect.top) * (cvs.height / rect.height),
            };
        }

        // ----------------------------------------------------------------
        //  Drag logic
        // ----------------------------------------------------------------

        function onMouseDown(e) {
            if (!previewImg) return;
            const pos = mousePos(e);

            const corner = hitCorner(pos.x, pos.y);
            if (corner) {
                const lockW = node.widgets.find(w => w.name === "lock_ratio");
                const locked = lockW?.value ?? false;

                if (locked && lockedRatio === null) {
                    const c = getCrop();
                    const w = (c.x2 - c.x1) * imgW;
                    const h = (c.y2 - c.y1) * imgH;
                    lockedRatio = h > 1e-6 ? w / h : 1;
                }
                if (!locked) lockedRatio = null;

                dragState = { type: "corner", corner, startCrop: getCrop() };
                e.preventDefault();
                e.stopPropagation();
                document.addEventListener("mousemove", onDragMove, true);
                document.addEventListener("mouseup", onDragEnd, true);
                return;
            }

            if (hitInterior(pos.x, pos.y)) {
                const n = toNorm(pos.x, pos.y);
                const c = getCrop();
                dragState = {
                    type: "move",
                    offX: n.x - c.x1,
                    offY: n.y - c.y1,
                    w: c.x2 - c.x1,
                    h: c.y2 - c.y1,
                };
                e.preventDefault();
                e.stopPropagation();
                document.addEventListener("mousemove", onDragMove, true);
                document.addEventListener("mouseup", onDragEnd, true);
            }
        }

        function onDragMove(e) {
            if (!dragState) return;
            e.preventDefault();
            const pos = mousePos(e);
            const n = toNorm(pos.x, pos.y);

            if (dragState.type === "corner") {
                applyCornerDrag(n);
            } else {
                applyMoveDrag(n);
            }
            draw();
        }

        function applyCornerDrag(n) {
            let { x1, y1, x2, y2 } = getCrop();
            const c = dragState.corner;

            // The diagonal opposite corner stays fixed
            let fixX, fixY;
            if (c === "tl")      { fixX = x2; fixY = y2; }
            else if (c === "tr") { fixX = x1; fixY = y2; }
            else if (c === "bl") { fixX = x2; fixY = y1; }
            else                 { fixX = x1; fixY = y1; }

            if (lockedRatio !== null) {
                // Determine dominant axis from total drag distance
                const sc = dragState.startCrop;
                const startNX = (c === "tl" || c === "bl") ? sc.x1 : sc.x2;
                const startNY = (c === "tl" || c === "tr") ? sc.y1 : sc.y2;
                const dx = Math.abs(n.x - startNX);
                const dy = Math.abs(n.y - startNY);

                // lockedRatio is in pixel space (pxW/pxH), convert to
                // normalized space for the constraint math
                const normRatio = lockedRatio * (imgH / Math.max(imgW, 1));

                let newNW = Math.abs(n.x - fixX);
                let newNH = Math.abs(n.y - fixY);

                if (dx >= dy) { newNH = newNW / normRatio; }
                else          { newNW = newNH * normRatio; }

                const signX = n.x >= fixX ? 1 : -1;
                const signY = n.y >= fixY ? 1 : -1;
                const dragX = fixX + signX * newNW;
                const dragY = fixY + signY * newNH;

                x1 = Math.min(fixX, dragX);
                y1 = Math.min(fixY, dragY);
                x2 = Math.max(fixX, dragX);
                y2 = Math.max(fixY, dragY);
            } else {
                // Free drag — update the two coords this corner controls
                if (c === "tl")      { x1 = n.x; y1 = n.y; }
                else if (c === "tr") { x2 = n.x; y1 = n.y; }
                else if (c === "bl") { x1 = n.x; y2 = n.y; }
                else                 { x2 = n.x; y2 = n.y; }

                if (x1 > x2) { const t = x1; x1 = x2; x2 = t; }
                if (y1 > y2) { const t = y1; y1 = y2; y2 = t; }
            }

            setCrop(x1, y1, x2, y2);
        }

        function applyMoveDrag(n) {
            let nx1 = n.x - dragState.offX;
            let ny1 = n.y - dragState.offY;
            nx1 = Math.max(0, Math.min(nx1, 1 - dragState.w));
            ny1 = Math.max(0, Math.min(ny1, 1 - dragState.h));
            setCrop(nx1, ny1, nx1 + dragState.w, ny1 + dragState.h);
        }

        function onDragEnd() {
            dragState = null;
            document.removeEventListener("mousemove", onDragMove, true);
            document.removeEventListener("mouseup", onDragEnd, true);
        }

        // Cursor feedback when not dragging
        cvs.addEventListener("mousemove", (e) => {
            if (dragState) return;
            if (!previewImg) { cvs.style.cursor = "default"; return; }
            const pos = mousePos(e);
            const corner = hitCorner(pos.x, pos.y);
            if (corner) cvs.style.cursor = cursorFor(corner);
            else if (hitInterior(pos.x, pos.y)) cvs.style.cursor = "move";
            else cvs.style.cursor = "default";
        });

        cvs.addEventListener("mousedown", onMouseDown);

        // ----------------------------------------------------------------
        //  Image loading
        // ----------------------------------------------------------------

        function loadImage(url) {
            const img = new window.Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
                const oldW = imgW, oldH = imgH;
                previewImg = img;
                imgW = img.naturalWidth;
                imgH = img.naturalHeight;

                // Adapt crop rectangle to preserve its shape and position
                adaptCropToNewDimensions(oldW, oldH, imgW, imgH);

                draw();
            };
            img.onerror = () =>
                console.warn("[CCN CroppedImage] Failed to load preview:", url);
            img.src = url;
        }

        // Exposed so the onExecuted hook can reach it
        node._ccnCrop = { loadImage };

        // ----------------------------------------------------------------
        //  DOM construction
        // ----------------------------------------------------------------

        const container = document.createElement("div");
        container.style.cssText = "width:100%; display:flex; flex-direction:column;";

        cvs.style.cssText =
            "width:100%; background:#1a1a1a; border-radius:4px;";
        cvs.height = CANVAS_HEIGHT;
        container.appendChild(cvs);

        // Button row
        const btnRow = document.createElement("div");
        btnRow.style.cssText = "display:flex; gap:4px; padding:4px 0;";

        function makeBtn(label) {
            const b = document.createElement("button");
            b.textContent = label;
            b.style.cssText =
                "flex:1; padding:4px 8px; border:1px solid #555; " +
                "background:#2a2a2a; color:#ccc; border-radius:3px; " +
                "cursor:pointer; font-size:11px;";
            b.addEventListener("mouseenter", () => (b.style.background = "#3a3a3a"));
            b.addEventListener("mouseleave", () => (b.style.background = "#2a2a2a"));
            return b;
        }

        const btnPreview = makeBtn("Load Preview");
        const btnUpload = makeBtn("Upload Image");
        const btnReset = makeBtn("Reset Crop");
        btnRow.appendChild(btnPreview);
        btnRow.appendChild(btnUpload);
        btnRow.appendChild(btnReset);
        container.appendChild(btnRow);

        // Hidden file input for the upload button
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "image/png,image/jpeg,image/webp,image/bmp,image/tiff";
        fileInput.style.display = "none";
        container.appendChild(fileInput);

        // ----------------------------------------------------------------
        //  Button handlers
        // ----------------------------------------------------------------

        // Find a preview image URL from an upstream node, checking both
        // the standard .imgs array and any <img> elements inside DOM widgets
        // (custom nodes like VideoScrubber use the latter).
        function findUpstreamPreviewUrl(upstreamNode) {
            if (!upstreamNode) return null;

            // Standard ComfyUI pattern — node.imgs[]
            if (upstreamNode.imgs?.[0]?.src) {
                return upstreamNode.imgs[0].src;
            }

            // DOM widget fallback — search for <img> elements with a loaded src
            for (const w of upstreamNode.widgets || []) {
                const el = w.element || w.inputEl;
                if (!el) continue;
                // Direct img widget
                if (el.tagName === "IMG" && el.src) return el.src;
                // Container with img children
                const img = el.querySelector?.("img");
                if (img?.src && img.naturalWidth > 0) return img.src;
            }

            return null;
        }

        btnPreview.addEventListener("click", () => {
            // Try upstream cached output
            const input = node.inputs?.find((i) => i.name === "image");
            if (input?.link != null) {
                const link = app.graph.links[input.link];
                if (link) {
                    const url = findUpstreamPreviewUrl(
                        app.graph.getNodeById(link.origin_id)
                    );
                    if (url) {
                        loadImage(url);
                        return;
                    }
                }
            }
            // Fallback: loaded_image selection
            const lw = node.widgets.find((w) => w.name === "loaded_image");
            if (lw?.value && lw.value !== "none") {
                loadImage(
                    api.apiURL(
                        `/view?filename=${encodeURIComponent(lw.value)}&type=input`
                    )
                );
                return;
            }
            console.log(
                "[CCN CroppedImage] No preview source. " +
                "Queue the workflow once, upload an image, or select one from the dropdown."
            );
        });

        btnUpload.addEventListener("click", () => fileInput.click());

        fileInput.addEventListener("change", async () => {
            const file = fileInput.files?.[0];
            if (!file) return;

            // Upload to ComfyUI's input directory via the standard API
            const formData = new FormData();
            formData.append("image", file, file.name);
            formData.append("overwrite", "true");

            try {
                const resp = await api.fetchApi("/upload/image", {
                    method: "POST",
                    body: formData,
                });
                if (!resp.ok) {
                    console.error("[CCN CroppedImage] Upload failed:", resp.statusText);
                    return;
                }
                const data = await resp.json();
                const uploadedName = data.name;

                // Update the loaded_image combo widget to reflect the upload
                const lw = node.widgets.find((w) => w.name === "loaded_image");
                if (lw) {
                    // Add to options list if not already present
                    if (lw.options?.values && !lw.options.values.includes(uploadedName)) {
                        lw.options.values.push(uploadedName);
                    }
                    lw.value = uploadedName;
                }

                // Load as preview
                loadImage(
                    api.apiURL(
                        `/view?filename=${encodeURIComponent(uploadedName)}&type=input`
                    )
                );
            } catch (err) {
                console.error("[CCN CroppedImage] Upload error:", err);
            }

            // Reset so the same file can be re-selected
            fileInput.value = "";
        });

        btnReset.addEventListener("click", () => {
            setCrop(0, 0, 1, 1);
            lockedRatio = null;
            draw();
        });

        // Auto-load preview when the loaded_image dropdown changes
        const loadedW = node.widgets.find((w) => w.name === "loaded_image");
        if (loadedW) {
            const origCb = loadedW.callback;
            loadedW.callback = function (...args) {
                origCb?.apply(this, args);
                if (loadedW.value && loadedW.value !== "none") {
                    loadImage(
                        api.apiURL(
                            `/view?filename=${encodeURIComponent(loadedW.value)}&type=input`
                        )
                    );
                }
            };
        }

        // ----------------------------------------------------------------
        //  Register DOM widget
        // ----------------------------------------------------------------

        const domWidget = node.addDOMWidget(
            "ccn_crop_canvas", "custom", container,
            { getValue: () => "", setValue: () => {} },
        );
        domWidget.computeSize = () => [node.size[0], CANVAS_HEIGHT + 36];

        // Keep canvas resolution in sync with layout width
        function syncSize() {
            const w = container.clientWidth || node.size?.[0] || 300;
            if (w > 0 && cvs.width !== w) {
                cvs.width = w;
                cvs.height = CANVAS_HEIGHT;
                draw();
            }
        }

        const observer = new ResizeObserver(syncSize);
        observer.observe(container);

        const origResize = node.onResize;
        node.onResize = function () {
            origResize?.apply(this, arguments);
            syncSize();
        };

        const origRemoved = node.onRemoved;
        node.onRemoved = function () {
            observer.disconnect();
            origRemoved?.apply(this, arguments);
        };

        requestAnimationFrame(() => {
            syncSize();
            draw();
        });
    },
});
