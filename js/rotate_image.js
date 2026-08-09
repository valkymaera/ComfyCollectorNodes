import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/*
 * Interactive rotation overlay for CCN_RotateImage.
 *
 * Renders the source image on a canvas rotated live by the angle widget.
 * Dragging on the image rotates around the pivot (Shift snaps to 15°);
 * the pivot is a draggable crosshair stored as normalized 0-1 values in
 * hidden widgets that the Python node reads at execution time.
 *
 * The canvas always shows the OUTPUT frame: with expand off that is the
 * input frame (uncovered areas show fill_color), with expand on it is the
 * rotated bounding box fitted to the canvas (the pivot is hidden — it has
 * no effect on expand output). The draw transform mirrors the Python
 * sampling matrix exactly, so the canvas is a true preview of the result.
 *
 * The backdrop changes only on explicit user action (Load Preview, Upload
 * Image, loaded_image selection) or on workflow load — never on execution,
 * so queueing a job never disturbs the rotation you've set up. Execution
 * only RECORDS the processed source so Load Preview can pull up a wired
 * backdrop on demand.
 *
 * Image sources:
 *   - "Load Preview" — wired upstream's cached output, else the source the
 *      node last processed (saved server-side by the Python node), else the
 *      loaded_image selection
 *   - "Upload Image" — browser file picker, copies to input dir
 *   - loaded_image combo — loads when the selection changes
 */

const PIVOT_HIT_RADIUS = 12;
const CANVAS_HEIGHT = 350;
const MIN_BOX = 200;

app.registerExtension({
    name: "CCN.RotateImage",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "CCN_RotateImage") return;

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origOnExecuted?.apply(this, arguments);
            // Record the source the node just processed WITHOUT changing the
            // displayed backdrop — queueing a job must never swap the preview.
            // Load Preview pulls this up on demand (see loadBestPreview). The
            // preview key is custom (not "images") so it never hits the feed.
            const preview = message?.ccn_rotate_preview?.[0];
            if (preview) {
                this._ccnLastPreview = {
                    url: api.apiURL(
                        `/view?filename=${encodeURIComponent(preview.filename)}` +
                        `&type=${preview.type}&subfolder=${preview.subfolder || ""}`
                    ),
                    source: message?.ccn_rotate_source?.[0] ?? "wired source",
                };
            }
        };
    },

    async nodeCreated(node) {
        if (node.comfyClass !== "CCN_RotateImage") return;

        // ----------------------------------------------------------------
        //  Hidden pivot widgets — still serialize, but invisible in the UI
        // ----------------------------------------------------------------

        const HIDDEN = ["pivot_x", "pivot_y"];
        const cw = {};
        for (const w of node.widgets) {
            if (HIDDEN.includes(w.name)) {
                cw[w.name] = w;
                w.type = "hidden";
                w.computeSize = () => [0, -4];
            }
        }

        function getPivot() {
            return {
                x: cw.pivot_x?.value ?? 0.5,
                y: cw.pivot_y?.value ?? 0.5,
            };
        }

        function setPivot(x, y) {
            const cl = (v) => Math.max(0, Math.min(1, v));
            if (cw.pivot_x) cw.pivot_x.value = cl(x);
            if (cw.pivot_y) cw.pivot_y.value = cl(y);
        }

        // ----------------------------------------------------------------
        //  Visible widget accessors
        // ----------------------------------------------------------------

        const angleW = node.widgets.find((w) => w.name === "angle");
        const expandW = node.widgets.find((w) => w.name === "expand");
        const fillW = node.widgets.find((w) => w.name === "fill_color");

        function angle() {
            return angleW?.value ?? 0;
        }

        // Round to the widget's 0.1 step so the displayed number stays clean
        // while dragging; repaint the widget row so the number tracks live.
        function setAngle(a) {
            if (angleW) angleW.value = Math.round(a * 10) / 10;
            node.setDirtyCanvas(true, false);
        }

        function wrap180(a) {
            return ((a + 180) % 360 + 360) % 360 - 180;
        }

        function isExpand() {
            return expandW?.value ?? false;
        }

        // Drawing-only fallback to black on malformed hex; Python raises at
        // queue time with the offending string.
        function safeFill() {
            const v = (fillW?.value ?? "").trim();
            if (/^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.test(v)) {
                return v.startsWith("#") ? v : "#" + v;
            }
            return "#000000";
        }

        // ----------------------------------------------------------------
        //  State
        // ----------------------------------------------------------------

        let previewImg = null;
        let imgW = 0;
        let imgH = 0;
        let dragState = null;
        let sizeLabel = null;    // dims/angle readout, assigned during DOM construction
        let sourceLabel = null;  // provenance line, assigned during DOM construction

        const cvs = document.createElement("canvas");

        // ----------------------------------------------------------------
        //  Coordinate conversion — all normalized coords live in the
        //  OUTPUT frame (input dims when fixed, rotated bbox when expanded)
        // ----------------------------------------------------------------

        // Output dimensions in pixels; the bbox uses the same round formula
        // as Python so the size label always matches the queued result.
        function outDims() {
            if (!imgW || !imgH) return { w: 0, h: 0 };
            if (!isExpand()) return { w: imgW, h: imgH };
            const t = angle() * Math.PI / 180;
            const c = Math.abs(Math.cos(t));
            const s = Math.abs(Math.sin(t));
            return {
                w: Math.max(1, Math.round(imgW * c + imgH * s)),
                h: Math.max(1, Math.round(imgW * s + imgH * c)),
            };
        }

        // Rectangle the output frame occupies inside the canvas, preserving aspect
        function displayRect() {
            const W = cvs.width, H = cvs.height;
            const d = outDims();
            if (!d.w || !d.h) return { x: 0, y: 0, w: W, h: H };
            const iA = d.w / d.h, cA = W / H;
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

        function pivotCanvasPos() {
            const p = getPivot();
            const r = displayRect();
            return { x: r.x + p.x * r.w, y: r.y + p.y * r.h };
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
                    "No preview — Load Preview, Upload Image, or queue once",
                    W / 2, H / 2,
                );
                updateSizeLabel();
                return;
            }

            const r = displayRect();
            const rad = angle() * Math.PI / 180;

            // Fill first, then rotated image clipped to the frame — uncovered
            // areas show the fill, matching Python's composite.
            ctx.save();
            ctx.beginPath();
            ctx.rect(r.x, r.y, r.w, r.h);
            ctx.clip();
            ctx.fillStyle = safeFill();
            ctx.fillRect(r.x, r.y, r.w, r.h);
            if (isExpand()) {
                // Rotate about the output center; ctx.rotate is clockwise in
                // y-down space, same convention as the Python matrix.
                const s = r.w / outDims().w;
                ctx.translate(r.x + r.w / 2, r.y + r.h / 2);
                ctx.rotate(rad);
                ctx.drawImage(
                    previewImg,
                    -imgW * s / 2, -imgH * s / 2, imgW * s, imgH * s,
                );
            } else {
                const p = getPivot();
                ctx.translate(r.x + p.x * r.w, r.y + p.y * r.h);
                ctx.rotate(rad);
                ctx.drawImage(previewImg, -p.x * r.w, -p.y * r.h, r.w, r.h);
            }
            ctx.restore();

            if (!isExpand()) drawPivotMarker(ctx);
            updateSizeLabel();
        }

        // Crosshair + circle, dark stroke under white so it reads on any backdrop.
        function drawPivotMarker(ctx) {
            const p = pivotCanvasPos();
            const arm = 9;
            for (const [style, width] of [["rgba(0,0,0,0.7)", 3], ["#fff", 1.5]]) {
                ctx.strokeStyle = style;
                ctx.lineWidth = width;
                ctx.beginPath();
                ctx.moveTo(p.x - arm, p.y);
                ctx.lineTo(p.x + arm, p.y);
                ctx.moveTo(p.x, p.y - arm);
                ctx.lineTo(p.x, p.y + arm);
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2);
                ctx.stroke();
            }
        }

        // Output dims + angle/mode readout in the DOM label below the buttons.
        function updateSizeLabel() {
            if (!sizeLabel) return;
            if (!previewImg || !imgW || !imgH) {
                sizeLabel.textContent = "—";
                return;
            }
            const d = outDims();
            const mode = isExpand() ? "expand" : "fixed";
            sizeLabel.textContent =
                `${d.w} × ${d.h} px\n${angle().toFixed(1)}° · ${mode}`;
        }

        // ----------------------------------------------------------------
        //  Hit detection
        // ----------------------------------------------------------------

        function hitPivot(cx, cy) {
            if (isExpand()) return false;  // pivot has no effect when expanded
            const p = pivotCanvasPos();
            return Math.hypot(cx - p.x, cy - p.y) <= PIVOT_HIT_RADIUS;
        }

        function hitFrame(cx, cy) {
            const r = displayRect();
            return cx >= r.x && cx <= r.x + r.w &&
                   cy >= r.y && cy <= r.y + r.h;
        }

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

        // Fixed mode rotates around the pivot; expand mode around the frame
        // center, which is always the canvas center regardless of bbox size.
        function rotationCenter() {
            if (isExpand()) {
                return { x: cvs.width / 2, y: cvs.height / 2 };
            }
            return pivotCanvasPos();
        }

        function onMouseDown(e) {
            if (!previewImg) return;
            const pos = mousePos(e);

            if (hitPivot(pos.x, pos.y)) {
                dragState = { type: "pivot" };
            } else if (hitFrame(pos.x, pos.y)) {
                const c = rotationCenter();
                dragState = {
                    type: "rotate",
                    center: c,
                    startAngle: angle(),
                    // atan2(dy, dx) grows clockwise in y-down space, matching
                    // the angle convention directly.
                    prevPointer: Math.atan2(pos.y - c.y, pos.x - c.x),
                    accum: 0,
                };
                cvs.style.cursor = "grabbing";
            } else {
                return;  // fall through so litegraph can drag the node
            }

            e.preventDefault();
            e.stopPropagation();
            document.addEventListener("mousemove", onDragMove, true);
            document.addEventListener("mouseup", onDragEnd, true);
        }

        function onDragMove(e) {
            if (!dragState) return;
            e.preventDefault();
            const pos = mousePos(e);

            if (dragState.type === "pivot") {
                const n = toNorm(pos.x, pos.y);
                setPivot(n.x, n.y);
            } else {
                // Accumulate unwrapped per-move deltas so continuous rotation
                // past ±180° keeps going; snap only the applied value so
                // releasing Shift resumes from the raw accumulator.
                const c = dragState.center;
                const cur = Math.atan2(pos.y - c.y, pos.x - c.x);
                let d = cur - dragState.prevPointer;
                if (d > Math.PI) d -= 2 * Math.PI;
                else if (d < -Math.PI) d += 2 * Math.PI;
                dragState.prevPointer = cur;
                dragState.accum += d * 180 / Math.PI;
                let a = dragState.startAngle + dragState.accum;
                if (e.shiftKey) a = Math.round(a / 15) * 15;
                setAngle(wrap180(a));
            }
            draw();
        }

        function onDragEnd() {
            dragState = null;
            document.removeEventListener("mousemove", onDragMove, true);
            document.removeEventListener("mouseup", onDragEnd, true);
            cvs.style.cursor = "default";
        }

        // Cursor feedback when not dragging
        cvs.addEventListener("mousemove", (e) => {
            if (dragState) return;
            if (!previewImg) { cvs.style.cursor = "default"; return; }
            const pos = mousePos(e);
            if (hitPivot(pos.x, pos.y)) cvs.style.cursor = "move";
            else if (hitFrame(pos.x, pos.y)) cvs.style.cursor = "grab";
            else cvs.style.cursor = "default";
        });

        cvs.addEventListener("mousedown", onMouseDown);

        // ----------------------------------------------------------------
        //  Image loading
        // ----------------------------------------------------------------

        // No dimension-adaptation logic needed here (unlike the crop node):
        // the angle and normalized pivot are dimension-independent.
        function loadImage(url, source) {
            const img = new window.Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
                previewImg = img;
                imgW = img.naturalWidth;
                imgH = img.naturalHeight;
                if (source !== undefined) setSource(source);
                draw();
            };
            img.onerror = () =>
                console.warn("[CCN RotateImage] Failed to load preview:", url);
            img.src = url;
        }

        // Provenance line under the canvas: which source this preview came from.
        function setSource(text) {
            if (sourceLabel) sourceLabel.textContent = text || "";
        }

        // ----------------------------------------------------------------
        //  DOM construction
        // ----------------------------------------------------------------

        const container = document.createElement("div");
        container.style.cssText =
            "width:100%;height:100%;box-sizing:border-box;" +
            "display:flex;flex-direction:column;";

        // Canvas is the flex-fill child; its drawing buffer is synced to its
        // displayed size in syncSize (one-way: layout -> buffer).
        cvs.style.cssText =
            "flex:1 1 auto;min-height:0;width:100%;display:block;" +
            "background:#1a1a1a;border-radius:4px;";
        cvs.height = CANVAS_HEIGHT;
        container.appendChild(cvs);

        sourceLabel = document.createElement("div");
        sourceLabel.style.cssText =
            "flex:0 0 auto;padding:2px 4px 0;font:10px monospace;" +
            "color:#888;text-align:center;";
        sourceLabel.textContent = "";
        container.appendChild(sourceLabel);

        const btnRow = document.createElement("div");
        btnRow.style.cssText =
            "flex:0 0 auto;display:flex;gap:4px;padding:4px 0;";

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
        const btnReset = makeBtn("Reset Rotation");
        btnRow.appendChild(btnPreview);
        btnRow.appendChild(btnUpload);
        btnRow.appendChild(btnReset);
        container.appendChild(btnRow);

        sizeLabel = document.createElement("div");
        sizeLabel.style.cssText =
            "flex:0 0 auto;white-space:pre-line;padding:3px 4px 1px;" +
            "font:11px monospace;color:#ccc;text-align:center;";
        sizeLabel.textContent = "—";
        container.appendChild(sizeLabel);

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
        function findUpstreamPreviewUrl(upstreamNode, originSlot) {
            if (!upstreamNode) return null;

            // 1) Standard ComfyUI pattern — node.imgs[]
            if (upstreamNode.imgs?.[0]?.src) {
                return upstreamNode.imgs[0].src;
            }

            // 2) CCN convention: a node exposing a live preview of its own
            //    output. Slot-aware hook first — nodes whose outputs carry
            //    different pixels resolve the preview for the exact slot the
            //    wire leaves from. Plain ccn_image is the fallback.
            const forSlot = upstreamNode.ccn_imageForSlot?.(originSlot);
            if (forSlot) return forSlot;
            const ccn = upstreamNode.ccn_image;
            if (ccn) return typeof ccn === "string" ? ccn : (ccn.src || null);

            // 3) DOM widget fallback — search for <img> elements with a loaded src
            for (const w of upstreamNode.widgets || []) {
                const el = w.element || w.inputEl;
                if (!el) continue;
                if (el.tagName === "IMG" && el.src) return el.src;
                const img = el.querySelector?.("img");
                if (img?.src && img.naturalWidth > 0) return img.src;
            }

            return null;
        }

        // The active node feeding the image input plus the output slot the
        // wire leaves from ({ up, slot }), or null. A link to a muted (mode 2)
        // or bypassed (mode 4) upstream counts as NOT wired: a disabled node
        // yields no image at run time, so the node falls back to loaded_image
        // — and Load Preview should reflect that, not the dead source's stale
        // preview.
        function activeUpstreamNode() {
            const input = node.inputs?.find((i) => i.name === "image");
            if (input?.link == null) return null;
            const link = app.graph.links[input.link];
            if (!link) return null;
            const up = app.graph.getNodeById(link.origin_id);
            if (!up) return null;
            if (up.mode === 2 || up.mode === 4) return null;  // muted / bypassed
            return { up, slot: link.origin_slot };
        }

        function isWired() {
            return activeUpstreamNode() != null;
        }

        function wiredPreviewUrl() {
            const wired = activeUpstreamNode();
            return wired ? findUpstreamPreviewUrl(wired.up, wired.slot) : null;
        }

        // Load the backdrop for whatever drives the run. Only ever called from
        // explicit actions (Load Preview) and workflow load — never on execute —
        // so the preview stays put between user actions.
        //   Wired:     upstream's live cached preview, else the exact source the
        //              node last processed (saved server-side), else nothing.
        //   Not wired: the current loaded_image selection.
        function loadBestPreview() {
            if (isWired()) {
                const wired = wiredPreviewUrl();
                if (wired) { loadImage(wired, "wired source"); return true; }
                if (node._ccnLastPreview) {
                    loadImage(node._ccnLastPreview.url, node._ccnLastPreview.source);
                    return true;
                }
                return false;
            }
            const lw = node.widgets.find((w) => w.name === "loaded_image");
            if (lw?.value && lw.value !== "none") {
                loadImage(
                    api.apiURL(
                        `/view?filename=${encodeURIComponent(lw.value)}&type=input`
                    ),
                    `loaded: ${lw.value}`,
                );
                return true;
            }
            return false;
        }

        btnPreview.addEventListener("click", () => {
            if (!loadBestPreview()) {
                console.log(
                    "[CCN RotateImage] No preview source. Queue once, wire an " +
                    "image, upload, or select one from the dropdown."
                );
            }
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
                    console.error("[CCN RotateImage] Upload failed:", resp.statusText);
                    return;
                }
                const data = await resp.json();
                const uploadedName = data.name;

                // Update the loaded_image combo widget to reflect the upload
                const lw = node.widgets.find((w) => w.name === "loaded_image");
                if (lw) {
                    if (lw.options?.values && !lw.options.values.includes(uploadedName)) {
                        lw.options.values.push(uploadedName);
                    }
                    lw.value = uploadedName;
                }

                loadImage(
                    api.apiURL(
                        `/view?filename=${encodeURIComponent(uploadedName)}&type=input`
                    ),
                    `uploaded: ${uploadedName}`,
                );
            } catch (err) {
                console.error("[CCN RotateImage] Upload error:", err);
            }

            // Reset so the same file can be re-selected
            fileInput.value = "";
        });

        // Reset restores angle and pivot only; expand/fill are left as set.
        btnReset.addEventListener("click", () => {
            setAngle(0);
            setPivot(0.5, 0.5);
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
                        ),
                        `loaded: ${loadedW.value}`,
                    );
                }
            };
        }

        // Redraw when a visible widget that affects the preview changes.
        for (const name of ["angle", "expand", "fill_color"]) {
            const w = node.widgets.find((x) => x.name === name);
            if (!w) continue;
            const origCb = w.callback;
            w.callback = function (...args) {
                origCb?.apply(this, args);
                draw();
            };
        }

        // ----------------------------------------------------------------
        //  Register DOM widget
        // ----------------------------------------------------------------

        // The canvas reports only a MINIMUM height to ComfyUI via getMinHeight.
        // ComfyUI derives the node's min-size from that, while the element fills
        // the node's available height above it — so the canvas scales up when the
        // node grows yet the node still resizes back down to the minimum.
        node.addDOMWidget(
            "ccn_rotate_canvas", "custom", container,
            { getValue: () => "", setValue: () => {}, getMinHeight: () => MIN_BOX },
        );

        // Keep the drawing buffer matched to the canvas's displayed size. One-way
        // only (layout -> buffer); buffer size never sizes the widget or node, so
        // it can't drive resize creep.
        function syncSize() {
            const w = Math.round(cvs.clientWidth);
            const h = Math.round(cvs.clientHeight);
            if (w > 0 && h > 0 && (cvs.width !== w || cvs.height !== h)) {
                cvs.width = w;
                cvs.height = h;
                draw();
            }
        }

        const observer = new ResizeObserver(syncSize);
        observer.observe(cvs);

        const origResize = node.onResize;
        node.onResize = function () {
            origResize?.apply(this, arguments);
            syncSize();
        };

        // Restore the backdrop on workflow load instead of going blank, using
        // the same wired > loaded order as the run.
        const origConfigure = node.onConfigure;
        node.onConfigure = function () {
            origConfigure?.apply(this, arguments);
            requestAnimationFrame(() => {
                syncSize();
                loadBestPreview();
            });
        };

        const origRemoved = node.onRemoved;
        node.onRemoved = function () {
            observer.disconnect();
            origRemoved?.apply(this, arguments);
        };

        // Expose a live preview of THIS node's rotated output so downstream
        // CCN nodes can show it without a queue. Computed on read via an
        // offscreen canvas at full output resolution, so interaction pays
        // nothing; null until a backdrop is loaded. try/catch guards a
        // tainted-canvas read so the consumer's search just falls through.
        function rotateDataUrl() {
            if (!previewImg || !imgW || !imgH) return null;
            try {
                const d = outDims();
                const off = document.createElement("canvas");
                off.width = d.w;
                off.height = d.h;
                const octx = off.getContext("2d");
                octx.fillStyle = safeFill();
                octx.fillRect(0, 0, d.w, d.h);
                const rad = angle() * Math.PI / 180;
                if (isExpand()) {
                    octx.translate(d.w / 2, d.h / 2);
                    octx.rotate(rad);
                    octx.drawImage(previewImg, -imgW / 2, -imgH / 2);
                } else {
                    const p = getPivot();
                    octx.translate(p.x * imgW, p.y * imgH);
                    octx.rotate(rad);
                    octx.drawImage(previewImg, -p.x * imgW, -p.y * imgH);
                }
                return off.toDataURL("image/png");
            } catch (err) {
                console.warn("[CCN RotateImage] rotate preview failed:", err);
                return null;
            }
        }

        // Slot-aware preview: the rotated result for "image", the unrotated
        // backdrop for "source_image". Matched by output name, not index,
        // so slot reordering can't silently break it.
        node.ccn_imageForSlot = (slot) => {
            const name = node.outputs?.[slot]?.name;
            if (name === "image") return rotateDataUrl();
            if (name === "source_image") return previewImg?.src ?? null;
            return null;
        };

        // Single-preview fallback for consumers without slot awareness.
        Object.defineProperty(node, "ccn_image", {
            configurable: true,
            get: rotateDataUrl,
        });

        requestAnimationFrame(() => {
            syncSize();
            loadBestPreview();   // show a wired/loaded source if one is ready
            draw();
        });
    },
});
