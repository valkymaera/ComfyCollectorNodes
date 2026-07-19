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
 * The backdrop changes only on explicit user action (Load Preview, Upload
 * Image, loaded_image selection) or on workflow load — never on execution,
 * so queueing a job never disturbs the crop you've set up. Execution only
 * RECORDS the processed source so Load Preview can pull up a wired backdrop
 * on demand.
 *
 * Image sources:
 *   - "Load Preview" — wired upstream's cached output, else the source the
 *      node last processed (saved server-side by the Python node), else the
 *      loaded_image selection
 *   - "Upload Image" — browser file picker, copies to input dir
 *   - loaded_image combo — loads when the selection changes
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
            // Record the source the node just processed WITHOUT changing the
            // displayed backdrop — queueing a job must never swap the preview.
            // Load Preview pulls this up on demand (see loadBestPreview), so a
            // wired backdrop stays reachable for any upstream node type. The
            // preview key is custom (not "images") so it never hits the feed.
            const preview = message?.ccn_crop_preview?.[0];
            if (preview) {
                this._ccnLastPreview = {
                    url: api.apiURL(
                        `/view?filename=${encodeURIComponent(preview.filename)}` +
                        `&type=${preview.type}&subfolder=${preview.subfolder || ""}`
                    ),
                    source: message?.ccn_crop_source?.[0] ?? "wired source",
                };
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
        let sizeLabel = null;    // px/ratio readout, assigned during DOM construction
        let sourceLabel = null;  // provenance line, assigned during DOM construction

        // ----------------------------------------------------------------
        //  Crop stabilization across image dimension changes.
        //
        //  lock_ratio OFF: keep the normalized box — the rectangle stays at the
        //  same relative position and size on the new frame. Same-aspect swaps
        //  (thumbnail -> full-res) are then exact no-ops.
        //
        //  lock_ratio ON: the ratio is the intent, so preserve it. Rebuild the
        //  crop at the locked ratio, sized to the same fraction of the image's
        //  longest edge, fitted inside the new frame, recentred.
        // ----------------------------------------------------------------

        function adaptCropToNewDimensions(oldW, oldH, newW, newH) {
            // First load or identical dimensions — keep the crop untouched.
            if (!oldW || !oldH || (oldW === newW && oldH === newH)) return;

            const lockW = node.widgets.find((w) => w.name === "lock_ratio");
            const locked = lockW?.value ?? false;
            const c = getCrop();

            // Unlocked: preserve the normalized rectangle as-is.
            if (!locked) return;

            // Locked: hold the ratio captured at lock-drag start if there is
            // one, else the crop's current pixel ratio on the old image.
            const oldPxW = (c.x2 - c.x1) * oldW;
            const oldPxH = (c.y2 - c.y1) * oldH;
            const ratio = (lockedRatio !== null)
                ? lockedRatio
                : (oldPxH > 1e-6 ? oldPxW / oldPxH : 1);

            // Keep the crop's size relative to the longest edge, then rebuild
            // at the locked ratio with the longer crop side driving.
            const oldLong = Math.max(oldW, oldH);
            const newLong = Math.max(newW, newH);
            const sizeFrac = Math.max(oldPxW, oldPxH) / oldLong;
            const targetLong = sizeFrac * newLong;

            let newPxW, newPxH;
            if (ratio >= 1) { newPxW = targetLong; newPxH = newPxW / ratio; }
            else            { newPxH = targetLong; newPxW = newPxH * ratio; }

            // Fit inside the new frame without distorting the ratio.
            if (newPxW > newW) { newPxW = newW; newPxH = newPxW / ratio; }
            if (newPxH > newH) { newPxH = newH; newPxW = newPxH * ratio; }

            // Preserve the normalized centre; clamp so the box stays in bounds.
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
                updateSizeLabel();
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

            // Crop size is shown in a fixed DOM label below the buttons, not
            // on the canvas, so it stays visible when the crop nears any edge.
            updateSizeLabel();
        }

        // Write the current crop's pixel dimensions into the DOM readout.
        // Reads live crop coords, so it tracks dragging in real time.
        function updateSizeLabel() {
            if (!sizeLabel) return;
            if (!previewImg || !imgW || !imgH) {
                sizeLabel.textContent = "\u2014";
                return;
            }
            const c = getCrop();
            const pw = Math.round((c.x2 - c.x1) * imgW);
            const ph = Math.round((c.y2 - c.y1) * imgH);
            const ratio = ph > 0 ? (pw / ph) : 0;
            // px on the first line, bare W/H ratio (2 dp) on the second.
            sizeLabel.textContent = `${pw} \u00d7 ${ph} px\nratio ${ratio.toFixed(2)}`;
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

        function loadImage(url, source) {
            const img = new window.Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
                const oldW = imgW, oldH = imgH;
                previewImg = img;
                imgW = img.naturalWidth;
                imgH = img.naturalHeight;

                // Adapt crop rectangle to the new dimensions (lock-aware).
                adaptCropToNewDimensions(oldW, oldH, imgW, imgH);

                if (source !== undefined) setSource(source);
                draw();
            };
            img.onerror = () =>
                console.warn("[CCN CroppedImage] Failed to load preview:", url);
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

        // Provenance line — right under the canvas so it reads as a caption for
        // what's shown (and, after a run, what was processed).
        sourceLabel = document.createElement("div");
        sourceLabel.style.cssText =
            "flex:0 0 auto;padding:2px 4px 0;font:10px monospace;" +
            "color:#888;text-align:center;";
        sourceLabel.textContent = "";
        container.appendChild(sourceLabel);

        // Button row
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
        const btnReset = makeBtn("Reset Crop");
        btnRow.appendChild(btnPreview);
        btnRow.appendChild(btnUpload);
        btnRow.appendChild(btnReset);
        container.appendChild(btnRow);

        // Crop dimensions + ratio readout — fixed below the buttons so it's
        // always visible. pre-line renders the px line and ratio line as two rows.
        sizeLabel = document.createElement("div");
        sizeLabel.style.cssText =
            "flex:0 0 auto;white-space:pre-line;padding:3px 4px 1px;" +
            "font:11px monospace;color:#ccc;text-align:center;";
        sizeLabel.textContent = "\u2014";
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
            //    different pixels (crop vs source_image) resolve the preview
            //    for the exact slot the wire leaves from. Plain ccn_image
            //    (string URL or { src }) is the single-preview fallback.
            const forSlot = upstreamNode.ccn_imageForSlot?.(originSlot);
            if (forSlot) return forSlot;
            const ccn = upstreamNode.ccn_image;
            if (ccn) return typeof ccn === "string" ? ccn : (ccn.src || null);

            // 3) DOM widget fallback — search for <img> elements with a loaded src
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

        // Is the image input wired to an ACTIVE upstream? Mirrors Python: a
        // wired image takes priority over loaded_image only when it will run.
        function isWired() {
            return activeUpstreamNode() != null;
        }

        // Best-effort URL for the active wired upstream's current cached preview.
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
                    "[CCN CroppedImage] No preview source. Queue once, wire an " +
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

                // Load as preview (explicit pick — show what was just brought in).
                loadImage(
                    api.apiURL(
                        `/view?filename=${encodeURIComponent(uploadedName)}&type=input`
                    ),
                    `uploaded: ${uploadedName}`,
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
                        ),
                        `loaded: ${loadedW.value}`,
                    );
                }
            };
        }

        // ----------------------------------------------------------------
        //  Register DOM widget
        // ----------------------------------------------------------------

        // The canvas reports only a MINIMUM height to ComfyUI via getMinHeight.
        // ComfyUI derives the node's min-size from that, while the element fills
        // the node's available height above it — so the canvas scales up when the
        // node grows yet the node still resizes back down to the minimum.
        const MIN_BOX = 200;
        const domWidget = node.addDOMWidget(
            "ccn_crop_canvas", "custom", container,
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

        // Expose live previews of THIS node's outputs so downstream CCN nodes
        // can show them without a queue. The crop is computed on read via an
        // offscreen canvas, so interaction pays nothing; null until a backdrop
        // is loaded. try/catch guards a tainted-canvas read so the consumer's
        // search just falls through.
        function cropDataUrl() {
            if (!previewImg || !imgW || !imgH) return null;
            try {
                const c = getCrop();
                const sx = Math.round(c.x1 * imgW);
                const sy = Math.round(c.y1 * imgH);
                const sw = Math.max(1, Math.round((c.x2 - c.x1) * imgW));
                const sh = Math.max(1, Math.round((c.y2 - c.y1) * imgH));
                const off = document.createElement("canvas");
                off.width = sw;
                off.height = sh;
                const octx = off.getContext("2d");
                octx.drawImage(previewImg, sx, sy, sw, sh, 0, 0, sw, sh);
                return off.toDataURL("image/png");
            } catch (err) {
                console.warn("[CCN CroppedImage] crop preview failed:", err);
                return null;
            }
        }

        // Slot-aware preview: consumers pass the wire's origin slot so
        // source_image resolves to the uncropped backdrop while the crop
        // outputs resolve to the crop. Matched by output name, not index,
        // so slot reordering can't silently break it.
        node.ccn_imageForSlot = (slot) => {
            const name = node.outputs?.[slot]?.name;
            if (name === "source_image") return previewImg?.src ?? null;
            if (name === "image" || name === "raw_image") return cropDataUrl();
            return null;
        };

        // Single-preview fallback for consumers without slot awareness.
        Object.defineProperty(node, "ccn_image", {
            configurable: true,
            get: cropDataUrl,
        });

        requestAnimationFrame(() => {
            syncSize();
            loadBestPreview();   // show a wired/loaded source if one is ready
            draw();
        });
    },
});
