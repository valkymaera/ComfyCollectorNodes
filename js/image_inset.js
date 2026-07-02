import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/*
 * Interactive inset overlay for CCN_ImageInset.
 *
 * Renders the base source on a canvas with up to three draggable
 * destination rectangles (embed1/2/3 -> red/green/blue). Each rectangle's
 * normalized 0-1 coordinates live in hidden widgets the Python node reads
 * at execution. A rectangle is active only when its embed input is wired to
 * an enabled upstream; a disabled (muted/bypassed) or unwired embed hides
 * its rectangle and is not composited, matching what Python resolves.
 *
 * WYSIWYG: each active embed's actual thumbnail is drawn filling its
 * rectangle, fetched the same way the base backdrop is — the live upstream
 * preview when available, else the thumbnail the node saved on its last run.
 * Because the canvas fill matches Python's fill exactly, the canvas is a
 * true preview of the compilation.
 *
 * lock_ratio (default on) constrains corner-dragging so each rectangle keeps
 * its embed's pixel aspect, so filling never distorts. Off lets a rectangle
 * be dragged to any shape (the embed stretches to fill it).
 *
 * The backdrop and embed thumbnails change only on explicit action (Load
 * Preview, Upload Image, a connection change, loaded_image selection) or on
 * workflow load — never on execution, so queueing never disturbs a layout.
 */

const CORNER_HIT_RADIUS = 12;
const HANDLE_SIZE = 8;
const CANVAS_HEIGHT = 350;
const MIN_BOX = 200;

const EMBED_IDS = [1, 2, 3];
const EMBED_COLORS = { 1: "#ff5252", 2: "#4caf50", 3: "#448aff" };
const EMBED_LETTER = { 1: "R", 2: "G", 3: "B" };

// Staggered default centers (normalized) so the three rects never start stacked.
const STAGGER = {
    1: { cx: 0.28, cy: 0.28 },
    2: { cx: 0.50, cy: 0.50 },
    3: { cx: 0.72, cy: 0.72 },
};
// Default rect size as a fraction of the base image's longest edge.
const DEFAULT_SIZE_FRAC = 0.28;

app.registerExtension({
    name: "CCN.ImageInset",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "CCN_ImageInset") return;

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origOnExecuted?.apply(this, arguments);
            // Record what the node processed WITHOUT redrawing — a queue must
            // never swap the canvas under the user. Load Preview pulls these up
            // on demand (see loadBestPreview / refreshEmbeds). Custom UI keys
            // (not "images") so nothing hits the node preview or image feed.
            const base = message?.ccn_inset_preview?.[0];
            if (base) {
                this._ccnLastPreview = {
                    url: api.apiURL(
                        `/view?filename=${encodeURIComponent(base.filename)}` +
                        `&type=${base.type}&subfolder=${base.subfolder || ""}`
                    ),
                    source: message?.ccn_inset_source?.[0] ?? "wired source",
                };
            }
            this._ccnLastEmbedPreview = this._ccnLastEmbedPreview || {};
            for (const id of EMBED_IDS) {
                const e = message?.[`ccn_inset_embed${id}`]?.[0];
                if (e) {
                    this._ccnLastEmbedPreview[id] = api.apiURL(
                        `/view?filename=${encodeURIComponent(e.filename)}` +
                        `&type=${e.type}&subfolder=${e.subfolder || ""}`
                    );
                }
            }
        };
    },

    async nodeCreated(node) {
        if (node.comfyClass !== "CCN_ImageInset") return;

        // ----------------------------------------------------------------
        //  Hidden rect widgets — serialize the layout, invisible in the UI
        // ----------------------------------------------------------------

        const HIDDEN = [];
        for (const id of EMBED_IDS) {
            HIDDEN.push(
                `embed${id}_x1`, `embed${id}_y1`,
                `embed${id}_x2`, `embed${id}_y2`,
            );
        }
        const cw = {};
        for (const w of node.widgets) {
            if (HIDDEN.includes(w.name)) {
                cw[w.name] = w;
                w.type = "hidden";
                w.computeSize = () => [0, -4];
            }
        }

        function getRect(id) {
            return {
                x1: cw[`embed${id}_x1`]?.value ?? 0,
                y1: cw[`embed${id}_y1`]?.value ?? 0,
                x2: cw[`embed${id}_x2`]?.value ?? 1,
                y2: cw[`embed${id}_y2`]?.value ?? 1,
            };
        }

        function setRect(id, x1, y1, x2, y2) {
            const cl = (v) => Math.max(0, Math.min(1, v));
            if (cw[`embed${id}_x1`]) cw[`embed${id}_x1`].value = cl(x1);
            if (cw[`embed${id}_y1`]) cw[`embed${id}_y1`].value = cl(y1);
            if (cw[`embed${id}_x2`]) cw[`embed${id}_x2`].value = cl(x2);
            if (cw[`embed${id}_y2`]) cw[`embed${id}_y2`].value = cl(y2);
        }

        // ----------------------------------------------------------------
        //  State
        // ----------------------------------------------------------------

        let previewImg = null;   // base backdrop
        let imgW = 0, imgH = 0;  // base pixel dims
        // Per-embed thumbnail + its natural pixel dims + last-loaded url (dedupe).
        const embeds = {
            1: { img: null, w: 0, h: 0, url: "" },
            2: { img: null, w: 0, h: 0, url: "" },
            3: { img: null, w: 0, h: 0, url: "" },
        };
        let dragState = null;
        let selected = null;     // last-interacted embed id (border emphasis)
        let sizeLabel = null;    // per-embed px readout (DOM, below buttons)
        let sourceLabel = null;  // base provenance line

        // Placement guard: a connected embed is auto-staggered only the first
        // time it activates and only after the node is ready, so loading a saved
        // workflow (which already carries coords) never re-staggers.
        node._ccnInsetPlaced = node._ccnInsetPlaced || { 1: false, 2: false, 3: false };
        node._ccnLastEmbedPreview = node._ccnLastEmbedPreview || {};
        node._ccnReady = false;
        // Embeds auto-staggered before base dims were known — they get an aspect
        // fit once both base dims and the embed aspect are available. Saved/user
        // rects never enter this set, so a restored layout is never refitted.
        node._ccnPendingFit = node._ccnPendingFit || new Set();

        const cvs = document.createElement("canvas");

        function lockOn() {
            const w = node.widgets.find((w) => w.name === "lock_ratio");
            return w?.value ?? true;
        }

        // Pixel aspect (w/h) of an embed once its thumbnail dims are known.
        function embedAspect(id) {
            const e = embeds[id];
            return (e.w > 0 && e.h > 0) ? e.w / e.h : null;
        }

        // ----------------------------------------------------------------
        //  Wired / active detection (mirrors Python's resolver)
        // ----------------------------------------------------------------

        // The active upstream feeding an input, or null. A muted (mode 2) or
        // bypassed (mode 4) upstream counts as not connected — it yields no
        // image at run time, so the rect hides and nothing composites for it.
        function activeUpstreamNode(inputName) {
            const input = node.inputs?.find((i) => i.name === inputName);
            if (input?.link == null) return null;
            const link = app.graph.links[input.link];
            if (!link) return null;
            const up = app.graph.getNodeById(link.origin_id);
            if (!up) return null;
            if (up.mode === 2 || up.mode === 4) return null;
            return up;
        }

        function baseWired() {
            return activeUpstreamNode("image") != null;
        }

        function embedActive(id) {
            return activeUpstreamNode(`embed${id}`) != null;
        }

        // Find a preview URL from an upstream node — the standard .imgs array,
        // else any <img> inside a DOM widget (custom nodes like VideoScrubber).
        function findUpstreamPreviewUrl(upstreamNode) {
            if (!upstreamNode) return null;
            // 1) Standard ComfyUI preview array.
            if (upstreamNode.imgs?.[0]?.src) return upstreamNode.imgs[0].src;
            // 2) CCN convention: a node exposing a live preview of its own
            //    output (string URL or { src }). Lets transforming nodes like
            //    the crop node be previewed without a queue.
            const ccn = upstreamNode.ccn_image;
            if (ccn) return typeof ccn === "string" ? ccn : (ccn.src || null);
            // 3) Any <img> inside a DOM widget (custom nodes like VideoScrubber).
            for (const w of upstreamNode.widgets || []) {
                const el = w.element || w.inputEl;
                if (!el) continue;
                if (el.tagName === "IMG" && el.src) return el.src;
                const img = el.querySelector?.("img");
                if (img?.src && img.naturalWidth > 0) return img.src;
            }
            return null;
        }

        // ----------------------------------------------------------------
        //  Coordinate conversion
        // ----------------------------------------------------------------

        // Rectangle the base image occupies inside the canvas (preserves aspect).
        function displayRect() {
            const W = cvs.width, H = cvs.height;
            if (!imgW || !imgH) return { x: 0, y: 0, w: W, h: H };
            const iA = imgW / imgH, cA = W / H;
            let dw, dh, dx, dy;
            if (iA > cA) { dw = W; dh = W / iA; dx = 0; dy = (H - dh) / 2; }
            else { dh = H; dw = H * iA; dx = (W - dw) / 2; dy = 0; }
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
        //  Rect placement / aspect fitting
        // ----------------------------------------------------------------

        // Rebuild a rect at the embed's aspect, preserving its center and its
        // size relative to the base's longest edge, fitted inside the frame.
        // No-op without base dims (can't map pixel aspect to normalized space).
        function refitRect(id, sizeFracOverride) {
            if (!imgW || !imgH) return;
            const aspect = embedAspect(id) ?? 1;
            const c = getRect(id);
            const curPxW = (c.x2 - c.x1) * imgW;
            const curPxH = (c.y2 - c.y1) * imgH;
            const longEdge = Math.max(imgW, imgH);
            const sizeFrac = (sizeFracOverride != null)
                ? sizeFracOverride
                : (Math.max(curPxW, curPxH) / longEdge || DEFAULT_SIZE_FRAC);

            let targetLong = sizeFrac * longEdge;
            let pxW, pxH;
            if (aspect >= 1) { pxW = targetLong; pxH = pxW / aspect; }
            else { pxH = targetLong; pxW = pxH * aspect; }

            // Fit inside the frame without distorting the aspect.
            if (pxW > imgW) { pxW = imgW; pxH = pxW / aspect; }
            if (pxH > imgH) { pxH = imgH; pxW = pxH * aspect; }

            const cx = (c.x1 + c.x2) / 2;
            const cy = (c.y1 + c.y2) / 2;
            const halfW = (pxW / imgW) / 2;
            const halfH = (pxH / imgH) / 2;
            const ccx = Math.max(halfW, Math.min(1 - halfW, cx));
            const ccy = Math.max(halfH, Math.min(1 - halfH, cy));
            setRect(id, ccx - halfW, ccy - halfH, ccx + halfW, ccy + halfH);
        }

        // Drop a rect at its staggered home position, aspect-correct if known.
        function placeStaggered(id) {
            const st = STAGGER[id];
            if (!imgW || !imgH) {
                const s = DEFAULT_SIZE_FRAC;
                setRect(id, st.cx - s / 2, st.cy - s / 2, st.cx + s / 2, st.cy + s / 2);
                node._ccnPendingFit.add(id);  // fit to aspect once base dims load
                return;
            }
            // Seed a tiny box at the staggered center, then size it by aspect.
            setRect(id, st.cx - 0.01, st.cy - 0.01, st.cx + 0.01, st.cy + 0.01);
            refitRect(id, DEFAULT_SIZE_FRAC);
            node._ccnPendingFit.delete(id);
        }

        // Apply the deferred aspect fit for an auto-staggered rect, once both
        // base dims and the embed aspect are known (and lock is on).
        function tryPendingFit(id) {
            if (!node._ccnPendingFit.has(id)) return;
            if (!imgW || !imgH || embedAspect(id) == null) return;
            if (lockOn()) refitRect(id, DEFAULT_SIZE_FRAC);
            node._ccnPendingFit.delete(id);
        }

        // When the base changes dimensions, hold each rect's intent: with lock
        // on, preserve the embed aspect (refit); with lock off, keep the
        // normalized box (same relative position and size on the new frame).
        function adaptRectsToBase(oldW, oldH, newW, newH) {
            if (!oldW || !oldH || (oldW === newW && oldH === newH)) return;
            if (!lockOn()) return;
            for (const id of EMBED_IDS) {
                if (embedActive(id)) refitRect(id);
            }
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
                    "No base \u2014 Load Preview, Upload Image, or queue once",
                    W / 2, H / 2,
                );
                updateSizeLabel();
                return;
            }

            const r = displayRect();
            ctx.drawImage(previewImg, r.x, r.y, r.w, r.h);

            // Draw 1->3 so blue lands on top, matching Python's paste order.
            for (const id of EMBED_IDS) {
                if (!embedActive(id)) continue;
                drawEmbed(ctx, id);
            }

            updateSizeLabel();
        }

        function drawEmbed(ctx, id) {
            const c = getRect(id);
            const tl = toCanvas(c.x1, c.y1);
            const br = toCanvas(c.x2, c.y2);
            const bw = br.x - tl.x;
            const bh = br.y - tl.y;
            const color = EMBED_COLORS[id];

            const e = embeds[id];
            if (e.img) {
                // WYSIWYG fill — exactly how Python pastes (stretch to box).
                ctx.drawImage(e.img, tl.x, tl.y, bw, bh);
            } else {
                // No thumbnail yet — translucent placeholder with a label.
                ctx.fillStyle = color + "33";
                ctx.fillRect(tl.x, tl.y, bw, bh);
                ctx.fillStyle = "#ddd";
                ctx.font = "11px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(
                    `embed${id} \u2014 no preview`,
                    tl.x + bw / 2, tl.y + bh / 2,
                );
            }

            // Colored border, emphasized when this is the selected embed.
            ctx.strokeStyle = color;
            ctx.lineWidth = (selected === id) ? 2.5 : 1.5;
            ctx.strokeRect(tl.x, tl.y, bw, bh);

            // Corner handles in the embed color.
            ctx.fillStyle = color;
            const hs = HANDLE_SIZE;
            const corners = [
                tl, { x: br.x, y: tl.y }, { x: tl.x, y: br.y }, br,
            ];
            for (const pt of corners) {
                ctx.fillRect(pt.x - hs / 2, pt.y - hs / 2, hs, hs);
            }
        }

        // Per-embed pixel-size readout, one line per active embed.
        function updateSizeLabel() {
            if (!sizeLabel) return;
            if (!previewImg || !imgW || !imgH) {
                sizeLabel.textContent = "\u2014";
                return;
            }
            const lines = [];
            for (const id of EMBED_IDS) {
                if (!embedActive(id)) continue;
                const c = getRect(id);
                const pw = Math.round((c.x2 - c.x1) * imgW);
                const ph = Math.round((c.y2 - c.y1) * imgH);
                const ratio = ph > 0 ? (pw / ph) : 0;
                lines.push(`${EMBED_LETTER[id]}  ${pw} \u00d7 ${ph} px  (${ratio.toFixed(2)})`);
            }
            sizeLabel.textContent = lines.length ? lines.join("\n") : "no embeds connected";
        }

        // ----------------------------------------------------------------
        //  Hit detection (topmost-first: blue > green > red)
        // ----------------------------------------------------------------

        function cornersFor(id) {
            const c = getRect(id);
            return {
                tl: toCanvas(c.x1, c.y1),
                tr: toCanvas(c.x2, c.y1),
                bl: toCanvas(c.x1, c.y2),
                br: toCanvas(c.x2, c.y2),
            };
        }

        function hitCorner(id, cx, cy) {
            const pts = cornersFor(id);
            for (const name of ["tl", "tr", "bl", "br"]) {
                const p = pts[name];
                if (Math.hypot(cx - p.x, cy - p.y) <= CORNER_HIT_RADIUS) return name;
            }
            return null;
        }

        function hitInterior(id, cx, cy) {
            const c = getRect(id);
            const n = toNorm(cx, cy);
            return n.x > c.x1 && n.x < c.x2 && n.y > c.y1 && n.y < c.y2;
        }

        // Returns { id, corner } | { id, corner:null } | null. Topmost active
        // embed wins, checking its corners before its interior.
        function hitTest(cx, cy) {
            for (let i = EMBED_IDS.length - 1; i >= 0; i--) {
                const id = EMBED_IDS[i];
                if (!embedActive(id)) continue;
                const corner = hitCorner(id, cx, cy);
                if (corner) return { id, corner };
                if (hitInterior(id, cx, cy)) return { id, corner: null };
            }
            return null;
        }

        function cursorFor(corner) {
            return (corner === "tl" || corner === "br")
                ? "nwse-resize" : "nesw-resize";
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

        function onMouseDown(e) {
            if (!previewImg) return;
            const pos = mousePos(e);
            const hit = hitTest(pos.x, pos.y);
            if (!hit) return;

            selected = hit.id;

            if (hit.corner) {
                dragState = {
                    type: "corner",
                    id: hit.id,
                    corner: hit.corner,
                    startRect: getRect(hit.id),
                };
            } else {
                const n = toNorm(pos.x, pos.y);
                const c = getRect(hit.id);
                dragState = {
                    type: "move",
                    id: hit.id,
                    offX: n.x - c.x1,
                    offY: n.y - c.y1,
                    w: c.x2 - c.x1,
                    h: c.y2 - c.y1,
                };
            }

            e.preventDefault();
            e.stopPropagation();
            document.addEventListener("mousemove", onDragMove, true);
            document.addEventListener("mouseup", onDragEnd, true);
            draw();
        }

        function onDragMove(e) {
            if (!dragState) return;
            e.preventDefault();
            const pos = mousePos(e);
            const n = toNorm(pos.x, pos.y);
            if (dragState.type === "corner") applyCornerDrag(n);
            else applyMoveDrag(n);
            draw();
        }

        function applyCornerDrag(n) {
            const id = dragState.id;
            let { x1, y1, x2, y2 } = getRect(id);
            const c = dragState.corner;

            // The diagonal opposite corner stays fixed.
            let fixX, fixY;
            if (c === "tl") { fixX = x2; fixY = y2; }
            else if (c === "tr") { fixX = x1; fixY = y2; }
            else if (c === "bl") { fixX = x2; fixY = y1; }
            else { fixX = x1; fixY = y1; }

            // Lock holds the embed's pixel aspect, if known.
            const ratio = lockOn() ? embedAspect(id) : null;

            if (ratio !== null && ratio !== undefined) {
                const sc = dragState.startRect;
                const startNX = (c === "tl" || c === "bl") ? sc.x1 : sc.x2;
                const startNY = (c === "tl" || c === "tr") ? sc.y1 : sc.y2;
                const dx = Math.abs(n.x - startNX);
                const dy = Math.abs(n.y - startNY);

                // ratio is pixel aspect (pxW/pxH); convert to normalized space.
                const normRatio = ratio * (imgH / Math.max(imgW, 1));

                let newNW = Math.abs(n.x - fixX);
                let newNH = Math.abs(n.y - fixY);
                if (dx >= dy) newNH = newNW / normRatio;
                else newNW = newNH * normRatio;

                const signX = n.x >= fixX ? 1 : -1;
                const signY = n.y >= fixY ? 1 : -1;
                const dragX = fixX + signX * newNW;
                const dragY = fixY + signY * newNH;

                x1 = Math.min(fixX, dragX);
                y1 = Math.min(fixY, dragY);
                x2 = Math.max(fixX, dragX);
                y2 = Math.max(fixY, dragY);
            } else {
                if (c === "tl") { x1 = n.x; y1 = n.y; }
                else if (c === "tr") { x2 = n.x; y1 = n.y; }
                else if (c === "bl") { x1 = n.x; y2 = n.y; }
                else { x2 = n.x; y2 = n.y; }
                if (x1 > x2) { const t = x1; x1 = x2; x2 = t; }
                if (y1 > y2) { const t = y1; y1 = y2; y2 = t; }
            }

            setRect(id, x1, y1, x2, y2);
        }

        function applyMoveDrag(n) {
            const id = dragState.id;
            let nx1 = n.x - dragState.offX;
            let ny1 = n.y - dragState.offY;
            nx1 = Math.max(0, Math.min(nx1, 1 - dragState.w));
            ny1 = Math.max(0, Math.min(ny1, 1 - dragState.h));
            setRect(id, nx1, ny1, nx1 + dragState.w, ny1 + dragState.h);
        }

        function onDragEnd() {
            if (dragState) node._ccnInsetPlaced[dragState.id] = true;
            dragState = null;
            document.removeEventListener("mousemove", onDragMove, true);
            document.removeEventListener("mouseup", onDragEnd, true);
        }

        cvs.addEventListener("mousemove", (e) => {
            if (dragState) return;
            if (!previewImg) { cvs.style.cursor = "default"; return; }
            const pos = mousePos(e);
            const hit = hitTest(pos.x, pos.y);
            if (hit?.corner) cvs.style.cursor = cursorFor(hit.corner);
            else if (hit) cvs.style.cursor = "move";
            else cvs.style.cursor = "default";
        });

        cvs.addEventListener("mousedown", onMouseDown);

        // ----------------------------------------------------------------
        //  Image loading
        // ----------------------------------------------------------------

        function loadBaseImage(url, source) {
            const img = new window.Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
                const oldW = imgW, oldH = imgH;
                previewImg = img;
                imgW = img.naturalWidth;
                imgH = img.naturalHeight;
                adaptRectsToBase(oldW, oldH, imgW, imgH);
                // Base dims are now known — fit any auto-staggered rects waiting
                // on them (saved/user rects are not in the pending set).
                for (const id of EMBED_IDS) tryPendingFit(id);
                if (source !== undefined) setSource(source);
                draw();
            };
            img.onerror = () =>
                console.warn("[CCN ImageInset] Failed to load base preview:", url);
            img.src = url;
        }

        function loadEmbedThumb(id, url) {
            const img = new window.Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
                embeds[id].img = img;
                embeds[id].w = img.naturalWidth;
                embeds[id].h = img.naturalHeight;
                embeds[id].url = url;
                // Aspect just became known — fit it only if this rect is an
                // auto-staggered one still awaiting a fit; never touch a
                // user-placed or workflow-restored rect.
                tryPendingFit(id);
                draw();
            };
            img.onerror = () =>
                console.warn(`[CCN ImageInset] Failed to load embed${id} preview:`, url);
            img.src = url;
        }

        function setSource(text) {
            if (sourceLabel) sourceLabel.textContent = text || "";
        }

        // ----------------------------------------------------------------
        //  Preview resolution
        // ----------------------------------------------------------------

        // Load the base backdrop for whatever drives the run. Explicit actions
        // and workflow load only — never on execute.
        //   Wired:     upstream's live cached preview, else the source the node
        //              last processed (saved server-side), else nothing.
        //   Not wired: the current loaded_image selection.
        function loadBestPreview() {
            if (baseWired()) {
                const up = activeUpstreamNode("image");
                const wired = up ? findUpstreamPreviewUrl(up) : null;
                if (wired) { loadBaseImage(wired, "wired source"); return true; }
                if (node._ccnLastPreview) {
                    loadBaseImage(node._ccnLastPreview.url, node._ccnLastPreview.source);
                    return true;
                }
                return false;
            }
            const lw = node.widgets.find((w) => w.name === "loaded_image");
            if (lw?.value && lw.value !== "none") {
                loadBaseImage(
                    api.apiURL(`/view?filename=${encodeURIComponent(lw.value)}&type=input`),
                    `loaded: ${lw.value}`,
                );
                return true;
            }
            return false;
        }

        // Pull each active embed's thumbnail — live upstream preview first, then
        // the thumbnail saved on the last run. Inactive embeds drop their thumb.
        function refreshEmbeds() {
            for (const id of EMBED_IDS) {
                if (!embedActive(id)) {
                    embeds[id].img = null;
                    embeds[id].url = "";
                    continue;
                }
                const up = activeUpstreamNode(`embed${id}`);
                let url = up ? findUpstreamPreviewUrl(up) : null;
                if (!url && node._ccnLastEmbedPreview[id]) {
                    url = node._ccnLastEmbedPreview[id];
                }
                if (url) {
                    if (embeds[id].url !== url) loadEmbedThumb(id, url);
                } else {
                    // Active but no thumbnail available yet — show placeholder.
                    embeds[id].img = null;
                    embeds[id].url = "";
                }
            }
            draw();
        }

        // Stagger newly-active embeds (once, after ready) and refresh thumbs.
        function handleConnections() {
            if (node._ccnReady) {
                for (const id of EMBED_IDS) {
                    if (embedActive(id) && !node._ccnInsetPlaced[id]) {
                        placeStaggered(id);
                        node._ccnInsetPlaced[id] = true;
                    }
                }
            }
            refreshEmbeds();
        }

        // ----------------------------------------------------------------
        //  DOM construction
        // ----------------------------------------------------------------

        const container = document.createElement("div");
        container.style.cssText =
            "width:100%;height:100%;box-sizing:border-box;" +
            "display:flex;flex-direction:column;";

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
        const btnReset = makeBtn("Reset Placements");
        btnRow.appendChild(btnPreview);
        btnRow.appendChild(btnUpload);
        btnRow.appendChild(btnReset);
        container.appendChild(btnRow);

        sizeLabel = document.createElement("div");
        sizeLabel.style.cssText =
            "flex:0 0 auto;white-space:pre-line;padding:3px 4px 1px;" +
            "font:11px monospace;color:#ccc;text-align:center;";
        sizeLabel.textContent = "\u2014";
        container.appendChild(sizeLabel);

        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "image/png,image/jpeg,image/webp,image/bmp,image/tiff";
        fileInput.style.display = "none";
        container.appendChild(fileInput);

        // ----------------------------------------------------------------
        //  Button handlers
        // ----------------------------------------------------------------

        btnPreview.addEventListener("click", () => {
            const ok = loadBestPreview();
            refreshEmbeds();
            if (!ok) {
                console.log(
                    "[CCN ImageInset] No base preview source. Queue once, wire " +
                    "an image, upload, or select one from the dropdown."
                );
            }
        });

        btnUpload.addEventListener("click", () => fileInput.click());

        fileInput.addEventListener("change", async () => {
            const file = fileInput.files?.[0];
            if (!file) return;

            const formData = new FormData();
            formData.append("image", file, file.name);
            formData.append("overwrite", "true");

            try {
                const resp = await api.fetchApi("/upload/image", {
                    method: "POST",
                    body: formData,
                });
                if (!resp.ok) {
                    console.error("[CCN ImageInset] Upload failed:", resp.statusText);
                    return;
                }
                const data = await resp.json();
                const uploadedName = data.name;

                const lw = node.widgets.find((w) => w.name === "loaded_image");
                if (lw) {
                    if (lw.options?.values && !lw.options.values.includes(uploadedName)) {
                        lw.options.values.push(uploadedName);
                    }
                    lw.value = uploadedName;
                }

                loadBaseImage(
                    api.apiURL(`/view?filename=${encodeURIComponent(uploadedName)}&type=input`),
                    `uploaded: ${uploadedName}`,
                );
            } catch (err) {
                console.error("[CCN ImageInset] Upload error:", err);
            }

            fileInput.value = "";
        });

        btnReset.addEventListener("click", () => {
            for (const id of EMBED_IDS) {
                placeStaggered(id);
                node._ccnInsetPlaced[id] = true;
            }
            selected = null;
            draw();
        });

        // Auto-load the base backdrop when the loaded_image dropdown changes.
        const loadedW = node.widgets.find((w) => w.name === "loaded_image");
        if (loadedW) {
            const origCb = loadedW.callback;
            loadedW.callback = function (...args) {
                origCb?.apply(this, args);
                if (loadedW.value && loadedW.value !== "none") {
                    loadBaseImage(
                        api.apiURL(`/view?filename=${encodeURIComponent(loadedW.value)}&type=input`),
                        `loaded: ${loadedW.value}`,
                    );
                }
            };
        }

        // ----------------------------------------------------------------
        //  Register DOM widget + lifecycle
        // ----------------------------------------------------------------

        // Canvas reports only a MINIMUM height; ComfyUI derives the node's
        // min-size from it while the element fills the available height above
        // the buttons, so the node scales up and still shrinks back down.
        node.addDOMWidget(
            "ccn_inset_canvas", "custom", container,
            { getValue: () => "", setValue: () => {}, getMinHeight: () => MIN_BOX },
        );

        // Keep the drawing buffer matched to the displayed size (one-way:
        // layout -> buffer), so it can't drive resize creep.
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

        // React to embeds being wired/unwired (and mute/bypass changes that
        // come through as connection events) without requiring a run.
        const origOCC = node.onConnectionsChange;
        node.onConnectionsChange = function () {
            origOCC?.apply(this, arguments);
            handleConnections();
        };

        // Restore on workflow load: coords come from the saved graph, so mark
        // active embeds placed (never re-stagger them), then load previews.
        const origConfigure = node.onConfigure;
        node.onConfigure = function () {
            origConfigure?.apply(this, arguments);
            requestAnimationFrame(() => {
                syncSize();
                for (const id of EMBED_IDS) {
                    if (embedActive(id)) node._ccnInsetPlaced[id] = true;
                }
                node._ccnReady = true;
                loadBestPreview();
                refreshEmbeds();
            });
        };

        const origRemoved = node.onRemoved;
        node.onRemoved = function () {
            observer.disconnect();
            origRemoved?.apply(this, arguments);
        };

        // Expose a live preview of THIS node's output (base + active embeds,
        // composited at full base resolution) so a downstream CCN node can show
        // it without a queue. Computed on read via an offscreen canvas, so
        // normal interaction pays nothing; null until a base backdrop is loaded.
        // Embeds without a loaded thumbnail are skipped (we have no pixels for
        // them yet), matching the on-canvas placeholder. try/catch guards a
        // tainted-canvas read so the consumer's search just falls through.
        Object.defineProperty(node, "ccn_image", {
            configurable: true,
            get() {
                if (!previewImg || !imgW || !imgH) return null;
                try {
                    const off = document.createElement("canvas");
                    off.width = imgW;
                    off.height = imgH;
                    const octx = off.getContext("2d");
                    octx.drawImage(previewImg, 0, 0, imgW, imgH);
                    // 1->3 so blue lands on top, matching Python's paste order.
                    for (const id of EMBED_IDS) {
                        if (!embedActive(id)) continue;
                        const e = embeds[id];
                        if (!e.img) continue;
                        const c = getRect(id);
                        const x = Math.round(c.x1 * imgW);
                        const y = Math.round(c.y1 * imgH);
                        const w = Math.max(1, Math.round((c.x2 - c.x1) * imgW));
                        const h = Math.max(1, Math.round((c.y2 - c.y1) * imgH));
                        octx.drawImage(e.img, x, y, w, h);
                    }
                    return off.toDataURL("image/png");
                } catch (err) {
                    console.warn("[CCN ImageInset] ccn_image getter failed:", err);
                    return null;
                }
            },
        });

        requestAnimationFrame(() => {
            syncSize();
            node._ccnReady = true;
            loadBestPreview();   // base backdrop if one is ready
            refreshEmbeds();     // any already-active embed thumbnails
            draw();
        });
    },
});
