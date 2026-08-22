import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/*
 * Composed stitch preview for CCN_ImageStitch.
 *
 * Passive canvas — no dragging. The stitch layout (scale-to-first, divider
 * strips between adjacent images) is composed client-side from each wired
 * slot's preview, so Load Preview shows the result without executing the
 * graph. Slot thumbnails are fetched the same way the other CCN canvas
 * nodes fetch theirs: the live upstream preview when available, else the
 * per-slot thumbnail the node saved on its last run.
 *
 * A slot counts only when wired to an enabled upstream; a muted/bypassed
 * or unwired slot is skipped, matching what Python resolves. Wired slots
 * whose preview pixels aren't available yet are skipped in the composition
 * (their aspect is unknown) and called out in the status line.
 *
 * WYSIWYG caveat: upstream previews may be downscaled thumbnails, so the
 * status line's absolute pixel counts can undershoot the real output — but
 * scaling is relative to the first image's edge, so the composed geometry
 * stays aspect-correct regardless.
 *
 * Thumbnails change only on explicit action (Load Preview, a connection
 * change) or on workflow load — never on execution, so queueing never
 * disturbs the canvas.
 */

const CANVAS_HEIGHT = 350;
const MIN_BOX = 200;

const SLOT_IDS = [1, 2, 3];

app.registerExtension({
    name: "CCN.ImageStitch",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "CCN_ImageStitch") return;

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origOnExecuted?.apply(this, arguments);
            // Record what the node processed WITHOUT redrawing — a queue must
            // never swap the canvas under the user. Load Preview pulls these
            // up on demand (see refreshSlots). Custom UI keys (not "images")
            // so nothing hits the node preview or image feed.
            this._ccnLastSlotPreview = this._ccnLastSlotPreview || {};
            for (const id of SLOT_IDS) {
                const e = message?.[`ccn_stitch_image${id}`]?.[0];
                if (e) {
                    this._ccnLastSlotPreview[id] = api.apiURL(
                        `/view?filename=${encodeURIComponent(e.filename)}` +
                        `&type=${e.type}&subfolder=${e.subfolder || ""}`
                    );
                }
            }
        };
    },

    async nodeCreated(node) {
        if (node.comfyClass !== "CCN_ImageStitch") return;

        // Per-slot thumbnail + its natural pixel dims + last-loaded url (dedupe).
        const slots = {
            1: { img: null, w: 0, h: 0, url: "" },
            2: { img: null, w: 0, h: 0, url: "" },
            3: { img: null, w: 0, h: 0, url: "" },
        };
        let statusLabel = null;

        node._ccnLastSlotPreview = node._ccnLastSlotPreview || {};

        const cvs = document.createElement("canvas");

        function directionValue() {
            const w = node.widgets.find((x) => x.name === "direction");
            return w?.value ?? "horizontal";
        }

        // Divider settings. Drawing-only fallbacks (0 / black) on malformed
        // values; Python validates at queue time.
        function borderWidthPx() {
            const w = node.widgets.find((x) => x.name === "border_width");
            const v = Number(w?.value);
            return Number.isFinite(v) ? Math.max(0, Math.floor(v)) : 0;
        }

        function borderFill() {
            const w = node.widgets.find((x) => x.name === "border_color");
            const v = String(w?.value ?? "").trim();
            if (/^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.test(v)) {
                return v.startsWith("#") ? v : "#" + v;
            }
            return "#000000";
        }

        // ----------------------------------------------------------------
        //  Wired / active detection (mirrors Python's resolver)
        // ----------------------------------------------------------------

        // The active upstream feeding an input plus the output slot the wire
        // leaves from ({ up, slot }), or null. A muted (mode 2) or bypassed
        // (mode 4) upstream counts as not connected — it yields no image at
        // run time, so its slot is skipped from the composition.
        function activeUpstreamNode(inputName) {
            const input = node.inputs?.find((i) => i.name === inputName);
            if (input?.link == null) return null;
            const link = app.graph.links[input.link];
            if (!link) return null;
            const up = app.graph.getNodeById(link.origin_id);
            if (!up) return null;
            if (up.mode === 2 || up.mode === 4) return null;
            return { up, slot: link.origin_slot };
        }

        function slotActive(id) {
            return activeUpstreamNode(`image${id}`) != null;
        }

        // Find a preview URL from an upstream node — the standard .imgs array,
        // else any <img> inside a DOM widget (custom nodes like VideoScrubber).
        function findUpstreamPreviewUrl(upstreamNode, originSlot) {
            if (!upstreamNode) return null;
            // 1) Standard ComfyUI preview array.
            if (upstreamNode.imgs?.[0]?.src) return upstreamNode.imgs[0].src;
            // 2) CCN convention: a node exposing a live preview of its own
            //    output. Slot-aware hook first — nodes whose outputs carry
            //    different pixels (crop vs source_image) resolve the preview
            //    for the exact slot the wire leaves from. Plain ccn_image
            //    (string URL or { src }) is the single-preview fallback.
            const forSlot = upstreamNode.ccn_imageForSlot?.(originSlot);
            if (forSlot) return forSlot;
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
        //  Layout + drawing
        // ----------------------------------------------------------------

        // Compose the stitch in preview pixels, mirroring Python: the first
        // loaded slot sets the shared edge, the rest scale to match (aspect
        // preserved), divider strips only between adjacent segments. Returns
        // null when no loaded thumbnails exist, else { W, H, segs, borders,
        // ids, missing }.
        function composedLayout() {
            const active = SLOT_IDS.filter(slotActive);
            const loaded = active.filter((id) => slots[id].img);
            const missing = active.filter((id) => !slots[id].img);
            if (!loaded.length) return null;

            const horiz = directionValue() === "horizontal";
            const first = slots[loaded[0]];
            const bp = loaded.length > 1 ? borderWidthPx() : 0;
            const segs = [];
            const borders = [];
            let cursor = 0;

            for (const id of loaded) {
                const s = slots[id];
                if (segs.length && bp > 0) {
                    borders.push(horiz
                        ? { x: cursor, y: 0, w: bp, h: first.h }
                        : { x: 0, y: cursor, w: first.w, h: bp });
                    cursor += bp;
                }
                if (horiz) {
                    const w = Math.max(1, Math.round(s.w * first.h / s.h));
                    segs.push({ img: s.img, x: cursor, y: 0, w, h: first.h });
                    cursor += w;
                } else {
                    const h = Math.max(1, Math.round(s.h * first.w / s.w));
                    segs.push({ img: s.img, x: 0, y: cursor, w: first.w, h });
                    cursor += h;
                }
            }

            return {
                W: horiz ? cursor : first.w,
                H: horiz ? first.h : cursor,
                segs, borders, ids: loaded, missing,
            };
        }

        function draw() {
            const ctx = cvs.getContext("2d");
            const W = cvs.width, H = cvs.height;
            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = "#1a1a1a";
            ctx.fillRect(0, 0, W, H);

            const layout = composedLayout();
            if (!layout) {
                ctx.fillStyle = "#666";
                ctx.font = "13px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(
                    SLOT_IDS.some(slotActive)
                        ? "No previews — Load Preview or queue once"
                        : "No images wired — connect image1–3",
                    W / 2, H / 2,
                );
                updateStatus(null);
                return;
            }

            // Letterbox-fit the composed layout, drawing straight from the
            // natural thumbnails to display size (one resample, not two).
            const s = Math.min(W / layout.W, H / layout.H);
            const dx = (W - layout.W * s) / 2;
            const dy = (H - layout.H * s) / 2;

            if (layout.borders.length) {
                ctx.fillStyle = borderFill();
                for (const b of layout.borders) {
                    ctx.fillRect(
                        dx + b.x * s, dy + b.y * s,
                        Math.max(1, b.w * s), Math.max(1, b.h * s),
                    );
                }
            }
            for (const seg of layout.segs) {
                ctx.drawImage(
                    seg.img,
                    dx + seg.x * s, dy + seg.y * s, seg.w * s, seg.h * s,
                );
            }

            updateStatus(layout);
        }

        function updateStatus(layout) {
            if (!statusLabel) return;
            if (!layout) {
                statusLabel.textContent = "";
                return;
            }
            let text = `${layout.W} × ${layout.H} px — `
                + layout.ids.map((id) => `image${id}`).join(" + ");
            if (layout.missing.length) {
                text += `  |  ${layout.missing.map((id) => `image${id}`).join(", ")}`
                    + ": no preview — queue once or Load Preview";
            }
            statusLabel.textContent = text;
        }

        // ----------------------------------------------------------------
        //  Slot thumbnail loading
        // ----------------------------------------------------------------

        function loadSlotThumb(id, url) {
            const img = new window.Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
                slots[id].img = img;
                slots[id].w = img.naturalWidth;
                slots[id].h = img.naturalHeight;
                slots[id].url = url;
                draw();
            };
            img.onerror = () =>
                console.warn(`[CCN ImageStitch] Failed to load image${id} preview:`, url);
            img.src = url;
        }

        // Pull each active slot's thumbnail — live upstream preview first,
        // then the thumbnail saved on the last run. Inactive slots drop theirs.
        function refreshSlots() {
            for (const id of SLOT_IDS) {
                if (!slotActive(id)) {
                    slots[id].img = null;
                    slots[id].url = "";
                    continue;
                }
                const src = activeUpstreamNode(`image${id}`);
                let url = src ? findUpstreamPreviewUrl(src.up, src.slot) : null;
                if (!url && node._ccnLastSlotPreview[id]) {
                    url = node._ccnLastSlotPreview[id];
                }
                if (url) {
                    if (slots[id].url !== url) loadSlotThumb(id, url);
                } else {
                    // Active but no thumbnail available yet — status line
                    // names it; the composition skips it.
                    slots[id].img = null;
                    slots[id].url = "";
                }
            }
            draw();
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

        statusLabel = document.createElement("div");
        statusLabel.style.cssText =
            "flex:0 0 auto;padding:2px 4px 0;font:10px monospace;" +
            "color:#888;text-align:center;";
        statusLabel.textContent = "";
        container.appendChild(statusLabel);

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
        btnRow.appendChild(btnPreview);
        container.appendChild(btnRow);

        btnPreview.addEventListener("click", () => {
            refreshSlots();
            if (!SLOT_IDS.some(slotActive)) {
                console.log(
                    "[CCN ImageStitch] No images wired. Connect image1–3, " +
                    "then Load Preview."
                );
            }
        });

        // Redraw when a visible widget that affects the preview changes.
        for (const name of ["direction", "border_width", "border_color"]) {
            const w = node.widgets.find((x) => x.name === name);
            if (!w) continue;
            const origCb = w.callback;
            w.callback = function (...args) {
                origCb?.apply(this, args);
                draw();
            };
        }

        // ----------------------------------------------------------------
        //  Register DOM widget + lifecycle
        // ----------------------------------------------------------------

        // Canvas reports only a MINIMUM height; ComfyUI derives the node's
        // min-size from it while the element fills the available height above
        // the buttons, so the node scales up and still shrinks back down.
        node.addDOMWidget(
            "ccn_stitch_canvas", "custom", container,
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

        // React to slots being wired/unwired (and mute/bypass changes that
        // come through as connection events) without requiring a run.
        const origOCC = node.onConnectionsChange;
        node.onConnectionsChange = function () {
            origOCC?.apply(this, arguments);
            refreshSlots();
        };

        // Restore on workflow load: pull whatever previews are reachable.
        const origConfigure = node.onConfigure;
        node.onConfigure = function () {
            origConfigure?.apply(this, arguments);
            requestAnimationFrame(() => {
                syncSize();
                refreshSlots();
            });
        };

        const origRemoved = node.onRemoved;
        node.onRemoved = function () {
            observer.disconnect();
            origRemoved?.apply(this, arguments);
        };

        // Expose a live preview of THIS node's output (the composed stitch at
        // full preview resolution) so a downstream CCN node can show it
        // without a queue. Computed on read via an offscreen canvas, so
        // normal interaction pays nothing; null until a thumbnail is loaded.
        // try/catch guards a tainted-canvas read so the consumer's search
        // just falls through.
        Object.defineProperty(node, "ccn_image", {
            configurable: true,
            get() {
                const layout = composedLayout();
                if (!layout) return null;
                try {
                    const off = document.createElement("canvas");
                    off.width = layout.W;
                    off.height = layout.H;
                    const octx = off.getContext("2d");
                    if (layout.borders.length) {
                        octx.fillStyle = borderFill();
                        for (const b of layout.borders) {
                            octx.fillRect(b.x, b.y, b.w, b.h);
                        }
                    }
                    for (const seg of layout.segs) {
                        octx.drawImage(seg.img, seg.x, seg.y, seg.w, seg.h);
                    }
                    return off.toDataURL("image/png");
                } catch (err) {
                    console.warn("[CCN ImageStitch] ccn_image getter failed:", err);
                    return null;
                }
            },
        });

        requestAnimationFrame(() => {
            syncSize();
            refreshSlots();   // any already-active slot thumbnails (+ draw)
        });
    },
});
