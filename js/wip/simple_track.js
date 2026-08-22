import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/* ───────────────────────────────────────────────────────────────
   CCN Simple Track — timeline widget for CCN_SimpleTrack.

   Layout (top to bottom): composited preview canvas, scrub slider,
   selected-clip name, then a lower canvas with a time ruler, two
   thumbnail tracks, and a volume-curve editor for the selected clip.

   Clips arrive on dynamic video_<n>/audio_<n> input pairs; wiring
   video_<n> grows the next pair. Clip pixels come from two sources,
   both normalized to "a video file the scrubber routes can serve":

     • live upstream discovery — walk the video_<n> link to a
       file-backed loader (LoadVideo, CCN_VideoScrubber, VHS) and
       scrub that file directly, before any run;
     • materialized previews — after a run, message.ccn_simple_track
       carries a low-res mp4 per clip in temp/simple_track (this is
       how generated, in-memory clips get pixels). Materialized wins
       until the upstream file visibly changes.

   State serializes through the "track_data" widget (schema v1,
   documented in wip/simple_track.py). The auto-placement formula,
   crossfade sampling (smoothstep at frame centers), and volume-curve
   sampling (hermite at f/(frames-1), clamped 0..1) are mirrored from
   the Python side — keep them in sync. Selection is stored in
   node.properties.ccn_selected, NOT in track_data, so clicking a
   clip never invalidates the execution cache.
   ─────────────────────────────────────────────────────────────── */

const NODE_NAME = "CCN_SimpleTrack";

const MIN_NODE_W = 640;
const PREVIEW_MIN_H = 140;
const RULER_H = 14;
const TRACK_H = 44;
const TRACK_GAP = 2;
const CURVE_H = 74;
const LOWER_H = RULER_H + TRACK_H * 2 + TRACK_GAP * 2 + CURVE_H + 4;
// Slack covers the dom-widget wrapper margin (~20px) plus flex gaps, so the
// fixed-height children fit the element even at the widget's minimum height.
const MIN_UI_H = PREVIEW_MIN_H + 26 + 16 + LOWER_H + 25;
const PAD_X = 8;

const SNAP_PX = 6;
const DEBOUNCE_MS = 50;
const FRAME_CACHE_CAP = 30;
const MAX_THUMBS = 24;
const THUMB_CONCURRENCY = 2;
const WALK_DEPTH_MAX = 24;
const PX_PER_FRAME_MAX = 4;
const TIMELINE_TAIL = 0.08;
const PLACEHOLDER_W = 74;

const KEY_R = 5;
const HDL_R = 3.5;
const HDL_LEN = 26;
const MIN_KEY_DX = 0.015;

const DEFAULT_CLIP_CURVE = [
    { x: 0, y: 1, in: 0, out: 0, mirrored: true },
    { x: 1, y: 1, in: 0, out: 0, mirrored: true },
];

const C = {
    bg:        "#181825",
    panel:     "#14141e",
    grid:      "#27273a",
    gridEdge:  "#2e2e48",
    curve:     "#4ec9b0",
    key:       "#e0e0e0",
    keyStroke: "#000000",
    keySel:    "#f0c050",
    keyBroken: "#c07040",
    hdl:       "#8888aa",
    hdlLine:   "#55557a",
    playhead:  "#e06070",
    marker:    "#e06070",
    label:     "#6a6a80",
    text:      "#c0c0d0",
    trackBg:   ["#1d1d2e", "#191928"],
    clipTint:  ["#2a3a55", "#3a2a55"],
    clipEdge:  "#4a4a6a",
    snapGuide: "#f0c050",
};

/* ── Pure helpers ───────────────────────────────────────────── */

function clamp(v, lo, hi) {
    return Math.min(hi, Math.max(lo, v));
}

/* Mirrors Python _smoothstep: s(u) + s(1-u) = 1 exactly. */
function smoothstep(u) {
    u = clamp(u, 0, 1);
    return u * u * (3 - 2 * u);
}

/* ── Hermite (mirrors ccn_curve_widget.js / curve_cfg_guider.py) ── */
function hermite(keys, t) {
    const n = keys.length;
    if (n === 0) return 0;
    if (n === 1) return keys[0].y;
    if (t <= keys[0].x) return keys[0].y;
    if (t >= keys[n - 1].x) return keys[n - 1].y;
    let i = 0;
    for (; i < n - 1; i++) {
        if (t >= keys[i].x && t <= keys[i + 1].x) break;
    }
    const k0 = keys[i], k1 = keys[i + 1];
    const dt = k1.x - k0.x;
    if (dt < 1e-10) return k0.y;
    const lt = (t - k0.x) / dt, lt2 = lt * lt, lt3 = lt2 * lt;
    return (2 * lt3 - 3 * lt2 + 1) * k0.y
         + (lt3 - 2 * lt2 + lt) * k0.out * dt
         + (-2 * lt3 + 3 * lt2) * k1.y
         + (lt3 - lt2) * k1.in * dt;
}

/* Numeric-aware ordering of input suffixes ("wire order"). */
function wireCmp(a, b) {
    const na = Number(a), nb = Number(b);
    if (Number.isFinite(na) && Number.isFinite(nb)) return na - nb;
    return String(a).localeCompare(String(b));
}

function containRect(srcW, srcH, dstX, dstY, dstW, dstH) {
    if (srcW <= 0 || srcH <= 0) return { x: dstX, y: dstY, w: dstW, h: dstH };
    const scale = Math.min(dstW / srcW, dstH / srcH);
    const w = srcW * scale, h = srcH * scale;
    return { x: dstX + (dstW - w) / 2, y: dstY + (dstH - h) / 2, w, h };
}

async function decodeBlob(blob) {
    if (window.createImageBitmap) return await createImageBitmap(blob);
    const url = URL.createObjectURL(blob);
    try {
        const img = new Image();
        img.src = url;
        await img.decode();
        img._ccnBlobUrl = url;
        return img;
    } catch (e) {
        URL.revokeObjectURL(url);
        throw e;
    }
}

function releaseFrame(bmp) {
    if (!bmp) return;
    bmp.close?.();
    if (bmp._ccnBlobUrl) URL.revokeObjectURL(bmp._ccnBlobUrl);
}

/* ── Link resolution (verbatim from moe_sampler_dual.js) ───────── */
function resolveInput(node, slotName) {
    const slot = node.inputs?.findIndex((i) => i.name === slotName);
    if (slot == null || slot < 0) return null;

    let src = null;
    if (typeof node.getInputNode === "function") {
        try { src = node.getInputNode(slot); } catch (_) {}
    }
    if (!src) {
        const linkId = node.inputs[slot].link;
        if (linkId == null) return null;
        const graph = node.graph || app.graph;
        const link = graph?.links?.[linkId]
                  ?? (graph?.links?.get && graph.links.get(linkId));
        if (!link) return null;
        src = graph.getNodeById(link.origin_id);
    }
    if (!src) return null;
    if (src.mode === 2 || src.mode === 4) return null; // muted / bypassed

    let depth = 0;
    while (src && src.type === "Reroute" && depth++ < WALK_DEPTH_MAX) {
        src = resolveInput(src, src.inputs?.[0]?.name ?? "");
    }
    return src ?? null;
}

function widgetValue(node, name) {
    return node?.widgets?.find((w) => w.name === name)?.value;
}

/* File-backed sources the scrubber routes can serve directly. */
const SOURCE_WIDGETS = {
    LoadVideo: "file",
    CCN_VideoScrubber: "video",
    VHS_LoadVideo: "video",
};
/* Pass-through nodes to hop while looking for the file source. */
const PASSTHROUGH_INPUT = {
    GetVideoComponents: "video",
};

function discoverVideoSource(node, inputName) {
    let src = resolveInput(node, inputName);
    const originLabel = src ? (src.title || src.type) : null;
    for (let depth = 0; src && depth < WALK_DEPTH_MAX; depth++) {
        const cls = src.comfyClass ?? src.type;
        if (SOURCE_WIDGETS[cls] != null) {
            const filename = widgetValue(src, SOURCE_WIDGETS[cls]);
            if (typeof filename === "string" && filename) {
                return { filename, label: src.title || cls, originLabel };
            }
            return { filename: null, label: originLabel, originLabel };
        }
        if (PASSTHROUGH_INPUT[cls] != null) {
            src = resolveInput(src, PASSTHROUGH_INPUT[cls]);
            continue;
        }
        return { filename: null, label: originLabel, originLabel };
    }
    return { filename: null, label: originLabel, originLabel };
}

/* ── Per-node UI factory ────────────────────────────────────── */
function createTrackUI(node) {
    const model = {
        clips: [],       // Clip objects, wire order
        selectedKey: null,
        scrub: 0,        // timeline frame, UI-only
    };
    let lastPayload = null;
    let drag = null;       // {type:"clip"|"scrub"|"curve", ...}
    let curveSelKey = -1;  // selected curve keyframe index
    let debounceTimer = null;
    let lowerDirty = false;

    function makeClip(key) {
        return {
            key,
            track: model.clips.length % 2,
            start: 0,
            frames: 0,
            curve: JSON.parse(JSON.stringify(DEFAULT_CLIP_CURVE)),
            placed: false,
            source: null,          // {kind:"live"|"materialized", filename, label}
            info: null,            // {width, height, fps}
            infoState: "idle",     // idle | loading | ok | failed
            frameStore: { cache: new Map(), inFlight: false, queued: null },
            strip: null,           // {canvas, filename, n, frames}
            _stripToken: 0,
            _runLive: null,        // live filename at materialize time
        };
    }

    function destroyClip(clip) {
        for (const bmp of clip.frameStore.cache.values()) releaseFrame(bmp);
        clip.frameStore.cache.clear();
        clip.strip = null;
        clip._stripToken++;
        clip._stripBaking = false;
        if (drag?.clip === clip) drag = null;
    }

    function resetPixels(clip) {
        for (const bmp of clip.frameStore.cache.values()) releaseFrame(bmp);
        clip.frameStore.cache.clear();
        clip.frameStore.queued = null;
        clip.strip = null;
        clip._stripToken++;
        clip._stripBaking = false;
        clip.info = null;
        clip.infoState = "idle";
    }

    /* ── DOM ────────────────────────────────────────────────── */
    const container = document.createElement("div");
    container.style.cssText =
        "display:flex;flex-direction:column;gap:3px;width:100%;height:100%;" +
        "box-sizing:border-box;padding:2px 0;overflow:hidden;user-select:none;";

    const previewCanvas = document.createElement("canvas");
    // flex-basis 0, not auto: a canvas's auto basis is its buffer height,
    // and syncSize writes layout height back into the buffer — basis auto
    // would let that feedback loop ratchet the element ever taller. The
    // explicit min-height is the visibility floor (never collapses) and
    // also overrides min-height:auto, which for a canvas is intrinsic size.
    previewCanvas.style.cssText =
        `width:100%;flex:1 1 0;min-height:${PREVIEW_MIN_H}px;display:block;` +
        "background:#000;border-radius:4px;-webkit-user-drag:none;";

    const sliderRow = document.createElement("div");
    sliderRow.style.cssText =
        "display:flex;align-items:center;gap:4px;flex:0 0 auto;";
    const arrowCss =
        "flex:0 0 auto;padding:2px 9px;cursor:pointer;line-height:1;" +
        "border:1px solid #555;background:#333;color:#ddd;" +
        "border-radius:4px;font-size:12px;";
    const prevBtn = document.createElement("button");
    prevBtn.textContent = "◀";
    prevBtn.title = "Step back one frame";
    prevBtn.style.cssText = arrowCss;
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "0";
    slider.value = "0";
    slider.style.cssText = "flex:1 1 auto;min-width:0;margin:0;cursor:pointer;";
    const nextBtn = document.createElement("button");
    nextBtn.textContent = "▶";
    nextBtn.title = "Step forward one frame";
    nextBtn.style.cssText = arrowCss;
    const frameLabel = document.createElement("span");
    frameLabel.style.cssText =
        "flex:0 0 auto;font:11px monospace;color:#999;user-select:none;" +
        "min-width:110px;text-align:right;";
    frameLabel.textContent = "0 / 0";
    sliderRow.appendChild(prevBtn);
    sliderRow.appendChild(slider);
    sliderRow.appendChild(nextBtn);
    sliderRow.appendChild(frameLabel);

    const nameLabel = document.createElement("div");
    nameLabel.style.cssText =
        "flex:0 0 16px;font:11px monospace;color:#aaa;user-select:none;" +
        "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" +
        "padding:0 4px;";
    nameLabel.textContent = "";

    const lowerCanvas = document.createElement("canvas");
    // Height pinned three ways: min-height:auto on a canvas resolves to the
    // buffer height, so a bare flex-basis lets the syncSize buffer write-back
    // ratchet the element taller on every layout pass (the "ever-growing
    // track" bug). Explicit min/max/height breaks that loop for good.
    lowerCanvas.style.cssText =
        `width:100%;flex:0 0 ${LOWER_H}px;height:${LOWER_H}px;` +
        `min-height:${LOWER_H}px;max-height:${LOWER_H}px;display:block;` +
        "background:#111;border-radius:4px;touch-action:none;" +
        "-webkit-user-drag:none;";

    container.appendChild(previewCanvas);
    container.appendChild(sliderRow);
    container.appendChild(nameLabel);
    container.appendChild(lowerCanvas);

    // Native HTML5 drag must never start inside the widget: once the browser
    // begins ghost-dragging the element, pointer events stop reaching the
    // scrub/track handlers entirely (video_scrubber sets draggable=false on
    // its <img> for the same reason). The capture-phase dragstart kill covers
    // any element a selection or the DOM-widget wrapper might try to drag.
    for (const el of [container, previewCanvas, lowerCanvas, nameLabel]) {
        el.draggable = false;
    }
    container.addEventListener("dragstart", (e) => {
        e.preventDefault();
        e.stopPropagation();
    }, true);
    container.addEventListener("selectstart", (e) => e.preventDefault(), true);

    const offscreen = document.createElement("canvas");

    /* ── track_data serialization ───────────────────────────── */
    const r4 = (v) => Math.round((Number(v) || 0) * 10000) / 10000;

    function getValue() {
        // Unplaced clips (length unknown) are omitted so the Python side
        // auto-places them with the shared formula instead of trusting a
        // meaningless start/frames pair.
        const clips = model.clips
            .filter((c) => c.placed && c.frames > 0)
            .slice()
            .sort((a, b) => wireCmp(a.key, b.key))
            .map((c) => ({
                key: c.key,
                track: c.track,
                start: Math.round(c.start),
                frames: Math.round(c.frames),
                curve: c.curve.map((k) => ({
                    x: r4(k.x), y: r4(k.y), in: r4(k.in), out: r4(k.out),
                    mirrored: k.mirrored !== false,
                })),
            }));
        return JSON.stringify({ version: 1, clips });
    }

    function setValue(v) {
        try {
            const data = JSON.parse(v || "{}");
            if (!data || !Array.isArray(data.clips)) return;
            for (const e of data.clips) {
                const key = String(e.key ?? "");
                if (!key) continue;
                let clip = model.clips.find((c) => c.key === key);
                if (!clip) {
                    clip = makeClip(key);
                    model.clips.push(clip);
                }
                clip.track = e.track === 1 ? 1 : 0;
                clip.start = Math.max(0, Math.round(Number(e.start) || 0));
                clip.frames = Math.max(0, Math.round(Number(e.frames) || 0));
                if (Array.isArray(e.curve) && e.curve.length >= 2) {
                    clip.curve = e.curve.map((k) => ({
                        x: Number(k.x) || 0,
                        y: Number.isFinite(Number(k.y)) ? Number(k.y) : 1,
                        in: Number(k.in) || 0,
                        out: Number(k.out) || 0,
                        mirrored: k.mirrored !== false,
                    }));
                }
                // A restored layout is never re-auto-placed.
                clip.placed = true;
            }
            model.clips.sort((a, b) => wireCmp(a.key, b.key));
        } catch (_) { /* keep current model on bad data */ }
        updateSliderRange();
        scheduleLower();
    }

    /* sync() commits a model mutation: serialization is lazy through
       getValue, so this only refreshes ranges and repaints. */
    function sync() {
        updateSliderRange();
        node.graph?.setDirtyCanvas?.(true, true);
        scheduleLower();
        requestComposite(model.scrub);
    }

    /* ── Timeline math ──────────────────────────────────────── */
    function placedClips() {
        return model.clips.filter((c) => c.placed && c.frames > 0);
    }

    function timelineTotal() {
        let end = 0;
        for (const c of placedClips()) end = Math.max(end, c.start + c.frames);
        return Math.max(end, 1);
    }

    function layout() {
        const w = lowerCanvas.width || 1;
        const total = timelineTotal();
        const ppf = Math.max(0.001, Math.min(
            (w - 2 * PAD_X) / (total * (1 + TIMELINE_TAIL)),
            PX_PER_FRAME_MAX));
        const track0Y = RULER_H + TRACK_GAP;
        const track1Y = track0Y + TRACK_H + TRACK_GAP;
        return {
            w, total, ppf,
            ruler: { y: 0, h: RULER_H },
            tracks: [{ y: track0Y, h: TRACK_H }, { y: track1Y, h: TRACK_H }],
            tracksBottom: track1Y + TRACK_H,
            curve: { y: track1Y + TRACK_H + 4, h: CURVE_H },
            xFromT: (t) => PAD_X + t * ppf,
            tFromX: (x) => (x - PAD_X) / ppf,
        };
    }

    function fpsValue() {
        const v = Number(widgetValue(node, "fps"));
        return Number.isFinite(v) && v >= 1 ? v : 16;
    }

    function defaultOverlap() {
        const v = Number(widgetValue(node, "default_overlap"));
        return Number.isFinite(v) && v >= 0 ? Math.round(v) : 8;
    }

    function outputDims() {
        let w = Math.round(Number(widgetValue(node, "width")) || 0);
        let h = Math.round(Number(widgetValue(node, "height")) || 0);
        if (w > 0 && h > 0) return [w, h];
        const first = model.clips.slice().sort((a, b) => wireCmp(a.key, b.key))[0];
        return [
            w > 0 ? w : (first?.info?.width || lastPayload?.width || 512),
            h > 0 ? h : (first?.info?.height || lastPayload?.height || 512),
        ];
    }

    /* Coverage + crossfade weight — sampling convention pinned with the
       Python render: w = smoothstep((t - ovStart + 0.5) / ovLen), and the
       same (start asc, end desc, key) tie-break for who fades in. */
    function coverageAt(t) {
        const covering = placedClips()
            .filter((c) => t >= c.start && t < c.start + c.frames)
            .sort((a, b) =>
                (a.start - b.start) ||
                ((b.start + b.frames) - (a.start + a.frames)) ||
                wireCmp(a.key, b.key));
        if (!covering.length) return {};
        if (covering.length === 1) return { a: covering[0] };
        const a = covering[0], b = covering[1];
        const ovLen = a.start + a.frames - b.start;
        const w = smoothstep((t - b.start + 0.5) / Math.max(ovLen, 1e-6));
        return { a, b, w };
    }

    function localFrame(clip, t) {
        return clamp(Math.round(t - clip.start), 0, clip.frames - 1);
    }

    /* Auto-placement — the exact formula the Python fallback uses. */
    function autoPlaceClip(clip) {
        const others = placedClips().filter((c) => c !== clip);
        if (!others.length) {
            clip.start = 0;
            clip.track = 0;
        } else {
            const prev = others.reduce((best, c) =>
                (c.start + c.frames > best.start + best.frames ? c : best));
            const overlap = Math.max(0, Math.min(
                defaultOverlap(), prev.frames - 1, clip.frames - 1));
            clip.start = Math.max(0, prev.start + prev.frames - overlap);
            clip.track = 1 - prev.track;
        }
        clip.placed = true;
        fixSameTrackOverlaps();
    }

    /* Place every clip whose length has become known, in wire order, so
       simultaneous arrivals land deterministically. */
    function placePending() {
        let placedAny = false;
        for (const clip of model.clips.slice().sort((a, b) => wireCmp(a.key, b.key))) {
            if (!clip.placed && clip.frames > 0) {
                autoPlaceClip(clip);
                placedAny = true;
            }
        }
        if (placedAny) sync();
    }

    /* Same-track overlap is invalid (cross-track overlap is the feature);
       push later clips right just enough to abut. */
    function fixSameTrackOverlaps() {
        for (const track of [0, 1]) {
            const list = placedClips()
                .filter((c) => c.track === track)
                .sort((a, b) => (a.start - b.start) || wireCmp(a.key, b.key));
            let minStart = 0;
            for (const c of list) {
                if (c.start < minStart) c.start = minStart;
                minStart = c.start + c.frames;
            }
        }
    }

    /* ── Sources and metadata ───────────────────────────────── */
    function clipFilename(clip) {
        return clip.source?.filename ?? null;
    }

    function refreshSource(clip) {
        const live = discoverVideoSource(node, `video_${clip.key}`);
        if (clip.source?.kind === "materialized") {
            // Materialized previews reflect executed pixels and win — until
            // the discovered upstream file visibly changes, which means the
            // graph moved on since that run.
            if (live?.filename && live.filename !== clip._runLive) {
                resetPixels(clip);
                clip.source = { kind: "live", filename: live.filename, label: live.label };
                fetchInfo(clip);
            } else if (live?.label) {
                clip.source.label = live.label;
            }
            return;
        }
        const filename = live?.filename ?? null;
        if (filename !== (clip.source?.filename ?? null)) {
            resetPixels(clip);
            // The upstream file changed, so the saved length is suspect —
            // let the next info fetch overwrite it even on a placed clip.
            clip._framesStale = true;
            clip.source = filename
                ? { kind: "live", filename, label: live.label }
                : (live?.label ? { kind: "live", filename: null, label: live.label } : null);
            if (filename) fetchInfo(clip);
        } else if (clip.source && live?.label) {
            clip.source.label = live.label;
        }
    }

    async function fetchInfo(clip) {
        const filename = clipFilename(clip);
        if (!filename || clip.infoState === "loading") return;
        clip.infoState = "loading";
        try {
            const resp = await api.fetchApi(
                `/ccn/video_scrubber/info?filename=${encodeURIComponent(filename)}`);
            if (!resp.ok) {
                clip.infoState = "failed";
                scheduleLower();
                return;
            }
            const info = await resp.json();
            clip.info = { width: info.width, height: info.height, fps: info.fps };
            clip.infoState = "ok";
            const frames = Math.max(0, Math.round(info.total_frames || 0));
            // Note: for live sources this is the FILE's frame count, which can
            // exceed the IMAGE batch actually reaching the node (upstream
            // caps/skips); the first run's payload corrects it.
            if (frames > 0 && (!clip.placed || clip._framesStale)) {
                clip.frames = frames;
                clip._framesStale = false;
                fixSameTrackOverlaps();
            }
            placePending();
            updateNameLabel();
            sync();
        } catch (e) {
            console.warn("[CCN SimpleTrack] info fetch:", e);
            clip.infoState = "failed";
            scheduleLower();
        }
    }

    /* ── Frame fetching (latest-wins per clip, video_scrubber style) ── */
    function frameUrl(filename, idx) {
        return `/ccn/video_scrubber/frame` +
               `?filename=${encodeURIComponent(filename)}&frame=${idx}`;
    }

    function cachedBitmap(clip, f) {
        const store = clip.frameStore;
        const hit = store.cache.get(f);
        if (hit) {
            store.cache.delete(f);
            store.cache.set(f, hit); // LRU bump
            return hit;
        }
        return null;
    }

    async function requestFrame(clip, f) {
        const store = clip.frameStore;
        const filename = clipFilename(clip);
        if (!filename || store.cache.has(f)) return;
        if (store.inFlight) {
            store.queued = f;
            return;
        }
        store.inFlight = true;
        const token = clip._stripToken;
        try {
            const resp = await api.fetchApi(frameUrl(filename, f));
            if (resp.ok) {
                const blob = await resp.blob();
                const bmp = await decodeBlob(blob);
                if (token !== clip._stripToken) {
                    releaseFrame(bmp); // clip pixels were invalidated mid-fetch
                } else {
                    store.cache.set(f, bmp);
                    while (store.cache.size > FRAME_CACHE_CAP) {
                        const oldest = store.cache.keys().next().value;
                        releaseFrame(store.cache.get(oldest));
                        store.cache.delete(oldest);
                    }
                }
            } else if (resp.status === 404 && clip.source?.kind === "materialized") {
                // Preview swept or server restarted — re-run to materialize.
                clip.infoState = "failed";
            }
        } catch (e) {
            console.warn("[CCN SimpleTrack] frame fetch:", e);
        } finally {
            store.inFlight = false;
            if (store.queued != null) {
                const next = store.queued;
                store.queued = null;
                requestFrame(clip, next);
            }
            tryComposite();
        }
    }

    /* ── Preview compositing (latest-wins across clips) ─────── */
    let pendingT = null;

    function requestComposite(t) {
        pendingT = t;
        tryComposite();
    }

    function debouncedComposite(t) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => requestComposite(t), DEBOUNCE_MS);
    }

    function tryComposite() {
        if (pendingT == null) return;
        const t = pendingT;
        const cov = coverageAt(t);
        const needs = [];
        if (cov.a) needs.push([cov.a, localFrame(cov.a, t)]);
        if (cov.b) needs.push([cov.b, localFrame(cov.b, t)]);
        let ready = true;
        for (const [clip, f] of needs) {
            if (!cachedBitmap(clip, f)) {
                if (clipFilename(clip)) {
                    ready = false;
                    requestFrame(clip, f);
                }
                // No filename: draw the placeholder, don't block on it.
            }
        }
        if (ready) drawComposite(t, cov);
    }

    function drawContained(ctx, bmp, rect) {
        const r = containRect(bmp.width, bmp.height, rect.x, rect.y, rect.w, rect.h);
        ctx.drawImage(bmp, r.x, r.y, r.w, r.h);
    }

    function drawComposite(t, cov) {
        const ctx = previewCanvas.getContext("2d");
        const cw = previewCanvas.width, ch = previewCanvas.height;
        if (!cw || !ch) return;
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, cw, ch);

        const [ow, oh] = outputDims();
        const rect = containRect(ow, oh, 0, 0, cw, ch);

        const centerText = (msg) => {
            ctx.fillStyle = C.label;
            ctx.font = "12px monospace";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(msg, cw / 2, ch / 2);
        };

        if (!cov.a) {
            centerText(placedClips().length ? "no clip at playhead" : "connect clips to begin");
            return;
        }

        const bmpA = cachedBitmap(cov.a, localFrame(cov.a, t));
        if (bmpA) {
            // Clip frames are drawn aspect-contained inside the output frame,
            // mirroring the Python fit-pad (letterbox) resize.
            drawContained(ctx, bmpA, rect);
        } else {
            centerText(`clip ${cov.a.key} — run to load pixels`);
        }

        if (cov.b) {
            const bmpB = cachedBitmap(cov.b, localFrame(cov.b, t));
            if (bmpB) {
                // Composite B's full output frame (pixels + letterbox bars) as
                // ONE layer at alpha w, so the result is exactly
                // (1-w)*A + w*B everywhere — including inside the bars.
                offscreen.width = Math.max(1, Math.round(rect.w));
                offscreen.height = Math.max(1, Math.round(rect.h));
                const octx = offscreen.getContext("2d");
                octx.fillStyle = "#000";
                octx.fillRect(0, 0, offscreen.width, offscreen.height);
                drawContained(octx, bmpB,
                    { x: 0, y: 0, w: offscreen.width, h: offscreen.height });
                ctx.globalAlpha = cov.w;
                ctx.drawImage(offscreen, rect.x, rect.y, rect.w, rect.h);
                ctx.globalAlpha = 1;
            }
        }

        ctx.strokeStyle = "#222";
        ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w - 1, rect.h - 1);
    }

    /* ── Thumbnail strips ───────────────────────────────────── */
    const thumbQueue = { active: 0, q: [] };

    function enqueueThumb(task) {
        thumbQueue.q.push(task);
        pumpThumbs();
    }

    function pumpThumbs() {
        while (thumbQueue.active < THUMB_CONCURRENCY && thumbQueue.q.length) {
            const task = thumbQueue.q.shift();
            thumbQueue.active++;
            task().finally(() => {
                thumbQueue.active--;
                pumpThumbs();
            });
        }
    }

    function maybeBakeStrip(clip, ppf) {
        const filename = clipFilename(clip);
        if (!filename || clip.frames <= 0 || clip._stripBaking) return;
        const aspect = clip.info?.width && clip.info?.height
            ? clip.info.width / clip.info.height : 16 / 9;
        const thumbW = Math.max(12, Math.round(TRACK_H * aspect));
        const stripPx = clip.frames * ppf;
        const n = clamp(Math.round(stripPx / thumbW) || 1, 1, MAX_THUMBS);
        if (clip.strip &&
            clip.strip.filename === filename &&
            clip.strip.frames === clip.frames &&
            Math.abs(clip.strip.n - n) < 2) return; // rebake hysteresis

        clip._stripBaking = true;
        const token = ++clip._stripToken;
        const canvas = document.createElement("canvas");
        canvas.width = n * thumbW;
        canvas.height = TRACK_H;
        const sctx = canvas.getContext("2d");
        sctx.fillStyle = "#101018";
        sctx.fillRect(0, 0, canvas.width, canvas.height);
        clip.strip = { canvas, filename, n, frames: clip.frames };

        let remaining = n;
        for (let i = 0; i < n; i++) {
            const f = Math.floor(i * (clip.frames - 1) / Math.max(n - 1, 1));
            const x = i * thumbW;
            enqueueThumb(async () => {
                if (token !== clip._stripToken) { remaining--; return; }
                try {
                    const resp = await api.fetchApi(frameUrl(filename, f));
                    if (resp.ok && token === clip._stripToken) {
                        const bmp = await decodeBlob(await resp.blob());
                        if (token === clip._stripToken) {
                            const r = containRect(bmp.width, bmp.height, x, 0, thumbW, TRACK_H);
                            sctx.drawImage(bmp, r.x, r.y, r.w, r.h);
                            scheduleLower();
                        }
                        releaseFrame(bmp);
                    }
                } catch (_) { /* strip slot stays dark */ }
                if (--remaining <= 0 && token === clip._stripToken) {
                    clip._stripBaking = false;
                }
            });
        }
    }

    /* ── Lower canvas rendering ─────────────────────────────── */
    let rafPending = false;

    function scheduleLower() {
        lowerDirty = true;
        if (rafPending) return;
        rafPending = true;
        requestAnimationFrame(() => {
            rafPending = false;
            // Catch buffer/layout drift the ResizeObserver may have missed
            // (element attached late, DOM widget repositioned, etc.).
            syncSize();
            if (lowerDirty) {
                lowerDirty = false;
                drawLower();
            }
        });
    }

    function selectedClip() {
        return model.clips.find((c) => c.key === model.selectedKey) ?? null;
    }

    function drawLower() {
        const ctx = lowerCanvas.getContext("2d");
        const L = layout();
        const w = lowerCanvas.width, h = lowerCanvas.height;
        if (!w || !h) return;

        for (const clip of model.clips) refreshSource(clip);

        ctx.fillStyle = C.panel;
        ctx.fillRect(0, 0, w, h);

        drawRuler(ctx, L);
        drawTracks(ctx, L);
        drawPlayhead(ctx, L);
        drawCurveRegion(ctx, L);
    }

    function drawRuler(ctx, L) {
        const fps = fpsValue();
        ctx.fillStyle = C.bg;
        ctx.fillRect(0, L.ruler.y, L.w, L.ruler.h);
        const secPx = Math.max(fps * L.ppf, 1e-6);
        // Pick the smallest step giving ticks at least ~24px apart, so the
        // loop below is always bounded.
        const STEPS = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1800, 3600];
        const step = STEPS.find((s) => s * secPx >= 24) ?? 3600;
        ctx.strokeStyle = C.grid;
        ctx.fillStyle = C.label;
        ctx.font = "9px monospace";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        for (let s = 0; ; s += step) {
            const x = L.xFromT(s * fps);
            if (x > L.w) break;
            ctx.beginPath();
            ctx.moveTo(x + 0.5, L.ruler.y + 8);
            ctx.lineTo(x + 0.5, L.ruler.y + L.ruler.h);
            ctx.stroke();
            ctx.fillText(`${s}s`, x + 2, L.ruler.y + 1);
        }
    }

    function drawTracks(ctx, L) {
        for (const track of [0, 1]) {
            const r = L.tracks[track];
            ctx.fillStyle = C.trackBg[track];
            ctx.fillRect(0, r.y, L.w, r.h);
        }

        const sel = selectedClip();
        let placeholderX = L.xFromT(L.total) + 10;
        for (const clip of model.clips.slice().sort((a, b) => wireCmp(a.key, b.key))) {
            const r = L.tracks[clip.track === 1 ? 1 : 0];
            if (clip.placed && clip.frames > 0) {
                const x = L.xFromT(clip.start);
                const cw = clip.frames * L.ppf;
                maybeBakeStrip(clip, L.ppf);
                if (clip.strip) {
                    ctx.drawImage(clip.strip.canvas, x, r.y, cw, r.h);
                } else {
                    ctx.fillStyle = C.clipTint[clip.track === 1 ? 1 : 0];
                    ctx.fillRect(x, r.y, cw, r.h);
                    ctx.fillStyle = C.label;
                    ctx.font = "10px monospace";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    const msg = clipFilename(clip)
                        ? `clip ${clip.key}` : `clip ${clip.key} — run to load`;
                    ctx.fillText(msg, x + cw / 2, r.y + r.h / 2);
                }
                ctx.lineWidth = clip === sel ? 2.5 : 1.5;
                ctx.strokeStyle = clip === sel ? C.keySel : C.clipEdge;
                ctx.strokeRect(x + 0.5, r.y + 0.5, cw - 1, r.h - 1);
            } else {
                // Length still unknown: parked placeholder past the timeline.
                ctx.setLineDash([4, 3]);
                ctx.strokeStyle = clip === sel ? C.keySel : C.clipEdge;
                ctx.lineWidth = 1.5;
                ctx.strokeRect(placeholderX + 0.5, r.y + 0.5, PLACEHOLDER_W - 1, r.h - 1);
                ctx.setLineDash([]);
                ctx.fillStyle = C.label;
                ctx.font = "9px monospace";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                const msg = clip.infoState === "loading" ? "loading…" : "run to load";
                ctx.fillText(`clip ${clip.key}`, placeholderX + PLACEHOLDER_W / 2, r.y + r.h / 2 - 6);
                ctx.fillText(msg, placeholderX + PLACEHOLDER_W / 2, r.y + r.h / 2 + 6);
                clip._phX = placeholderX; // hit-test anchor for selection
                placeholderX += PLACEHOLDER_W + 6;
            }
        }

        // Snap guide during a clip drag.
        if (drag?.type === "clip" && drag.snapX != null) {
            ctx.strokeStyle = C.snapGuide;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(drag.snapX + 0.5, L.ruler.y);
            ctx.lineTo(drag.snapX + 0.5, L.tracksBottom);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    function drawPlayhead(ctx, L) {
        const x = L.xFromT(model.scrub + 0.5);
        ctx.strokeStyle = C.playhead;
        ctx.beginPath();
        ctx.moveTo(x + 0.5, L.ruler.y);
        ctx.lineTo(x + 0.5, L.tracksBottom);
        ctx.stroke();
        ctx.fillStyle = C.playhead;
        ctx.beginPath();
        ctx.moveTo(x - 4, L.ruler.y);
        ctx.lineTo(x + 4, L.ruler.y);
        ctx.lineTo(x, L.ruler.y + 6);
        ctx.closePath();
        ctx.fill();
    }

    /* Curve plot area within the curve rect. Gain domain is 0..1. */
    function curvePlot(L) {
        const r = L.curve;
        const l = 30, rt = L.w - 12;
        const t = r.y + 8, b = r.y + r.h - 14;
        return { l, r: rt, t, b, w: rt - l, h: b - t };
    }

    function c2p(cx, cy, a) {
        return [a.l + cx * a.w, a.b - cy * a.h];
    }

    function p2c(px, py, a) {
        return [(px - a.l) / a.w, (a.b - py) / a.h];
    }

    function handlePos(key, type, a) {
        const [kx, ky] = c2p(key.x, key.y, a);
        const slope = type === "out" ? key.out : key.in;
        const sign = type === "out" ? 1 : -1;
        const dx = sign * a.w;
        const dy = -slope * a.h * sign;
        const len = Math.hypot(dx, dy);
        if (len < 1e-6) return [kx + sign * HDL_LEN, ky];
        return [kx + (dx / len) * HDL_LEN, ky + (dy / len) * HDL_LEN];
    }

    function curveHitTest(px, py, keys, a) {
        const r2key = (KEY_R + 4) ** 2;
        const r2hdl = (HDL_R + 5) ** 2;
        for (let i = 0; i < keys.length; i++) {
            if (i > 0) {
                const [hx, hy] = handlePos(keys[i], "in", a);
                if ((px - hx) ** 2 + (py - hy) ** 2 <= r2hdl)
                    return { type: "in", index: i };
            }
            if (i < keys.length - 1) {
                const [hx, hy] = handlePos(keys[i], "out", a);
                if ((px - hx) ** 2 + (py - hy) ** 2 <= r2hdl)
                    return { type: "out", index: i };
            }
        }
        for (let i = 0; i < keys.length; i++) {
            const [kx, ky] = c2p(keys[i].x, keys[i].y, a);
            if ((px - kx) ** 2 + (py - ky) ** 2 <= r2key)
                return { type: "key", index: i };
        }
        return null;
    }

    function drawCurveRegion(ctx, L) {
        const r = L.curve;
        ctx.fillStyle = C.bg;
        ctx.beginPath();
        ctx.roundRect(2, r.y, L.w - 4, r.h - 2, 5);
        ctx.fill();

        const clip = selectedClip();
        if (!clip) {
            ctx.fillStyle = C.label;
            ctx.font = "11px monospace";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("select a clip to edit its volume curve",
                L.w / 2, r.y + r.h / 2);
            return;
        }

        const a = curvePlot(L);
        const keys = clip.curve;

        ctx.strokeStyle = C.grid;
        ctx.lineWidth = 0.5;
        for (let i = 1; i < 4; i++) {
            const gx = a.l + (i / 4) * a.w;
            const gy = a.b - (i / 4) * a.h;
            ctx.beginPath(); ctx.moveTo(gx, a.t); ctx.lineTo(gx, a.b); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(a.l, gy); ctx.lineTo(a.r, gy); ctx.stroke();
        }
        ctx.strokeStyle = C.gridEdge;
        ctx.lineWidth = 1;
        ctx.strokeRect(a.l, a.t, a.w, a.h);

        ctx.fillStyle = C.label;
        ctx.font = "9px monospace";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText("0", a.l - 4, a.b);
        ctx.fillText("1", a.l - 4, a.t);
        ctx.textAlign = "left";
        ctx.fillText("vol", 4, a.t + 4);

        ctx.strokeStyle = C.curve;
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        ctx.beginPath();
        const steps = Math.max(Math.round(a.w), 60);
        for (let s = 0; s <= steps; s++) {
            const t = s / steps;
            const yv = clamp(hermite(keys, t), 0, 1);
            const [px, py] = c2p(t, yv, a);
            s === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        }
        ctx.stroke();

        for (let i = 0; i < keys.length; i++) {
            const k = keys[i];
            const [kx, ky] = c2p(k.x, k.y, a);
            const drawHandle = (type) => {
                const [hx, hy] = handlePos(k, type, a);
                ctx.strokeStyle = C.hdlLine;
                ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(kx, ky); ctx.lineTo(hx, hy); ctx.stroke();
                ctx.fillStyle = C.hdl;
                ctx.beginPath(); ctx.arc(hx, hy, HDL_R, 0, Math.PI * 2); ctx.fill();
            };
            if (i > 0) drawHandle("in");
            if (i < keys.length - 1) drawHandle("out");

            const isBroken = k.mirrored === false;
            ctx.fillStyle = i === curveSelKey ? C.keySel
                          : isBroken ? C.keyBroken : C.key;
            ctx.strokeStyle = C.keyStroke;
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            if (isBroken) {
                ctx.moveTo(kx, ky - KEY_R - 1);
                ctx.lineTo(kx + KEY_R + 1, ky);
                ctx.lineTo(kx, ky + KEY_R + 1);
                ctx.lineTo(kx - KEY_R - 1, ky);
                ctx.closePath();
            } else {
                ctx.arc(kx, ky, KEY_R, 0, Math.PI * 2);
            }
            ctx.fill();
            ctx.stroke();
        }

        // Scrub-position marker when the playhead is inside this clip.
        if (clip.frames > 1 &&
            model.scrub >= clip.start && model.scrub < clip.start + clip.frames) {
            const tv = (model.scrub - clip.start) / (clip.frames - 1);
            const yv = clamp(hermite(keys, tv), 0, 1);
            const [mx, my] = c2p(clamp(tv, 0, 1), yv, a);
            ctx.fillStyle = C.marker;
            ctx.beginPath();
            ctx.arc(mx, my, 3.5, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    /* ── Selection / labels / slider ────────────────────────── */
    function updateNameLabel() {
        const clip = selectedClip();
        if (!clip) {
            nameLabel.textContent = "";
            return;
        }
        const label = clip.source?.label ?? null;
        const dim = clip.info?.width
            ? ` · ${clip.info.width}×${clip.info.height}` : "";
        const audio = lastPayload?.clips?.find(
            (p) => String(p.key) === clip.key)?.has_audio;
        nameLabel.textContent =
            `clip ${clip.key}` + (label ? ` — ${label}` : "") +
            ` · ${clip.frames || "?"} frames${dim}` +
            (audio === false ? " · no audio" : "");
    }

    function selectClip(key) {
        if (model.selectedKey === key) return;
        model.selectedKey = key;
        curveSelKey = -1;
        // Properties persist with the workflow but are invisible to the
        // execution cache — selection never causes a re-run.
        node.properties = node.properties ?? {};
        node.properties.ccn_selected = key;
        updateNameLabel();
        scheduleLower();
    }

    function updateSliderRange() {
        const total = timelineTotal();
        slider.max = String(Math.max(0, total - 1));
        if (model.scrub > total - 1) model.scrub = Math.max(0, total - 1);
        slider.value = String(model.scrub);
        const secs = (model.scrub / fpsValue()).toFixed(2);
        frameLabel.textContent = `${model.scrub} / ${total} · ${secs}s`;
    }

    function setScrub(t, immediate = false) {
        const total = timelineTotal();
        model.scrub = clamp(Math.round(t), 0, Math.max(0, total - 1));
        updateSliderRange();
        scheduleLower();
        if (immediate) requestComposite(model.scrub);
        else debouncedComposite(model.scrub);
    }

    /* ── Pointer interaction on the lower canvas ────────────── */
    function canvasPos(e) {
        const r = lowerCanvas.getBoundingClientRect();
        return [
            (e.clientX - r.left) * (lowerCanvas.width / Math.max(r.width, 1)),
            (e.clientY - r.top) * (lowerCanvas.height / Math.max(r.height, 1)),
        ];
    }

    function clipAt(L, x, y) {
        for (const clip of model.clips) {
            const r = L.tracks[clip.track === 1 ? 1 : 0];
            if (y < r.y || y > r.y + r.h) continue;
            if (clip.placed && clip.frames > 0) {
                const cx = L.xFromT(clip.start);
                if (x >= cx && x <= cx + clip.frames * L.ppf) return clip;
            } else if (clip._phX != null &&
                       x >= clip._phX && x <= clip._phX + PLACEHOLDER_W) {
                return clip;
            }
        }
        return null;
    }

    /* Document-level safety net for missed pointer releases (alt-tab,
       release outside the window) — ccn_curve_widget pattern. */
    function forceRelease() {
        if (drag?.moved && (drag.type === "clip" || drag.type === "curve")) sync();
        drag = null;
        document.removeEventListener("pointerup", onDocPointerUp, true);
        document.removeEventListener("pointercancel", onDocPointerUp, true);
        window.removeEventListener("blur", onDocPointerUp);
        scheduleLower();
    }

    function onDocPointerUp() {
        if (drag) forceRelease();
    }

    function installDragSafety() {
        document.addEventListener("pointerup", onDocPointerUp, true);
        document.addEventListener("pointercancel", onDocPointerUp, true);
        window.addEventListener("blur", onDocPointerUp);
    }

    lowerCanvas.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        const [x, y] = canvasPos(e);
        const L = layout();

        if (y <= L.ruler.y + L.ruler.h) {
            drag = { type: "scrub" };
            installDragSafety();
            lowerCanvas.setPointerCapture(e.pointerId);
            setScrub(L.tFromX(x), true);
            return;
        }

        if (y <= L.tracksBottom) {
            const clip = clipAt(L, x, y);
            if (clip) {
                selectClip(clip.key);
                if (clip.placed && clip.frames > 0) {
                    drag = {
                        type: "clip", clip,
                        grabOffset: L.tFromX(x) - clip.start,
                        left: null, right: null,
                        startStart: clip.start,
                        moved: false, snapX: null,
                    };
                    // Stable neighbors from the pre-drag order: the clip hard
                    // stops against them, no pass-through reordering.
                    for (const o of placedClips()) {
                        if (o === clip || o.track !== clip.track) continue;
                        if (o.start < clip.start &&
                            (!drag.left || o.start > drag.left.start)) drag.left = o;
                        if (o.start > clip.start &&
                            (!drag.right || o.start < drag.right.start)) drag.right = o;
                    }
                    installDragSafety();
                    lowerCanvas.setPointerCapture(e.pointerId);
                }
            } else {
                selectClip(null);
            }
            scheduleLower();
            return;
        }

        // Curve region.
        const clip = selectedClip();
        if (!clip) return;
        const a = curvePlot(L);
        const hit = curveHitTest(x, y, clip.curve, a);
        if (hit) {
            drag = { type: "curve", clip, hit, moved: false };
            curveSelKey = hit.index;
            installDragSafety();
            lowerCanvas.setPointerCapture(e.pointerId);
            scheduleLower();
        } else if (x >= a.l && x <= a.r && y >= a.t && y <= a.b) {
            curveSelKey = -1;
            scheduleLower();
        }
    });

    lowerCanvas.addEventListener("pointermove", (e) => {
        if (!drag) return;
        const [x, y] = canvasPos(e);
        const L = layout();

        if (drag.type === "scrub") {
            setScrub(L.tFromX(x), false);
            return;
        }

        if (drag.type === "clip") {
            const clip = drag.clip;
            let cand = Math.round(L.tFromX(x) - drag.grabOffset);
            drag.snapX = null;

            // Snap to timeline 0 and opposite-track clip edges (either of this
            // clip's own edges may engage); clamping below wins over snapping.
            const targets = [0];
            for (const o of placedClips()) {
                if (o === clip || o.track === clip.track) continue;
                targets.push(o.start, o.start + o.frames);
            }
            let bestDelta = SNAP_PX + 1;
            let bestStart = null, bestX = null;
            for (const g of targets) {
                for (const [snapStart, edgeT] of [[g, cand], [g - clip.frames, cand + clip.frames]]) {
                    const d = Math.abs(L.xFromT(edgeT) - L.xFromT(g));
                    if (d <= SNAP_PX && d < bestDelta) {
                        bestDelta = d;
                        bestStart = snapStart;
                        bestX = L.xFromT(g);
                    }
                }
            }
            if (bestStart != null) {
                cand = bestStart;
                drag.snapX = bestX;
            }

            const minS = Math.max(0, drag.left ? drag.left.start + drag.left.frames : 0);
            const maxS = drag.right ? drag.right.start - clip.frames : Infinity;
            cand = clamp(cand, minS, Math.max(maxS, minS));
            if (cand !== clip.start) {
                clip.start = cand;
                drag.moved = cand !== drag.startStart;
                updateSliderRange();
                debouncedComposite(model.scrub);
            }
            scheduleLower();
            return;
        }

        if (drag.type === "curve") {
            const a = curvePlot(L);
            const keys = drag.clip.curve;
            const k = keys[drag.hit.index];
            if (!k) return;
            drag.moved = true;

            if (drag.hit.type === "key") {
                const isFirst = drag.hit.index === 0;
                const isLast = drag.hit.index === keys.length - 1;
                const [cx, cy] = p2c(x, y, a);
                const gy = clamp(cy, 0, 1); // gain domain
                if (isFirst || isLast) {
                    k.y = gy;
                } else {
                    const lo = keys[drag.hit.index - 1].x + MIN_KEY_DX;
                    const hi = keys[drag.hit.index + 1].x - MIN_KEY_DX;
                    k.x = clamp(cx, lo, hi);
                    k.y = gy;
                }
            } else {
                const [kpx, kpy] = c2p(k.x, k.y, a);
                const dx = x - kpx;
                const dy = y - kpy;
                if (e.shiftKey && k.mirrored !== false) k.mirrored = false;
                let newSlope = null;
                if (drag.hit.type === "out" && dx > 1) {
                    newSlope = -dy * a.w / (dx * a.h);
                    k.out = newSlope;
                } else if (drag.hit.type === "in" && dx < -1) {
                    newSlope = -dy * a.w / (dx * a.h);
                    k.in = newSlope;
                }
                if (newSlope !== null && k.mirrored !== false) {
                    k.in = newSlope;
                    k.out = newSlope;
                }
            }
            scheduleLower();
        }
    });

    lowerCanvas.addEventListener("pointerup", () => {
        if (!drag) return;
        forceRelease();
    });

    lowerCanvas.addEventListener("dblclick", (e) => {
        const [x, y] = canvasPos(e);
        const L = layout();
        const clip = selectedClip();
        if (!clip || y < L.curve.y) return;
        const a = curvePlot(L);
        const keys = clip.curve;
        const hit = curveHitTest(x, y, keys, a);
        if (hit?.type === "key" &&
            hit.index > 0 && hit.index < keys.length - 1) {
            keys.splice(hit.index, 1);
            curveSelKey = -1;
            sync();
            return;
        }
        if (!hit && x >= a.l && x <= a.r && y >= a.t && y <= a.b) {
            const [cx] = p2c(x, y, a);
            if (cx > MIN_KEY_DX && cx < 1 - MIN_KEY_DX &&
                !keys.some((k) => Math.abs(k.x - cx) < MIN_KEY_DX)) {
                const evalY = clamp(hermite(keys, cx), 0, 1);
                keys.push({ x: cx, y: evalY, in: 0, out: 0, mirrored: true });
                keys.sort((ka, kb) => ka.x - kb.x);
                const ni = keys.findIndex((k) => Math.abs(k.x - cx) < 0.001);
                if (ni > 0 && ni < keys.length - 1) {
                    const slope = (keys[ni + 1].y - keys[ni - 1].y)
                                / (keys[ni + 1].x - keys[ni - 1].x);
                    keys[ni].in = slope;
                    keys[ni].out = slope;
                }
                curveSelKey = ni;
                sync();
            }
        }
    });

    lowerCanvas.addEventListener("contextmenu", (e) => {
        // Always swallow the browser menu on the timeline surface.
        e.preventDefault();
        e.stopPropagation();
        const [, y] = canvasPos(e);
        const L = layout();
        const clip = selectedClip();
        if (!clip || y < L.curve.y) return;
        const items = [{
            content: "Reset Volume Curve",
            callback: () => {
                clip.curve = JSON.parse(JSON.stringify(DEFAULT_CLIP_CURVE));
                curveSelKey = -1;
                sync();
            },
        }];
        const keys = clip.curve;
        if (curveSelKey > 0 && curveSelKey < keys.length - 1) {
            const sk = keys[curveSelKey];
            items.push({
                content: `Delete Curve Key ${curveSelKey}`,
                callback: () => {
                    keys.splice(curveSelKey, 1);
                    curveSelKey = -1;
                    sync();
                },
            });
            items.push(sk.mirrored !== false
                ? {
                    content: "Break Tangents",
                    callback: () => { sk.mirrored = false; sync(); },
                }
                : {
                    content: "Mirror Tangents",
                    callback: () => { sk.mirrored = true; sk.in = sk.out; sync(); },
                });
        }
        new LiteGraph.ContextMenu(items, { event: e });
    });

    /* ── Slider / step buttons ──────────────────────────────── */
    slider.addEventListener("input", () => {
        setScrub(parseInt(slider.value, 10) || 0, false);
    });
    prevBtn.addEventListener("click", () => setScrub(model.scrub - 1, true));
    nextBtn.addEventListener("click", () => setScrub(model.scrub + 1, true));

    /* ── Dynamic input pairs ────────────────────────────────── */
    function ensurePairs(count) {
        let changed = false;
        for (let i = 1; i <= count; i++) {
            if (!node.inputs?.some((inp) => inp.name === `video_${i}`)) {
                node.addInput(`video_${i}`, "IMAGE");
                changed = true;
            }
            if (!node.inputs?.some((inp) => inp.name === `audio_${i}`)) {
                node.addInput(`audio_${i}`, "AUDIO");
                changed = true;
            }
        }
        return changed;
    }

    function maxConnectedPair() {
        let max = 0;
        for (const inp of node.inputs ?? []) {
            const m = /^(?:video|audio)_(\d+)$/.exec(inp.name);
            if (m && inp.link != null) max = Math.max(max, Number(m[1]));
        }
        return max;
    }

    function reconcilePairs() {
        const want = maxConnectedPair() + 1;
        let changed = ensurePairs(want);
        // Prune trailing fully-unlinked pairs beyond the first empty one.
        for (let i = (node.inputs?.length ?? 0) - 1; i >= 0; i--) {
            const m = /^(?:video|audio)_(\d+)$/.exec(node.inputs[i].name);
            if (m && Number(m[1]) > want && node.inputs[i].link == null) {
                node.removeInput(i);
                changed = true;
            }
        }
        if (changed) {
            // computeSize would collapse the deliberately-wide timeline;
            // grow to fit the new sockets but never shrink what the user has.
            const [w, h] = node.size;
            node.setSize(node.computeSize());
            node.setSize([Math.max(node.size[0], w), Math.max(node.size[1], h)]);
        }
    }

    function connectedVideoKeys() {
        const keys = [];
        for (const inp of node.inputs ?? []) {
            const m = /^video_(\d+)$/.exec(inp.name);
            if (m && inp.link != null) keys.push(m[1]);
        }
        keys.sort(wireCmp);
        return keys;
    }

    function reconcileClips() {
        const connected = connectedVideoKeys();
        let changed = false;
        for (const key of connected) {
            if (!model.clips.some((c) => c.key === key)) {
                const clip = makeClip(key);
                model.clips.push(clip);
                changed = true;
            }
        }
        for (let i = model.clips.length - 1; i >= 0; i--) {
            if (!connected.includes(model.clips[i].key)) {
                destroyClip(model.clips[i]);
                model.clips.splice(i, 1);
                changed = true;
            }
        }
        model.clips.sort((a, b) => wireCmp(a.key, b.key));
        if (changed && model.selectedKey != null &&
            !connected.includes(model.selectedKey)) {
            selectClip(null);
        }
        for (const clip of model.clips) refreshSource(clip);
        for (const clip of model.clips) {
            if (clip.frames <= 0 && clipFilename(clip) && clip.infoState === "idle") {
                fetchInfo(clip);
            }
        }
        placePending();
        if (changed) sync();
        updateNameLabel();
    }

    /* ── Widget registration ────────────────────────────────── */
    // The DOM widget IS the track_data widget: same name, getValue/setValue
    // round-trip the JSON, and it is spliced into the STRING widget's slot so
    // widgets_values line up by both name and index across frontends.
    const stringIdx = node.widgets.findIndex((w) => w.name === "track_data");
    if (stringIdx >= 0) node.widgets.splice(stringIdx, 1);
    const domWidget = node.addDOMWidget("track_data", "custom", container, {
        getValue,
        setValue,
        getMinHeight: () => MIN_UI_H,
    });
    const domIdx = node.widgets.indexOf(domWidget);
    if (stringIdx >= 0 && domIdx >= 0 && domIdx !== stringIdx) {
        node.widgets.splice(domIdx, 1);
        node.widgets.splice(stringIdx, 0, domWidget);
    }

    // Repaint on the widgets that change the preview or timeline scale.
    for (const name of ["width", "height", "fps"]) {
        const w = node.widgets.find((x) => x.name === name);
        if (!w) continue;
        const origCb = w.callback;
        w.callback = function (...args) {
            origCb?.apply(this, args);
            updateSliderRange();
            scheduleLower();
            requestComposite(model.scrub);
        };
    }

    /* One-way size sync (layout -> buffer) so the canvases can't drive
       resize creep; image_inset pattern. */
    function syncSize() {
        let changed = false;
        for (const cvs of [previewCanvas, lowerCanvas]) {
            const w = Math.round(cvs.clientWidth);
            const h = Math.round(cvs.clientHeight);
            if (w > 0 && h > 0 && (cvs.width !== w || cvs.height !== h)) {
                cvs.width = w;
                cvs.height = h;
                changed = true;
            }
        }
        if (changed) {
            drawLower();
            tryComposite();
        }
    }

    const observer = new ResizeObserver(syncSize);
    observer.observe(previewCanvas);
    observer.observe(lowerCanvas);

    /* ── Surface for the prototype hooks ────────────────────── */
    return {
        syncSize,

        onConnectionsChange() {
            // Deferred: never mutate node.inputs inside LiteGraph's
            // connection callback.
            setTimeout(() => {
                reconcilePairs();
                reconcileClips();
            }, 0);
        },

        beforeConfigure(info) {
            // Sockets must exist before links restore (lora_pair_loader
            // pattern); scan the serialized inputs for the pair count.
            let maxIdx = 1;
            for (const inp of info?.inputs ?? []) {
                const m = /^(?:video|audio)_(\d+)$/.exec(inp?.name ?? "");
                if (m) maxIdx = Math.max(maxIdx, Number(m[1]));
            }
            ensurePairs(maxIdx);
        },

        afterConfigure() {
            reconcilePairs();
            reconcileClips();
            const sel = node.properties?.ccn_selected;
            if (sel != null && model.clips.some((c) => c.key === sel)) {
                model.selectedKey = sel;
            }
            requestAnimationFrame(() => {
                syncSize();
                for (const clip of model.clips) {
                    refreshSource(clip);
                    if (clipFilename(clip) && clip.infoState === "idle") {
                        fetchInfo(clip);
                    }
                }
                updateNameLabel();
                updateSliderRange();
                scheduleLower();
                requestComposite(model.scrub);
            });
        },

        onExecutedPayload(data) {
            if (!data) return;
            lastPayload = data;
            for (const pc of data.clips ?? []) {
                const clip = model.clips.find((c) => c.key === String(pc.key));
                if (!clip) continue; // input disconnected since the run
                if (pc.frames > 0 && pc.frames !== clip.frames) {
                    clip.frames = pc.frames;
                }
                if (!clip.placed) {
                    // The backend auto-placed it with the shared formula;
                    // adopt its result.
                    clip.start = pc.start;
                    clip.track = pc.track === 1 ? 1 : 0;
                    clip.placed = true;
                }
                if (!clip.info) {
                    clip.info = { width: pc.source_width, height: pc.source_height };
                }
                if (pc.preview?.filename) {
                    const filename =
                        (pc.preview.subfolder ? pc.preview.subfolder + "/" : "") +
                        pc.preview.filename + ` [${pc.preview.type || "temp"}]`;
                    if (clip.source?.kind !== "materialized" ||
                        clip.source.filename !== filename) {
                        const label = clip.source?.label;
                        resetPixels(clip);
                        clip.source = { kind: "materialized", filename, label };
                        clip.infoState = "ok";
                        clip.info = { width: pc.source_width, height: pc.source_height };
                        clip._runLive = discoverVideoSource(
                            node, `video_${clip.key}`)?.filename ?? null;
                    }
                }
            }
            fixSameTrackOverlaps();
            updateNameLabel();
            sync();
        },

        destroy() {
            observer.disconnect();
            clearTimeout(debounceTimer);
            forceRelease();
            for (const clip of model.clips) destroyClip(clip);
            thumbQueue.q.length = 0;
        },
    };
}

/* ── Extension registration ─────────────────────────────────── */
app.registerExtension({
    name: "CCN.SimpleTrack",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);
            this._ccnTrackUI = createTrackUI(this);
            this._ccnTrackUI.onConnectionsChange(); // creates the first pair
            // Fresh creation: LiteGraph applies its own computeSize AFTER
            // this hook, so claim the wide timeline footprint a frame later
            // (ccn_curve_widget pattern). A workflow load sets _ccnRestored
            // before the frame fires and keeps its saved size.
            requestAnimationFrame(() => {
                if (this._ccnRestored) return;
                this.setSize([
                    Math.max(this.size[0] || 0, MIN_NODE_W),
                    Math.max(this.size[1] || 0, this.computeSize()[1], MIN_UI_H + 150),
                ]);
                this._ccnTrackUI?.syncSize();
                this.setDirtyCanvas(true, true);
            });
        };

        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            this._ccnRestored = true;
            this._ccnTrackUI?.beforeConfigure(info);
            const result = origConfigure?.apply(this, arguments);
            this._ccnTrackUI?.afterConfigure(info);
            return result;
        };

        const origConnChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            origConnChange?.apply(this, arguments);
            this._ccnTrackUI?.onConnectionsChange();
        };

        const origExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origExecuted?.apply(this, arguments);
            this._ccnTrackUI?.onExecutedPayload(message?.ccn_simple_track?.[0]);
        };

        const origResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            origResize?.apply(this, arguments);
            this._ccnTrackUI?.syncSize();
        };

        const origRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this._ccnTrackUI?.destroy();
            this._ccnTrackUI = null;
            origRemoved?.apply(this, arguments);
        };
    },
});
