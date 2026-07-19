import { app } from "../../scripts/app.js";

/* ───────────────────────────────────────────────────────────────
   CCN MoE Cockpit — canvas widget for CCN_MoESamplerDual.

   Draws the sigma schedule with the Wan 2.2 expert boundary and
   per-phase CFG lanes, live, BEFORE execution, by walking upstream
   widget values:

     sigmas input → recognized scheduler node (BasicScheduler with
     scheduler ∈ {simple, normal, sgm_uniform, beta}) → its model
     input → through pass-through model nodes → ModelSamplingSD3
     (shift) or a terminal loader (shift 8.0, marked "assumed").

     curve_high / curve_low inputs → CurveDefinition curve_data
     widget (same read as ccn_curve_widget's syncExternalCurve).

   After each run the Python node returns the authoritative schedule
   in message.ccn_moe; the widget overlays it and shows a divergence
   badge if the live preview disagrees (>1e-3 on any sigma). Priority
   of what is drawn: live walk > last executed > placeholder text.

   Interactions:
     • Drag the boundary line — switches boundary to "custom" and
       writes custom_boundary on the node.

   The schedule math mirrors ComfyUI's flow-matching schedulers and
   is trued up by the executed overlay; the divergence badge is the
   tripwire if core's implementations drift from these ports.
   ─────────────────────────────────────────────────────────────── */

const MIN_HEIGHT = 220;
const PAD_X = 34;
const PAD_R = 40;
const PAD_Y = 16;
const PAD_B = 24;
const SHIFT_DEFAULT = 8.0; // Wan model-config default when no ModelSamplingSD3 found
const WALK_DEPTH_MAX = 24;
const DIVERGENCE_EPS = 1e-3;
const LANE_STRIP_H = 46;
const LANE_COLORS = ["#e07070", "#70e0a0", "#c080f0", "#f0d060", "#60c0f0", "#f09050"];

const C = {
    bg:        "#181825",
    grid:      "#27273a",
    gridEdge:  "#2e2e48",
    sigma:     "#4ec9b0",
    stepHigh:  "#e0a050",
    stepLow:   "#5090e0",
    highFill:  "#e0a05014",
    boundary:  "#e06070",
    executed:  "#8888aa",
    cfgHigh:   "#f0c050",
    cfgLow:    "#70a0f0",
    label:     "#6a6a80",
    text:      "#c0c0d0",
    warn:      "#e06070",
};

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

/* ── Link resolution (Map/array aware, muted/bypassed = unwired) ── */
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

    // Skip reroutes transparently.
    let depth = 0;
    while (src && src.type === "Reroute" && depth++ < WALK_DEPTH_MAX) {
        src = resolveInput(src, src.inputs?.[0]?.name ?? "");
    }
    return src ?? null;
}

function widgetValue(node, name) {
    return node?.widgets?.find((w) => w.name === name)?.value;
}

/* ── Shift discovery: walk the model chain upward ──────────────── */
function findShift(startNode) {
    let cur = startNode;
    for (let depth = 0; cur && depth < WALK_DEPTH_MAX; depth++) {
        if (cur.type === "ModelSamplingSD3") {
            const s = widgetValue(cur, "shift");
            if (typeof s === "number") return { shift: s, assumed: false };
        }
        // Hop through anything with a model-ish input; prefer the high chain
        // through our own pair loader.
        const next =
            resolveInput(cur, "model") ??
            resolveInput(cur, "model_high");
        if (!next) break;
        cur = next;
    }
    return { shift: SHIFT_DEFAULT, assumed: true };
}

/* ── Flow-matching schedule math (ports of comfy's schedulers) ─── */
function shifted(u, shift) {
    return (shift * u) / (1 + (shift - 1) * u);
}

// Regularized incomplete beta via Lentz continued fraction, inverted by
// bisection — used only for the "beta" scheduler preview.
function betacf(a, b, x) {
    const MAXIT = 200, EPS = 3e-12, FPMIN = 1e-300;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
        const m2 = 2 * m;
        let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
        d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
        c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
        d = 1 / d; h *= d * c;
        aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
        d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
        c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
        d = 1 / d;
        const del = d * c;
        h *= del;
        if (Math.abs(del - 1) < EPS) break;
    }
    return h;
}

function lgamma(z) {
    // Lanczos approximation
    const g = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
        -176.61502916214059, 12.507343278686905, -0.13857109526572012,
        9.9843695780195716e-6, 1.5056327351493116e-7];
    if (z < 0.5) {
        return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
    }
    z -= 1;
    let x = 0.99999999999980993;
    for (let i = 0; i < g.length; i++) x += g[i] / (z + i + 1);
    const t = z + g.length - 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function ibeta(a, b, x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
    const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lbeta);
    if (x < (a + 1) / (a + b + 2)) {
        return (front * betacf(a, b, x)) / a;
    }
    return 1 - (front * betacf(b, a, 1 - x)) / b;
}

function betaPpf(p, a, b) {
    if (p <= 0) return 0;
    if (p >= 1) return 1;
    let lo = 0, hi = 1;
    for (let i = 0; i < 80; i++) {
        const mid = (lo + hi) / 2;
        if (ibeta(a, b, mid) < p) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
}

function computeSigmas(scheduler, steps, shift) {
    const sigs = [];
    if (scheduler === "simple") {
        const ss = 1000 / steps;
        for (let x = 0; x < steps; x++) {
            const idx = 1000 - Math.trunc(x * ss); // table[-(1+int(x*ss))]
            sigs.push(shifted(idx / 1000, shift));
        }
        sigs.push(0.0);
    } else if (scheduler === "normal" || scheduler === "sgm_uniform") {
        const sigmaMax = shifted(1.0, shift);            // table end
        const sigmaMin = shifted(1 / 1000, shift);       // table start
        const start = sigmaMax * 1000;                    // timestep(sigma)
        const end = sigmaMin * 1000;
        const n = scheduler === "sgm_uniform" ? steps + 1 : steps;
        const pts = [];
        for (let i = 0; i < n; i++) {
            pts.push(start + ((end - start) * i) / (n - 1));
        }
        if (scheduler === "sgm_uniform") pts.pop();
        for (const ts of pts) sigs.push(shifted(ts / 1000, shift));
        sigs.push(0.0);
    } else if (scheduler === "beta") {
        const alpha = 0.6, beta = 0.6, total = 999;
        for (let i = 0; i < steps; i++) {
            const p = 1 - i / (steps - 1 || 1);
            const t = Math.round(betaPpf(p, alpha, beta) * total);
            sigs.push(shifted((t + 1) / 1000, shift));
        }
        sigs.push(0.0);
    } else {
        return null;
    }
    return sigs;
}

const SUPPORTED_SCHEDULERS = ["simple", "normal", "sgm_uniform", "beta"];

/* ── Upstream schedule discovery ───────────────────────────────── */
function discoverSchedule(node) {
    const sched = resolveInput(node, "sigmas");
    if (!sched) return { status: "unwired" };

    if (sched.type !== "BasicScheduler") {
        return { status: "unsupported", reason: `source: ${sched.type}` };
    }
    const scheduler = widgetValue(sched, "scheduler");
    const steps = widgetValue(sched, "steps");
    const denoise = widgetValue(sched, "denoise");
    if (!SUPPORTED_SCHEDULERS.includes(scheduler)) {
        return { status: "unsupported", reason: `scheduler: ${scheduler}` };
    }
    if (!(steps >= 1)) return { status: "unsupported", reason: "steps" };

    const model = resolveInput(sched, "model");
    const { shift, assumed } = model ? findShift(model)
                                     : { shift: SHIFT_DEFAULT, assumed: true };

    let effSteps = steps;
    let sigmas;
    if (typeof denoise === "number" && denoise < 1.0) {
        if (denoise <= 0.0) return { status: "unsupported", reason: "denoise 0" };
        const total = Math.trunc(steps / denoise);
        const full = computeSigmas(scheduler, total, shift);
        if (!full) return { status: "unsupported", reason: scheduler };
        sigmas = full.slice(-(steps + 1));
        effSteps = steps;
    } else {
        sigmas = computeSigmas(scheduler, steps, shift);
    }
    if (!sigmas) return { status: "unsupported", reason: scheduler };

    return {
        status: "ok", sigmas, scheduler, steps: effSteps, shift, assumed,
        caption: `${scheduler} · ${effSteps} steps · shift ${shift}${assumed ? " (assumed)" : ""}`,
    };
}

/* ── Curve lane discovery (syncExternalCurve pattern) ──────────── */
function discoverCurve(node, slotName) {
    const src = resolveInput(node, slotName);
    if (!src) return null;
    const w = src.widgets?.find((w) => w.name === "curve_data");
    if (!w) return { unreadable: true };
    try {
        const keys = JSON.parse(w.value);
        if (Array.isArray(keys) && keys.length >= 1) {
            return { keys: [...keys].sort((a, b) => a.x - b.x) };
        }
    } catch (_) { /* fall through */ }
    return { unreadable: true };
}

/* ── LoRA lane discovery + compilation (mirrors MoESamplerDual) ── */
function baseName(path) {
    const cut = path.lastIndexOf("/");
    const stem = cut === -1 ? path : path.slice(cut + 1);
    const dot = stem.lastIndexOf(".");
    return dot > 0 ? stem.slice(0, dot) : stem;
}

function walkLanes(node) {
    const lanes = [];
    let src = resolveInput(node, "lora_lanes");
    let depth = 0;
    while (src && depth++ < WALK_DEPTH_MAX) {
        const cls = src.comfyClass ?? src.type;
        if (cls !== "CCN_LoraPairLoader") break;
        const rows = (src.widgets ?? []).filter(
            (w) => w.type === "CCN_LORA_PAIR_ROW");
        const collected = [];
        for (const w of rows) {
            const v = w.value ?? {};
            if (v.mode !== "lane" || v.on === false ||
                !v.lora || v.lora === "None") continue;
            const suffix = w.name.slice(w.name.indexOf("_") + 1);
            const curve = discoverCurve(src, `curve_${suffix}`);
            collected.push({
                name: baseName(v.lora),
                sh: v.strength_high ?? 1,
                sl: v.strength_low ?? 1,
                keys: curve?.keys ?? null,
            });
        }
        lanes.unshift(...collected);
        src = resolveInput(src, "lanes");
    }
    return lanes;
}

function compileLanePhases(sigmas, switchIdx, segCount, lanes, mode) {
    const n = sigmas.length - 1;
    const gMax = sigmas[0], gMin = sigmas[sigmas.length - 1];
    const strengthsAt = (t) => lanes.map((ln) => {
        const m = ln.keys ? hermite(ln.keys, Math.max(0, Math.min(1, t))) : 1;
        return [Math.round(ln.sh * m * 1e4) / 1e4, Math.round(ln.sl * m * 1e4) / 1e4];
    });
    const compile = (phaseSig, phaseLen, offset) => {
        if (phaseLen <= 0) return [];
        const k = Math.max(1, Math.min(segCount, phaseLen));
        const bounds = [...new Set(
            Array.from({ length: k + 1 }, (_, i) => Math.round((i * phaseLen) / k))
        )].sort((a, b) => a - b);
        if (bounds[0] !== 0) bounds.unshift(0);
        if (bounds[bounds.length - 1] !== phaseLen) bounds.push(phaseLen);
        const segs = [];
        for (let i = 0; i < bounds.length - 1; i++) {
            const a = bounds[i], b = bounds[i + 1];
            if (b <= a) continue;
            let t;
            if (mode === "sigma") {
                const mid = phaseSig[Math.floor((a + b) / 2)];
                const rng = gMax - gMin;
                t = rng > 1e-10 ? (gMax - mid) / rng : 0;
            } else {
                t = (offset + (a + b) / 2) / Math.max(n, 1);
            }
            segs.push({ start: a, end: b, strengths: strengthsAt(t) });
        }
        const merged = [segs[0]];
        for (const s of segs.slice(1)) {
            if (JSON.stringify(s.strengths) ===
                JSON.stringify(merged[merged.length - 1].strengths)) {
                merged[merged.length - 1].end = s.end;
            } else {
                merged.push(s);
            }
        }
        return merged;
    };
    const k = switchIdx === -1 ? n : switchIdx;
    return {
        names: lanes.map((l) => l.name),
        high: compile(sigmas.slice(0, k + 1), k, 0),
        low: compile(sigmas.slice(k), n - k, k),
    };
}

function drawLaneStrip(ctx, strip, xl, xr, X, laneData, switchIdx, n) {
    ctx.fillStyle = C.bg;
    ctx.fillRect(xl, strip.top, xr - xl, strip.bottom - strip.top);
    ctx.strokeStyle = C.gridEdge;
    ctx.strokeRect(xl, strip.top, xr - xl, strip.bottom - strip.top);

    let maxAbs = 0.01;
    for (const phase of ["high", "low"]) {
        for (const seg of laneData[phase] ?? []) {
            for (const pair of seg.strengths) {
                maxAbs = Math.max(maxAbs, Math.abs(pair[0]), Math.abs(pair[1]));
            }
        }
    }
    const pad = 4;
    const Ys = (v) => strip.bottom - pad -
        ((strip.bottom - strip.top - 2 * pad) * (v + maxAbs)) / (2 * maxAbs);

    // zero line
    ctx.strokeStyle = C.grid;
    ctx.beginPath();
    ctx.moveTo(xl, Ys(0));
    ctx.lineTo(xr, Ys(0));
    ctx.stroke();

    const k = switchIdx === -1 ? n : switchIdx;
    laneData.names.forEach((name, i) => {
        ctx.strokeStyle = LANE_COLORS[i % LANE_COLORS.length];
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        let started = false;
        for (const [phase, offset, side] of [["high", 0, 0], ["low", k, 1]]) {
            for (const seg of laneData[phase] ?? []) {
                const v = seg.strengths[i]?.[side] ?? 0;
                const x0 = X(offset + seg.start), x1 = X(offset + seg.end);
                if (!started) { ctx.moveTo(x0, Ys(v)); started = true; }
                else ctx.lineTo(x0, Ys(v));
                ctx.lineTo(x1, Ys(v));
            }
        }
        ctx.stroke();
        ctx.fillStyle = LANE_COLORS[i % LANE_COLORS.length];
        ctx.font = "8px monospace";
        ctx.textAlign = "left";
        const label = name.length > 18 ? name.slice(0, 17) + "\u2026" : name;
        ctx.fillText(label, xl + 3 + i * 90, strip.top + 9);
    });
    ctx.fillStyle = C.label;
    ctx.font = "8px monospace";
    ctx.textAlign = "right";
    ctx.fillText(`\u00b1${maxAbs.toFixed(2)}`, xr - 2, strip.top + 9);
}

/* ── Per-phase CFG evaluation (mirrors CurveCFG semantics) ─────── */
function phaseCfgSeries(phaseSigmas, cfgMax, cfgMin, curve, mode, decay) {
    const n = phaseSigmas.length - 1;
    if (n < 1) return [];
    const sMax = phaseSigmas[0], sMin = phaseSigmas[phaseSigmas.length - 1];
    const out = [];
    for (let j = 0; j < n; j++) {
        const sigma = phaseSigmas[j];
        let t;
        if (mode === "step") {
            t = j / Math.max(n, 1);
        } else {
            const rng = sMax - sMin;
            t = rng > 1e-10 ? (sMax - sigma) / rng : 0;
        }
        let cfg;
        if (curve) {
            const blend = hermite(curve, Math.max(0, Math.min(1, t)));
            cfg = cfgMin + (cfgMax - cfgMin) * blend;
        } else {
            cfg = cfgMax;
        }
        if (decay && sMax > 1e-10) {
            cfg = 1 + (cfg - 1) * (sigma / sMax);
        }
        out.push(cfg);
    }
    return out;
}

/* ── Extension ─────────────────────────────────────────────────── */
app.registerExtension({
    name: "CCN.MoECockpit",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "CCN_MoESamplerDual") return;

        const origExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origExecuted?.apply(this, arguments);
            const data = message?.ccn_moe?.[0];
            if (data?.sigmas) {
                this._ccnExecuted = data;
                app.graph?.setDirtyCanvas?.(true, true);
            }
        };

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);
            const node = this;
            let currentHost = this;
            let dragBoundary = false;

            function boundaryValue() {
                const preset = widgetValue(node, "boundary");
                if (preset === "t2v (0.875)") return 0.875;
                if (preset === "i2v (0.900)") return 0.900;
                return widgetValue(node, "custom_boundary") ?? 0.875;
            }

            function setBoundary(v) {
                const bw = node.widgets.find((w) => w.name === "boundary");
                const cw = node.widgets.find((w) => w.name === "custom_boundary");
                if (bw) bw.value = "custom";
                if (cw) cw.value = Math.max(0.0, Math.min(1.0, Math.round(v * 1000) / 1000));
                app.graph?.setDirtyCanvas?.(true, true);
            }

            function plot(a) {
                return {
                    l: PAD_X, r: currentHost.size[0] - PAD_R,
                    t: a.wy + PAD_Y, b: a.wy + a.h - PAD_B,
                };
            }

            function getArea() {
                const wy = widget.last_y ?? 0;
                const h = Math.max(MIN_HEIGHT, currentHost.size[1] - wy - 8);
                return { wy, h };
            }

            function stripNeeded() {
                return walkLanes(node).length > 0 ||
                    !!node._ccnExecuted?.lanes;
            }

            const widget = {
                type: "CCN_MOE_COCKPIT",
                name: "moe_cockpit",
                options: { serialize: false },
                value: "",

                computeSize(width) {
                    return [width ?? 300, MIN_HEIGHT];
                },

                draw(ctx, drawNode, w, y) {
                    currentHost = drawNode;
                    this.last_y = y;
                    const a = getArea();
                    const p = plot(a);
                    const bound = boundaryValue();

                    const lanesLive = walkLanes(node);
                    const lanesExecOnly = (!lanesLive.length &&
                        node._ccnExecuted?.lanes) ? node._ccnExecuted.lanes : null;
                    const strip = { active: false, top: 0, bottom: 0 };
                    if (lanesLive.length || lanesExecOnly) {
                        strip.active = true;
                        strip.bottom = p.b;
                        p.b -= LANE_STRIP_H + 6;
                        strip.top = p.b + 6;
                    }

                    ctx.save();
                    ctx.fillStyle = C.bg;
                    ctx.fillRect(4, a.wy + 2, w - 8, a.h - 4);

                    const live = discoverSchedule(node);
                    const exec = node._ccnExecuted;
                    let sigmas = null, caption = "", executedOverlay = null;

                    if (live.status === "ok") {
                        sigmas = live.sigmas;
                        caption = live.caption;
                        if (exec?.sigmas?.length) executedOverlay = exec.sigmas;
                    } else if (exec?.sigmas?.length) {
                        sigmas = exec.sigmas;
                        caption = "last executed schedule";
                    }

                    if (!sigmas) {
                        ctx.fillStyle = C.label;
                        ctx.font = "12px monospace";
                        ctx.textAlign = "center";
                        const msg = live.status === "unwired"
                            ? "wire sigmas to preview"
                            : `no live preview (${live.reason}) \u2014 run once to populate`;
                        ctx.fillText(msg, (p.l + p.r) / 2, (p.t + p.b) / 2);
                        ctx.restore();
                        return;
                    }

                    const n = sigmas.length - 1;
                    const X = (i) => p.l + ((p.r - p.l) * i) / Math.max(n, 1);
                    const Y = (s) => p.b - (p.b - p.t) * Math.max(0, Math.min(1, s));

                    // Grid + axes.
                    ctx.strokeStyle = C.grid;
                    ctx.lineWidth = 1;
                    for (const gy of [0, 0.25, 0.5, 0.75, 1.0]) {
                        ctx.beginPath();
                        ctx.moveTo(p.l, Y(gy));
                        ctx.lineTo(p.r, Y(gy));
                        ctx.stroke();
                        ctx.fillStyle = C.label;
                        ctx.font = "9px monospace";
                        ctx.textAlign = "right";
                        ctx.fillText(gy.toFixed(2), p.l - 4, Y(gy) + 3);
                    }

                    // High-noise domain shading.
                    ctx.fillStyle = C.highFill;
                    ctx.fillRect(p.l, p.t, p.r - p.l, Y(bound) - p.t);

                    // Executed overlay (behind the live curve).
                    if (executedOverlay) {
                        const m = executedOverlay.length - 1;
                        ctx.strokeStyle = C.executed;
                        ctx.setLineDash([3, 3]);
                        ctx.beginPath();
                        executedOverlay.forEach((s, i) => {
                            const px = p.l + ((p.r - p.l) * i) / Math.max(m, 1);
                            i === 0 ? ctx.moveTo(px, Y(s)) : ctx.lineTo(px, Y(s));
                        });
                        ctx.stroke();
                        ctx.setLineDash([]);
                    }

                    // Sigma curve + expert-colored step dots.
                    ctx.strokeStyle = C.sigma;
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    sigmas.forEach((s, i) => {
                        i === 0 ? ctx.moveTo(X(i), Y(s)) : ctx.lineTo(X(i), Y(s));
                    });
                    ctx.stroke();

                    let highSteps = 0;
                    for (let i = 0; i < n; i++) {
                        const isHigh = sigmas[i] >= bound;
                        if (isHigh) highSteps++;
                        ctx.fillStyle = isHigh ? C.stepHigh : C.stepLow;
                        ctx.beginPath();
                        ctx.arc(X(i), Y(sigmas[i]), 3, 0, Math.PI * 2);
                        ctx.fill();
                    }
                    const lowSteps = n - highSteps;

                    // Boundary line.
                    ctx.strokeStyle = C.boundary;
                    ctx.setLineDash([5, 3]);
                    ctx.lineWidth = dragBoundary ? 2 : 1.2;
                    ctx.beginPath();
                    ctx.moveTo(p.l, Y(bound));
                    ctx.lineTo(p.r, Y(bound));
                    ctx.stroke();
                    ctx.setLineDash([]);
                    ctx.fillStyle = C.boundary;
                    ctx.font = "9px monospace";
                    ctx.textAlign = "left";
                    ctx.fillText(
                        `boundary ${bound.toFixed(3)}`, p.l + 2, Y(bound) - 3);

                    // CFG lanes on the right axis, per phase over its own span.
                    const mode = widgetValue(node, "curve_mode") ?? "step";
                    const decay = widgetValue(node, "sigma_decay") ?? false;
                    const switchIdx = sigmas.findIndex((s) => s < bound);
                    const k = switchIdx === -1 ? n : switchIdx;
                    const phases = [
                        {
                            span: [0, k], sig: sigmas.slice(0, k + 1),
                            cfg: widgetValue(node, "cfg_high") ?? 3.5,
                            min: widgetValue(node, "cfg_high_min") ?? 1.0,
                            curve: discoverCurve(node, "curve_high"),
                            color: C.cfgHigh,
                        },
                        {
                            span: [k, n], sig: sigmas.slice(k),
                            cfg: widgetValue(node, "cfg_low") ?? 3.5,
                            min: widgetValue(node, "cfg_low_min") ?? 1.0,
                            curve: discoverCurve(node, "curve_low"),
                            color: C.cfgLow,
                        },
                    ];

                    let cfgMax = 1;
                    const series = phases.map((ph) => {
                        const s = phaseCfgSeries(
                            ph.sig, ph.cfg, ph.min,
                            ph.curve?.keys ?? null, mode, decay);
                        for (const v of s) cfgMax = Math.max(cfgMax, v);
                        return s;
                    });
                    cfgMax *= 1.1;
                    const Yc = (v) => p.b - ((p.b - p.t) * v) / cfgMax;

                    phases.forEach((ph, pi) => {
                        const s = series[pi];
                        if (!s.length) return;
                        ctx.strokeStyle = ph.color;
                        ctx.lineWidth = 1.5;
                        ctx.beginPath();
                        s.forEach((v, j) => {
                            const px = X(ph.span[0] + j);
                            j === 0 ? ctx.moveTo(px, Yc(v)) : ctx.lineTo(px, Yc(v));
                        });
                        ctx.stroke();
                        if (ph.curve?.unreadable) {
                            ctx.fillStyle = C.label;
                            ctx.font = "9px monospace";
                            ctx.textAlign = "center";
                            ctx.fillText("curve unreadable \u2014 constant shown",
                                X((ph.span[0] + ph.span[1]) / 2), p.t + 10);
                        }
                    });
                    ctx.fillStyle = C.label;
                    ctx.font = "9px monospace";
                    ctx.textAlign = "left";
                    ctx.fillText(`cfg \u2192 ${cfgMax.toFixed(1)}`, p.r + 3, p.t + 8);

                    // LoRA lane strip: live compile if lane nodes are wired,
                    // else the last executed compile from the ui payload.
                    if (strip.active) {
                        const segs = widgetValue(node, "lane_segments") ?? 4;
                        const laneData = lanesLive.length
                            ? compileLanePhases(sigmas, switchIdx, segs, lanesLive, mode)
                            : lanesExecOnly;
                        if (laneData?.names?.length) {
                            drawLaneStrip(ctx, strip, p.l, p.r, X, laneData,
                                switchIdx, n);
                        }
                    }

                    // Captions.
                    ctx.fillStyle = C.text;
                    ctx.font = "10px monospace";
                    ctx.textAlign = "left";
                    ctx.fillText(
                        `${highSteps} high / ${lowSteps} low` +
                        (k > 0 && k < n
                            ? ` \u00b7 hand-off \u03c3=${sigmas[k].toFixed(4)} (t=${(sigmas[k] * 1000).toFixed(0)})`
                            : ""),
                        p.l, (strip.active ? strip.bottom : p.b) + 12);
                    ctx.fillStyle = C.label;
                    ctx.fillText(caption, p.l, a.wy + 12);

                    // Divergence badge.
                    if (executedOverlay &&
                        executedOverlay.length === sigmas.length) {
                        let maxD = 0;
                        for (let i = 0; i < sigmas.length; i++) {
                            maxD = Math.max(maxD,
                                Math.abs(sigmas[i] - executedOverlay[i]));
                        }
                        if (maxD > DIVERGENCE_EPS) {
                            ctx.fillStyle = C.warn;
                            ctx.textAlign = "right";
                            ctx.fillText(
                                `\u26a0 preview differs from last run (\u0394${maxD.toFixed(4)})`,
                                p.r, a.wy + 12);
                        }
                    } else if (executedOverlay &&
                               executedOverlay.length !== sigmas.length) {
                        ctx.fillStyle = C.warn;
                        ctx.textAlign = "right";
                        ctx.fillText("\u26a0 step count changed since last run",
                            p.r, a.wy + 12);
                    }

                    ctx.restore();
                },

                mouse(event, pos, mouseNode) {
                    const a = getArea();
                    const p = plot(a);
                    if (stripNeeded()) p.b -= LANE_STRIP_H + 6;
                    const bound = boundaryValue();
                    const by = p.b - (p.b - p.t) * bound;
                    const type = event.type;

                    if (type === "pointerdown" || type === "mousedown") {
                        if (pos[0] >= p.l && pos[0] <= p.r &&
                            Math.abs(pos[1] - by) <= 6) {
                            dragBoundary = true;
                            return true;
                        }
                        return false;
                    }
                    if ((type === "pointermove" || type === "mousemove") && dragBoundary) {
                        const v = (p.b - pos[1]) / Math.max(p.b - p.t, 1);
                        setBoundary(v);
                        return true;
                    }
                    if ((type === "pointerup" || type === "mouseup") && dragBoundary) {
                        dragBoundary = false;
                        return true;
                    }
                    return false;
                },
            };

            this.widgets = this.widgets ?? [];
            this.widgets.push(widget);

            /* Overflow mouse hooks so a stretched node keeps interaction
               below the widget's computeSize bounds (curve-widget pattern). */
            function isInOverflow(pos) {
                const a = getArea();
                const widgetBottom = (widget.last_y ?? 0) + MIN_HEIGHT;
                return pos[1] > widgetBottom && pos[1] <= (widget.last_y ?? 0) + a.h
                    && pos[0] >= PAD_X && pos[0] <= currentHost.size[0] - PAD_R;
            }
            const hookedHosts = new WeakSet();
            function ensureOverflowHooks(host) {
                if (!host || hookedHosts.has(host)) return;
                hookedHosts.add(host);
                const origDown = host.onMouseDown;
                host.onMouseDown = function (event, pos, canvas) {
                    if (isInOverflow(pos)) {
                        return widget.mouse({ type: "pointerdown" }, pos, node);
                    }
                    return origDown?.call(this, event, pos, canvas);
                };
                const origMove = host.onMouseMove;
                host.onMouseMove = function (event, pos, canvas) {
                    if (dragBoundary) {
                        return widget.mouse({ type: "pointermove" }, pos, node);
                    }
                    return origMove?.call(this, event, pos, canvas);
                };
                const origUp = host.onMouseUp;
                host.onMouseUp = function (event, pos, canvas) {
                    if (dragBoundary) {
                        return widget.mouse({ type: "pointerup" }, pos, node);
                    }
                    return origUp?.call(this, event, pos, canvas);
                };
            }
            ensureOverflowHooks(node);

            requestAnimationFrame(() => {
                this.setSize(this.computeSize());
            });
        };
    },
});
