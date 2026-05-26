import { app } from "../../scripts/app.js";

/* ───────────────────────────────────────────────────────────────
   CCN Curve Editor Widget
   
   Interactive cubic-Hermite curve editor rendered on the LiteGraph
   canvas.  Endpoints pinned at x=0 and x=1 (y is free).
   Intermediate keyframes can be added / removed / dragged freely.
   Tangent handles control Hermite in/out slopes per keyframe.

   Interactions:
     • Drag keyframe         – move position  (endpoint X locked)
     • Drag tangent handle   – adjust slope (mirrored by default)
     • Shift+drag handle     – break tangent, adjust one side only
     • Double-click curve    – add keyframe
     • Double-click key      – delete keyframe (not endpoints)
     • Right-click key       – mirror / break tangents
   ─────────────────────────────────────────────────────────────── */

const MIN_HEIGHT = 200;
const PAD_X      = 28;
const PAD_R      = 14;
const PAD_Y      = 14;
const PAD_B      = 20;
const KEY_R      = 5;
const HDL_R      = 3.5;
const HDL_LEN    = 40;
const DBL_MS     = 350;
const MIN_KEY_DX = 0.015;

const C = {
    bg:          "#181825",
    grid:        "#27273a",
    gridEdge:    "#2e2e48",
    curve:       "#4ec9b0",
    curveW:      2.5,
    key:         "#e0e0e0",
    keyStroke:   "#000000",
    keySel:      "#f0c050",
    keyBroken:   "#c07040",
    hdl:         "#8888aa",
    hdlLine:     "#55557a",
    marker:      "#e06070",
    markerCross: "#e0607030",
    label:       "#6a6a80",
};

/* Per-node-type default curves */
const DEFAULT_KEYS_LINEAR = [
    { x: 0, y: 0, in: 0, out: 1, mirrored: true },
    { x: 1, y: 1, in: 1, out: 0, mirrored: true },
];
const DEFAULT_KEYS_CFG = [
    { x: 0, y: 1, in: 0, out: -1, mirrored: true },
    { x: 1, y: 0, in: -1, out: 0, mirrored: true },
];

/* ── Hermite evaluation (mirrors Python _hermite exactly) ──── */
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

    const lt  = (t - k0.x) / dt;
    const lt2 = lt * lt;
    const lt3 = lt2 * lt;

    return (2*lt3 - 3*lt2 + 1) * k0.y
         + (lt3 - 2*lt2 + lt)  * k0.out * dt
         + (-2*lt3 + 3*lt2)    * k1.y
         + (lt3 - lt2)         * k1.in  * dt;
}

/* ── Coordinate helpers ─────────────────────────────────────── */
function plotArea(nodeW, wy, h) {
    const l = PAD_X, r = nodeW - PAD_R;
    const t = wy + PAD_Y, b = wy + h - PAD_B;
    return { l, r, t, b, w: r - l, h: b - t };
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
    const sign  = type === "out" ? 1 : -1;
    const dx = sign * a.w;
    const dy = -slope * a.h * sign;
    const len = Math.hypot(dx, dy);
    if (len < 1e-6) return [kx + sign * HDL_LEN, ky];
    return [kx + (dx / len) * HDL_LEN, ky + (dy / len) * HDL_LEN];
}

function hitTest(px, py, keys, a) {
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

/* ── Extension registration ─────────────────────────────────── */
app.registerExtension({
    name: "CCN.CurveWidget",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "CCN_CurveSample"
         && nodeData.name !== "CCN_CurveCFGGuider"
         && nodeData.name !== "CCN_NeutralPromptGuider"
         && nodeData.name !== "CCN_CurveDefinition") return;

        const DEFAULT_KEYS = (nodeData.name === "CCN_CurveCFGGuider" || nodeData.name === "CCN_NeutralPromptGuider") ? DEFAULT_KEYS_CFG : DEFAULT_KEYS_LINEAR;
        const hasCurveInput = nodeData.name !== "CCN_CurveDefinition";

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);

            const idx = this.widgets.findIndex((w) => w.name === "curve_data");
            if (idx < 0) return;
            this.widgets.splice(idx, 1);

            /* ── State ──────────────────────────────────────────── */
            let keys        = JSON.parse(JSON.stringify(DEFAULT_KEYS));
            let selectedKey = -1;
            let dragTarget  = null;
            let lastClickMs = 0;
            let loadedFromSave = false;
            let locked = false;
            // The node whose draw callback most recently fired.  In a
            // subgraph with a promoted widget, this is the parent
            // subgraph node rather than the original.  All geometry
            // and mouse math should use this, not the closure `node`.
            let currentHost = this;

            const node = this;

            function sync() {
                _value = JSON.stringify(keys);
                app.graph?.setDirtyCanvas?.(true, true);
            }

            /* Compute the current plot area from live node geometry */
            function getCurveArea() {
                const wy = widget.last_y;
                const h  = Math.max(MIN_HEIGHT, currentHost.size[1] - wy - 10);
                return { wy, h, plot: plotArea(currentHost.size[0], wy, h) };
            }

            /* ── Shared mouse logic ─────────────────────────────── */
            function onDown(mx, my, event) {
                if (locked) return false;
                const { plot: a } = getCurveArea();
                const hit = hitTest(mx, my, keys, a);
                const now = performance.now();

                // Double-click
                if (now - lastClickMs < DBL_MS) {
                    if (hit && hit.type === "key" && hit.index > 0 && hit.index < keys.length - 1) {
                        keys.splice(hit.index, 1);
                        selectedKey = -1;
                        dragTarget  = null;
                        lastClickMs = 0;
                        sync();
                        return true;
                    }
                    if (!hit && mx >= a.l && mx <= a.r && my >= a.t && my <= a.b) {
                        const [cx] = p2c(mx, my, a);
                        if (cx > MIN_KEY_DX && cx < 1 - MIN_KEY_DX) {
                            const tooClose = keys.some((k) => Math.abs(k.x - cx) < MIN_KEY_DX);
                            if (!tooClose) {
                                const evalY = hermite(keys, cx);
                                let slope = 1;
                                keys.push({ x: cx, y: evalY, in: 0, out: 0, mirrored: true });
                                keys.sort((ka, kb) => ka.x - kb.x);
                                const ni = keys.findIndex((k) => Math.abs(k.x - cx) < 0.001);
                                if (ni > 0 && ni < keys.length - 1) {
                                    slope = (keys[ni + 1].y - keys[ni - 1].y)
                                          / (keys[ni + 1].x - keys[ni - 1].x);
                                    keys[ni].in  = slope;
                                    keys[ni].out = slope;
                                }
                                selectedKey = ni;
                                lastClickMs = 0;
                                sync();
                                return true;
                            }
                        }
                    }
                    lastClickMs = 0;
                    return true;
                }

                lastClickMs = now;

                if (hit) {
                    dragTarget  = hit;
                    selectedKey = hit.index;
                    installDragSafety();
                    app.canvas.dirty_canvas = true;
                    return true;
                }

                if (mx >= a.l && mx <= a.r && my >= a.t && my <= a.b) {
                    selectedKey = -1;
                    app.canvas.dirty_canvas = true;
                    return true;
                }
                return false;
            }

            function onMove(mx, my, event) {
                if (!dragTarget) return false;
                const { plot: a } = getCurveArea();
                const k = keys[dragTarget.index];

                if (dragTarget.type === "key") {
                    const isFirst = dragTarget.index === 0;
                    const isLast  = dragTarget.index === keys.length - 1;
                    const [cx, cy] = p2c(mx, my, a);

                    if (isFirst || isLast) {
                        k.y = cy;
                    } else {
                        const lo = keys[dragTarget.index - 1].x + MIN_KEY_DX;
                        const hi = keys[dragTarget.index + 1].x - MIN_KEY_DX;
                        k.x = Math.max(lo, Math.min(hi, cx));
                        k.y = cy;
                    }
                } else {
                    const [kpx, kpy] = c2p(k.x, k.y, a);
                    const dx = mx - kpx;
                    const dy = my - kpy;

                    if (event.shiftKey && k.mirrored !== false) {
                        k.mirrored = false;
                    }

                    let newSlope = null;
                    if (dragTarget.type === "out" && dx > 1) {
                        newSlope = -dy * a.w / (dx * a.h);
                        k.out = newSlope;
                    } else if (dragTarget.type === "in" && dx < -1) {
                        newSlope = -dy * a.w / (dx * a.h);
                        k.in = newSlope;
                    }

                    if (newSlope !== null && k.mirrored !== false) {
                        k.in  = newSlope;
                        k.out = newSlope;
                    }
                }
                sync();
                return true;
            }

            /* Document-level safety net for missed pointer releases.
               Installed when a drag starts, removed when it ends.
               Catches releases outside the node, pointercancel from
               the browser, and window blur (alt-tab mid-drag). */
            function forceRelease() {
                dragTarget = null;
                document.removeEventListener("pointerup", onDocPointerUp, true);
                document.removeEventListener("pointercancel", onDocPointerUp, true);
                window.removeEventListener("blur", onDocPointerUp);
            }
            function onDocPointerUp() {
                if (dragTarget) {
                    forceRelease();
                    app.canvas.dirty_canvas = true;
                }
            }
            function installDragSafety() {
                document.addEventListener("pointerup", onDocPointerUp, true);
                document.addEventListener("pointercancel", onDocPointerUp, true);
                window.addEventListener("blur", onDocPointerUp);
            }

            function onUp() {
                if (!dragTarget) return false;
                forceRelease();
                return true;
            }

            /* ── Widget ─────────────────────────────────────────── */
            let _value = JSON.stringify(DEFAULT_KEYS);

            const widget = {
                type:    "CURVE_EDITOR",
                name:    "curve_data",
                options: { serialize: true },
                last_y:  0,

                /* Fixed size — no circular dependency with node.size.
                   LiteGraph routes mouse events within this region to
                   the widget; overflow is handled by node-level hooks. */
                computeSize() { return [250, MIN_HEIGHT]; },

                draw(ctx, _node, width, wy, _h) {
                    currentHost = _node;
                    ensureOverflowHooks(_node);
                    updateLockState();
                    syncExternalCurve();
                    const availH = Math.max(MIN_HEIGHT, _node.size[1] - wy - 10);
                    widget.last_y = wy;
                    const a = plotArea(width, wy, availH);

                    ctx.save();
                    ctx.fillStyle = C.bg;
                    ctx.beginPath();
                    ctx.roundRect(2, wy + 2, width - 4, availH - 4, 6);
                    ctx.fill();
                    ctx.clip();

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
                    ctx.textAlign = "center";
                    ctx.textBaseline = "top";
                    ctx.fillText("0", a.l, a.b + 3);
                    ctx.fillText(".5", a.l + a.w * 0.5, a.b + 3);
                    ctx.fillText("1", a.r, a.b + 3);
                    ctx.textAlign = "right";
                    ctx.textBaseline = "middle";
                    ctx.fillText("0", a.l - 4, a.b);
                    ctx.fillText("1", a.l - 4, a.t);

                    ctx.strokeStyle = C.curve;
                    ctx.lineWidth = C.curveW;
                    ctx.lineJoin = "round";
                    ctx.beginPath();
                    const steps = Math.max(Math.round(a.w), 80);
                    for (let s = 0; s <= steps; s++) {
                        const t  = s / steps;
                        const yv = hermite(keys, t);
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

                        if (i > 0)              drawHandle("in");
                        if (i < keys.length - 1) drawHandle("out");

                        const isBroken = k.mirrored === false;
                        ctx.fillStyle   = i === selectedKey ? C.keySel
                                        : isBroken         ? C.keyBroken
                                        :                    C.key;
                        ctx.strokeStyle = C.keyStroke;
                        ctx.lineWidth   = 1.2;
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
                        ctx.fill(); ctx.stroke();
                    }

                    const tW = currentHost.widgets?.find((w) => w.name === "t");
                    if (tW != null) {
                        const xsW = currentHost.widgets?.find((w) => w.name === "x_scale");
                        const xs = (xsW && Math.abs(xsW.value) > 1e-10) ? xsW.value : 1;
                        const tv = Math.max(0, Math.min(1, tW.value / xs));
                        const yv = hermite(keys, tv);
                        const [mx, my] = c2p(tv, yv, a);

                        ctx.strokeStyle = C.markerCross;
                        ctx.lineWidth = 1;
                        ctx.setLineDash([3, 3]);
                        ctx.beginPath(); ctx.moveTo(mx, a.t); ctx.lineTo(mx, a.b); ctx.stroke();
                        ctx.beginPath(); ctx.moveTo(a.l, my); ctx.lineTo(a.r, my); ctx.stroke();
                        ctx.setLineDash([]);

                        ctx.fillStyle = C.marker;
                        ctx.beginPath(); ctx.arc(mx, my, 4, 0, Math.PI * 2); ctx.fill();
                    }

                    // Locked overlay when external CURVE is connected
                    if (locked) {
                        const bannerH = 20;
                        ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
                        ctx.fillRect(a.l, a.t, a.w, bannerH);
                        ctx.fillStyle = "#8888aa";
                        ctx.font = "10px monospace";
                        ctx.textAlign = "center";
                        ctx.textBaseline = "middle";
                        ctx.fillText("External Curve", (a.l + a.r) / 2, a.t + bannerH / 2);
                    }

                    ctx.restore();
                },

                /* Widget mouse — covers the MIN_HEIGHT region that
                   LiteGraph routes to us based on computeSize. */
                mouse(event, pos, _node) {
                    currentHost = _node;
                    if (event.type === "pointerdown") return onDown(pos[0], pos[1], event);
                    if (event.type === "pointermove") return onMove(pos[0], pos[1], event);
                    if (event.type === "pointerup")   return onUp();
                    return false;
                },
            };

            Object.defineProperty(widget, "value", {
                get()  { return _value; },
                set(v) {
                    loadedFromSave = true;
                    _value = v;
                    try {
                        const parsed = JSON.parse(v);
                        if (Array.isArray(parsed) && parsed.length >= 2) {
                            keys.length = 0;
                            keys.push(...parsed.map((k) => ({
                                ...k,
                                mirrored: k.mirrored !== false,
                            })));
                        }
                    } catch (_) { /* keep current keys on bad data */ }
                },
                configurable: true,
                enumerable:   true,
            });

            this.widgets.splice(idx, 0, widget);

            /* ── Context menu ───────────────────────────────────── */
            const origMenu = this.getExtraMenuOptions;
            this.getExtraMenuOptions = function (canvas, options) {
                origMenu?.apply(this, arguments);
                if (selectedKey >= 0 && selectedKey < keys.length) {
                    const sk = keys[selectedKey];
                    const isEndpoint = selectedKey === 0 || selectedKey === keys.length - 1;

                    if (!isEndpoint) {
                        if (sk.mirrored !== false) {
                            options.unshift({
                                content: "Break Tangents",
                                callback: () => { sk.mirrored = false; sync(); },
                            });
                        } else {
                            options.unshift({
                                content: "Mirror Tangents",
                                callback: () => {
                                    sk.mirrored = true;
                                    sk.in = sk.out;
                                    sync();
                                },
                            });
                        }

                        options.unshift({
                            content: `Delete Curve Key ${selectedKey}`,
                            callback: () => {
                                keys.splice(selectedKey, 1);
                                selectedKey = -1;
                                sync();
                            },
                        });
                    }
                }
                options.unshift({
                    content: "Reset Curve",
                    callback: () => {
                        keys.length = 0;
                        keys.push(...JSON.parse(JSON.stringify(DEFAULT_KEYS)));
                        selectedKey = -1;
                        sync();
                    },
                });
            };

            /* ── Node-level mouse for the overflow region ────────
               When the node is taller than MIN_HEIGHT, the drawn
               curve extends below the widget's computeSize bounds.
               LiteGraph won't route those clicks to the widget, so
               we catch them here.  Clicks within the widget's own
               bounds are handled by widget.mouse above.
               ──────────────────────────────────────────────────── */
            function isInOverflow(pos) {
                const { wy, h } = getCurveArea();
                const widgetBottom = wy + MIN_HEIGHT;
                const curveBottom  = wy + h;
                return pos[1] > widgetBottom && pos[1] <= curveBottom
                    && pos[0] >= PAD_X && pos[0] <= currentHost.size[0] - PAD_R;
            }

            /* Lazily install overflow mouse hooks on any host node that
               draws this widget — including subgraph wrappers when the
               widget is promoted.  WeakSet tracks which hosts are
               already hooked so we don't double-wrap. */
            const hookedHosts = new WeakSet();
            function ensureOverflowHooks(host) {
                if (!host || hookedHosts.has(host)) return;
                hookedHosts.add(host);

                const origDown = host.onMouseDown;
                host.onMouseDown = function (event, pos, canvas) {
                    if (isInOverflow(pos)) {
                        return onDown(pos[0], pos[1], event);
                    }
                    return origDown?.call(this, event, pos, canvas);
                };

                const origMove = host.onMouseMove;
                host.onMouseMove = function (event, pos, canvas) {
                    if (dragTarget) return onMove(pos[0], pos[1], event);
                    return origMove?.call(this, event, pos, canvas);
                };

                const origUp = host.onMouseUp;
                host.onMouseUp = function (event, pos, canvas) {
                    if (dragTarget) return onUp();
                    return origUp?.call(this, event, pos, canvas);
                };
            }

            // Hook the original node immediately
            ensureOverflowHooks(node);

            /* ── Lock widget when external CURVE is connected ──── */
            function updateLockState() {
                if (!hasCurveInput) return;
                const curveSlot = node.inputs?.findIndex((i) => i.name === "curve");
                const isLinked = curveSlot >= 0 && node.inputs[curveSlot].link != null;
                if (isLinked !== locked) {
                    locked = isLinked;
                    if (locked) {
                        dragTarget = null;
                        selectedKey = -1;
                    }
                    app.graph?.setDirtyCanvas?.(true, true);
                }
            }

            /* Read curve keys from the connected source node's widget.
               Works in subgraph contexts by using the graph that
               actually contains this node, and falling back to
               LiteGraph's getInputNode if direct link lookup fails. */
            function syncExternalCurve() {
                if (!locked) return;
                const curveSlot = node.inputs?.findIndex((i) => i.name === "curve");
                if (curveSlot < 0) return;

                let srcNode = null;

                // Preferred: LiteGraph's built-in accessor
                if (typeof node.getInputNode === "function") {
                    try { srcNode = node.getInputNode(curveSlot); } catch (_) {}
                }

                // Fallback: resolve via the graph containing this node
                if (!srcNode) {
                    const linkId = node.inputs[curveSlot].link;
                    if (linkId == null) return;
                    const graph = node.graph || app.graph;
                    const link = graph?.links?.[linkId]
                              ?? (graph?.links?.get && graph.links.get(linkId));
                    if (!link) return;
                    srcNode = graph.getNodeById(link.origin_id);
                }

                if (!srcNode) return;
                const srcWidget = srcNode.widgets?.find((w) => w.name === "curve_data");
                if (!srcWidget) return;

                try {
                    const srcKeys = JSON.parse(srcWidget.value);
                    if (Array.isArray(srcKeys) && srcKeys.length >= 2) {
                        keys.length = 0;
                        keys.push(...srcKeys.map((k) => ({
                            ...k,
                            mirrored: k.mirrored !== false,
                        })));
                    }
                } catch (_) { /* keep current keys */ }
            }

            if (hasCurveInput) {
                const origConnChange = node.onConnectionsChange;
                node.onConnectionsChange = function (type, slotIndex, isConnected, link, ioSlot) {
                    origConnChange?.call(this, type, slotIndex, isConnected, link, ioSlot);
                    updateLockState();
                };
            }

            /* ── Initial sizing ──────────────────────────────────
               On fresh creation, LiteGraph sizes the node before our
               widget swap, leaving it too tall.  Defer a resize to
               fix it.  On workflow load, ComfyUI restores saved
               widget values (setting loadedFromSave) before the next
               frame, so we skip and preserve the saved size.
               ──────────────────────────────────────────────────── */
            requestAnimationFrame(() => {
                if (!loadedFromSave) {
                    this.setSize(this.computeSize());
                }
            });
        };
    },
});
