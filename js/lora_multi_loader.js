// LoRA Multi Loader (CCN) -- dynamic row widget with trigger-word chips.
//
// Rows serialize as widget values named "lora_<n>":
//   { on: bool, lora: string, strength: float, id: int }
// The Python side (lora_multi_loader.py) harvests them from **kwargs.
// Trigger chips are display-only and never serialized; they come from
// POST /ccn/lora_triggers on pick, on workflow configure, and via the
// Refresh Triggers button (which bypasses the cache).

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "CCN_LoraMultiLoader";
const ROW_TYPE = "CCN_LORA_ROW";
const STRENGTH_MIN = -10.0;
const STRENGTH_MAX = 10.0;
const DRAG_SCALE = 0.01; // strength units per pixel
const CLICK_TOLERANCE = 3; // px of movement still treated as a click
const CHIP_LINE_H = 16; // extra row height when trigger chips are shown

let loraListCache = null;

async function fetchLoraList() {
  try {
    const res = await api.fetchApi("/object_info/LoraLoaderModelOnly");
    const data = await res.json();
    const list = data?.LoraLoaderModelOnly?.input?.required?.lora_name?.[0];
    if (Array.isArray(list)) {
      loraListCache = list;
    }
  } catch (err) {
    console.warn("[CCN LoraMultiLoader] failed to fetch lora list", err);
  }
  return loraListCache ?? [];
}

let loraEntriesCache = null; // [{ name, mtime|null }]
let chooserSortMode = "name"; // "name" | "date"
try {
  chooserSortMode = localStorage.getItem("ccn.loramulti.sort") || "name";
} catch (_) { /* storage unavailable */ }

function setSortMode(mode) {
  chooserSortMode = mode;
  try { localStorage.setItem("ccn.loramulti.sort", mode); } catch (_) {}
}

export async function fetchLoraEntries() {
  try {
    const res = await api.fetchApi("/ccn/lora_filter/list");
    if (res.ok) {
      const data = await res.json();
      if (data && typeof data === "object" && !Array.isArray(data)) {
        loraEntriesCache = Object.entries(data).map(
          ([name, m]) => ({ name, mtime: m?.mtime || null }));
        return loraEntriesCache;
      }
    }
  } catch (err) {
    console.warn("[CCN LoraMultiLoader] /ccn/lora_filter/list unavailable, using name-only list", err);
  }
  const names = await fetchLoraList();
  loraEntriesCache = names.map((n) => ({ name: n, mtime: null }));
  return loraEntriesCache;
}

function relDate(mtime) {
  if (!mtime) return "";
  const s = Date.now() / 1000 - mtime;
  if (s < 3600) return `${Math.max(1, Math.floor(s / 60))}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  if (s < 86400 * 30) return `${Math.floor(s / 86400)}d`;
  if (s < 86400 * 365) return `${Math.floor(s / (86400 * 30))}mo`;
  return `${Math.floor(s / (86400 * 365))}y`;
}

// Multi-term AND filter (case-insensitive substrings), then sort by name
// or by mtime descending (missing mtimes last, name as tiebreaker).
function filterSortEntries(entries, query, mode) {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  const out = entries.filter((e) => {
    const n = e.name.toLowerCase();
    return terms.every((t) => n.includes(t));
  });
  if (mode === "date") {
    out.sort((a, b) =>
      ((b.mtime ?? -Infinity) - (a.mtime ?? -Infinity)) ||
      a.name.localeCompare(b.name));
  } else {
    out.sort((a, b) => a.name.localeCompare(b.name));
  }
  return out;
}

const CHOOSER_MAX_ITEMS = 400;

function openLoraChooser(event, onPick) {
  document.getElementById("ccn-lora-chooser")?.remove();

  const box = document.createElement("div");
  box.id = "ccn-lora-chooser";
  box.style.cssText =
    "position:fixed;z-index:10000;width:380px;max-height:440px;" +
    "display:flex;flex-direction:column;background:#1e1e2a;" +
    "border:1px solid #555;border-radius:5px;box-shadow:0 4px 16px #000a;" +
    "font:12px monospace;color:#ccc;";
  const cx = event?.clientX ?? window.innerWidth / 2;
  const cy = event?.clientY ?? window.innerHeight / 3;
  box.style.left = Math.max(4, Math.min(cx, window.innerWidth - 396)) + "px";
  box.style.top = Math.max(4, Math.min(cy, window.innerHeight - 456)) + "px";

  const bar = document.createElement("div");
  bar.style.cssText = "display:flex;gap:4px;padding:6px;flex:0 0 auto;";
  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "search…";
  input.style.cssText =
    "flex:1;background:#14141e;border:1px solid #444;border-radius:3px;" +
    "color:#ddd;padding:4px 6px;font:12px monospace;outline:none;";
  const btnName = document.createElement("button");
  const btnDate = document.createElement("button");
  for (const [b, label] of [[btnName, "A-Z"], [btnDate, "Newest"]]) {
    b.textContent = label;
    b.style.cssText =
      "flex:0 0 auto;padding:4px 8px;border:1px solid #555;border-radius:3px;" +
      "background:#2a2a2a;color:#ccc;cursor:pointer;font:11px monospace;";
  }
  bar.appendChild(input);
  bar.appendChild(btnName);
  bar.appendChild(btnDate);
  box.appendChild(bar);

  const listEl = document.createElement("div");
  listEl.style.cssText = "flex:1 1 auto;overflow-y:auto;padding:0 0 4px;";
  box.appendChild(listEl);

  let entries = loraEntriesCache ?? [];
  let visible = [];
  let highlight = 0;

  const hasDates = () => entries.some((e) => e.mtime);

  function styleSortButtons() {
    const active = (b, on) => {
      b.style.background = on ? "#3a3a55" : "#2a2a2a";
      b.style.color = on ? "#fff" : "#ccc";
    };
    active(btnName, chooserSortMode !== "date");
    active(btnDate, chooserSortMode === "date");
    const dated = hasDates();
    btnDate.disabled = !dated;
    btnDate.style.opacity = dated ? "1" : "0.4";
    btnDate.title = dated ? "sort by file date, newest first"
                          : "date info needs the CCN server route";
  }

  function paintHighlight() {
    for (const el of listEl.children) {
      el.style.background =
        Number(el.dataset.idx) === highlight ? "#33334a" : "transparent";
    }
  }

  function render() {
    const filtered = filterSortEntries(entries, input.value, chooserSortMode);
    visible = input.value.trim()
      ? filtered.map((e) => e)
      : [{ name: "None", mtime: null }, ...filtered];
    highlight = Math.min(highlight, Math.max(visible.length - 1, 0));

    listEl.textContent = "";
    const frag = document.createDocumentFragment();
    const shown = visible.slice(0, CHOOSER_MAX_ITEMS);
    shown.forEach((e, idx) => {
      const row = document.createElement("div");
      row.dataset.idx = String(idx);
      row.style.cssText =
        "display:flex;gap:8px;padding:3px 8px;cursor:pointer;" +
        "align-items:baseline;";
      const name = document.createElement("span");
      name.textContent = e.name;
      name.style.cssText =
        "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
      row.appendChild(name);
      if (e.mtime) {
        const d = document.createElement("span");
        d.textContent = relDate(e.mtime);
        d.style.cssText = "flex:0 0 auto;color:#777;font-size:10px;";
        row.appendChild(d);
      }
      row.addEventListener("mousemove", () => {
        if (highlight !== idx) { highlight = idx; paintHighlight(); }
      });
      row.addEventListener("mousedown", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        pick(e.name);
      });
      frag.appendChild(row);
    });
    if (visible.length > CHOOSER_MAX_ITEMS) {
      const more = document.createElement("div");
      more.textContent =
        `… ${visible.length - CHOOSER_MAX_ITEMS} more — refine the search`;
      more.style.cssText = "padding:4px 8px;color:#777;font-size:10px;";
      frag.appendChild(more);
    }
    listEl.appendChild(frag);
    paintHighlight();
    styleSortButtons();
  }

  function close() {
    document.removeEventListener("pointerdown", onOutside, true);
    box.remove();
  }

  function pick(name) {
    close();
    onPick(name);
  }

  function onOutside(ev) {
    if (!box.contains(ev.target)) close();
  }

  btnName.addEventListener("mousedown", (ev) => {
    ev.preventDefault(); setSortMode("name"); render();
  });
  btnDate.addEventListener("mousedown", (ev) => {
    ev.preventDefault(); if (hasDates()) { setSortMode("date"); render(); }
  });
  input.addEventListener("input", () => { highlight = 0; render(); });
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") { close(); }
    else if (ev.key === "Enter") {
      if (visible[highlight]) pick(visible[highlight].name);
    } else if (ev.key === "ArrowDown") {
      ev.preventDefault();
      highlight = Math.min(highlight + 1, Math.min(visible.length, CHOOSER_MAX_ITEMS) - 1);
      paintHighlight();
      listEl.children[highlight]?.scrollIntoView({ block: "nearest" });
    } else if (ev.key === "ArrowUp") {
      ev.preventDefault();
      highlight = Math.max(highlight - 1, 0);
      paintHighlight();
      listEl.children[highlight]?.scrollIntoView({ block: "nearest" });
    }
  });

  document.addEventListener("pointerdown", onOutside, true);
  document.body.appendChild(box);
  render();
  requestAnimationFrame(() => input.focus());

  // Refresh entries so a just-trained LoRA shows up with a fresh mtime.
  fetchLoraEntries().then((fresh) => {
    entries = fresh;
    render();
  });
}

const triggerCache = new Map(); // lora name -> string[] ([] = known none)

export async function fetchTriggers(names, { force = false } = {}) {
  const wanted = [...new Set(
    names.filter((n) => n && n !== "None" && (force || !triggerCache.has(n))))];
  if (!wanted.length) return;
  try {
    const res = await api.fetchApi("/ccn/lora_triggers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names: wanted }),
    });
    if (!res.ok) return;
    const data = await res.json();
    for (const n of wanted) {
      triggerCache.set(n, Array.isArray(data?.[n]) ? data[n] : []);
    }
  } catch (err) {
    console.warn("[CCN LoraMultiLoader] trigger fetch failed", err);
  }
}

function rowTriggers(row) {
  const name = row.value.lora;
  if (!name || name === "None") return null;
  const t = triggerCache.get(name);
  return t?.length ? t : null;
}

function copyText(text) {
  if (navigator.clipboard?.writeText) {
    return navigator.clipboard.writeText(text);
  }
  // Fallback for non-secure contexts (plain-http LAN).
  const ta = document.createElement("textarea");
  ta.value = text;
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  ta.remove();
  return Promise.resolve();
}

function baseName(path) {
  const cut = path.lastIndexOf("/");
  return cut === -1 ? path : path.slice(cut + 1);
}

function clampStrength(v) {
  if (!Number.isFinite(v)) return 0;
  return Math.min(STRENGTH_MAX, Math.max(STRENGTH_MIN, Math.round(v * 100) / 100));
}

export function isRowValue(v) {
  return v && typeof v === "object" && typeof v.lora === "string";
}

export function rowWidgets(node) {
  return node.widgets ? node.widgets.filter((w) => w.type === ROW_TYPE) : [];
}

export function rowLoraNames(node) {
  return rowWidgets(node).map((r) => r.value.lora);
}

// Re-fit height after row heights change, preserving user-set width.
export function resizeRows(node) {
  const computed = node.computeSize();
  node.setSize([Math.max(node.size[0], computed[0]), computed[1]]);
  node.setDirtyCanvas(true, true);
}

function makeRow(node, value) {
  node._ccnRowCounter = (node._ccnRowCounter ?? 0) + 1;
  const id = value?.id ?? node._ccnRowCounter;
  node._ccnRowCounter = Math.max(node._ccnRowCounter, Number(id) || 0);
  const widget = {
    type: ROW_TYPE,
    name: `lora_${id}`,
    value: {
      on: true,
      lora: "None",
      strength: 1.0,
      ...(value ?? {}),
      id,
    },
    _hit: {},
    _drag: null,
    _copied: null,

    computeSize(width) {
      const base = LiteGraph.NODE_WIDGET_HEIGHT;
      return [width ?? 200, rowTriggers(this) ? base + CHIP_LINE_H : base];
    },

    serializeValue() {
      const { on, lora, strength, id: rowId } = this.value;
      return { on, lora, strength, id: rowId };
    },

    draw(ctx, drawNode, width, y, height) {
      const margin = 12;
      const lineH = LiteGraph.NODE_WIDGET_HEIGHT;
      const chips = rowTriggers(this);
      const totalH = lineH + (chips ? CHIP_LINE_H : 0);
      const midY = y + lineH / 2;
      const on = !!this.value.on;
      ctx.save();

      ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
      ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
      ctx.beginPath();
      ctx.roundRect(margin, y + 1, width - margin * 2, totalH - 2, lineH * 0.3);
      ctx.fill();
      ctx.stroke();

      // Toggle dot.
      const toggleX = margin + 11;
      ctx.fillStyle = on ? "#7f7" : "#666";
      ctx.beginPath();
      ctx.arc(toggleX, midY, 4.5, 0, Math.PI * 2);
      ctx.fill();
      this._hit.toggle = [margin, margin + 22];
      this._hit.line1 = [y, y + lineH];

      // Remove [x] at far right.
      const removeX1 = width - margin - 4;
      const removeX0 = removeX1 - 14;
      ctx.fillStyle = "#a66";
      ctx.font = "11px Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("✕", (removeX0 + removeX1) / 2, midY);
      this._hit.remove = [removeX0 - 2, removeX1 + 2];

      // Strength field, right-aligned before the remove button.
      ctx.font = "12px Arial";
      const fieldW = 52;
      const strengthX1 = removeX0 - 6;
      const strengthX0 = strengthX1 - fieldW;
      const textColor = on
        ? LiteGraph.WIDGET_TEXT_COLOR
        : LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;

      ctx.textAlign = "left";
      ctx.fillStyle = textColor;
      ctx.fillText(this.value.strength.toFixed(2), strengthX0 + 8, midY);
      this._hit.strength = [strengthX0 - 2, strengthX1];

      // Name fills the remaining middle.
      const nameX0 = margin + 24;
      const nameX1 = strengthX0 - 8;
      let display = this.value.lora === "None"
        ? "click to choose…"
        : baseName(this.value.lora);
      ctx.fillStyle = this.value.lora === "None"
        ? LiteGraph.WIDGET_SECONDARY_TEXT_COLOR
        : textColor;
      const maxW = nameX1 - nameX0;
      while (display.length > 4 && ctx.measureText(display).width > maxW) {
        display = display.slice(0, -2);
      }
      ctx.fillText(display, nameX0, midY);
      this._hit.name = [nameX0 - 4, nameX1];

      // Trigger chips on the second line, click to copy.
      this._hit.chips = [];
      if (chips) {
        const chipY0 = y + lineH - 1;
        const chipH = CHIP_LINE_H - 4;
        const maxX = width - margin - 6;
        ctx.font = "9px monospace";
        let cxp = nameX0;
        for (const word of chips) {
          let label = word;
          let tw = ctx.measureText(label).width;
          while (label.length > 4 && cxp + tw + 10 > maxX) {
            label = label.slice(0, -2) + "…";
            tw = ctx.measureText(label).width;
          }
          if (cxp + tw + 10 > maxX) break;
          const chipW = tw + 10;
          const copied = this._copied?.word === word &&
            Date.now() < this._copied.until;
          ctx.fillStyle = copied ? "#3d6b3d" : "#33334a";
          ctx.beginPath();
          ctx.roundRect(cxp, chipY0, chipW, chipH, 3);
          ctx.fill();
          ctx.fillStyle = copied ? "#cfc" : on ? "#aab" : "#777";
          ctx.fillText(label, cxp + 5, chipY0 + chipH / 2 + 0.5);
          this._hit.chips.push(
            { x0: cxp, x1: cxp + chipW, y0: chipY0, y1: chipY0 + chipH, word });
          cxp += chipW + 4;
        }
      }

      ctx.restore();
    },

    mouse(event, pos, mouseNode) {
      const x = pos[0];
      const yy = pos[1];
      const type = event.type;
      const inZone = (zone) => zone && x >= zone[0] && x <= zone[1];
      const line1 = this._hit.line1;
      const inLine1 = !line1 || (yy >= line1[0] && yy <= line1[1]);

      if (type === "pointerdown" || type === "mousedown") {
        if (!inLine1) {
          const chip = (this._hit.chips ?? []).find(
            (c) => x >= c.x0 && x <= c.x1 && yy >= c.y0 && yy <= c.y1);
          if (chip) {
            copyText(chip.word).then(() => {
              this._copied = { word: chip.word, until: Date.now() + 800 };
              mouseNode.setDirtyCanvas(true, true);
              setTimeout(() => mouseNode.setDirtyCanvas(true, true), 850);
            });
            return true;
          }
          return false;
        }
        if (inZone(this._hit.toggle)) {
          this.value.on = !this.value.on;
          mouseNode.setDirtyCanvas(true, true);
          return true;
        }
        if (inZone(this._hit.remove)) {
          const idx = mouseNode.widgets.indexOf(this);
          if (idx !== -1) {
            mouseNode.widgets.splice(idx, 1);
            resizeRows(mouseNode);
          }
          return true;
        }
        if (inZone(this._hit.strength)) {
          this._drag = { startX: x, startValue: this.value.strength, moved: false };
          return true;
        }
        if (inZone(this._hit.name)) {
          this._drag = null;
          this._openChooser(event, mouseNode);
          return true;
        }
        return false;
      }

      if ((type === "pointermove" || type === "mousemove") && this._drag) {
        const delta = x - this._drag.startX;
        if (Math.abs(delta) > CLICK_TOLERANCE) this._drag.moved = true;
        if (this._drag.moved) {
          this.value.strength = clampStrength(this._drag.startValue + delta * DRAG_SCALE);
          mouseNode.setDirtyCanvas(true, true);
        }
        return true;
      }

      if ((type === "pointerup" || type === "mouseup") && this._drag) {
        const drag = this._drag;
        this._drag = null;
        if (!drag.moved) {
          // Treat as a click: type an exact value.
          app.canvas.prompt(
            "Strength",
            this.value.strength,
            (v) => {
              const parsed = parseFloat(v);
              if (Number.isFinite(parsed)) {
                this.value.strength = clampStrength(parsed);
                mouseNode.setDirtyCanvas(true, true);
              }
            },
            event
          );
        }
        return true;
      }

      return false;
    },

    _openChooser(event, chooserNode) {
      openLoraChooser(event, (v) => {
        this.value.lora = v;
        resizeRows(chooserNode);
        fetchTriggers([v]).then(() => resizeRows(chooserNode));
      });
    },
  };
  return widget;
}

export function addRow(node, value) {
  const row = makeRow(node, value);
  const buttonIndex = node.widgets.findIndex((w) => w === node._ccnAddButton);
  if (buttonIndex === -1) {
    node.widgets.push(row);
  } else {
    node.widgets.splice(buttonIndex, 0, row);
  }
  resizeRows(node);
  return row;
}

export function ensureRowCount(node, count) {
  let rows = rowWidgets(node);
  while (rows.length < count) {
    addRow(node);
    rows = rowWidgets(node);
  }
  while (rows.length > count) {
    const idx = node.widgets.indexOf(rows[rows.length - 1]);
    node.widgets.splice(idx, 1);
    rows = rowWidgets(node);
  }
  resizeRows(node);
}

app.registerExtension({
  name: "CCN.LoraMultiLoader",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      origOnNodeCreated?.apply(this, arguments);
      this._ccnRowCounter = 0;
      this.widgets = this.widgets ?? [];

      this._ccnAddButton = this.addWidget(
        "button",
        "＋ Add LoRA",
        null,
        () => {
          addRow(this);
          this.setDirtyCanvas(true, true);
        }
      );
      this._ccnAddButton.options = this._ccnAddButton.options ?? {};
      this._ccnAddButton.options.serialize = false;

      this._ccnRefreshButton = this.addWidget(
        "button",
        "↻ Refresh Triggers",
        null,
        () => {
          fetchTriggers(rowLoraNames(this), { force: true })
            .then(() => resizeRows(this));
        }
      );
      this._ccnRefreshButton.options = this._ccnRefreshButton.options ?? {};
      this._ccnRefreshButton.options.serialize = false;

      addRow(this); // one starter row for discoverability
      fetchLoraEntries().then(() => this.setDirtyCanvas(true, true));
    };

    const origConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      // Rebuild the correct number of rows BEFORE LiteGraph applies
      // widgets_values by index, so the arrays line up.
      if (info?.widgets_values) {
        ensureRowCount(this, info.widgets_values.filter(isRowValue).length);
      }
      const result = origConfigure?.apply(this, arguments);
      // Re-sync row values defensively (index alignment can differ across
      // frontend versions when non-serialized widgets are present).
      if (info?.widgets_values) {
        const rowValues = info.widgets_values.filter(isRowValue);
        const rows = rowWidgets(this);
        let maxId = 0;
        for (let i = 0; i < rows.length && i < rowValues.length; i++) {
          const id = rowValues[i].id ?? i + 1;
          rows[i].value = {
            on: true,
            lora: "None",
            strength: 1.0,
            ...rowValues[i],
            id,
          };
          rows[i].name = `lora_${id}`;
          maxId = Math.max(maxId, Number(id) || 0);
        }
        this._ccnRowCounter = Math.max(this._ccnRowCounter ?? 0, maxId);

        // Saves that predate the scalar widgets hold only row dicts, so
        // positional application can stuff a row into them. Recover the
        // saved values by type, falling back to defaults.
        const enabledWidget = this.widgets.find((w) => w.name === "enabled");
        if (enabledWidget) {
          const saved = info.widgets_values.find((v) => typeof v === "boolean");
          if (saved !== undefined) enabledWidget.value = saved;
          else if (typeof enabledWidget.value !== "boolean") enabledWidget.value = true;
        }
        const multWidget = this.widgets.find(
          (w) => w.name === "strength_multiplier");
        if (multWidget) {
          const saved = info.widgets_values.find(
            (v) => typeof v === "number" && Number.isFinite(v));
          if (saved !== undefined) multWidget.value = saved;
          else if (!Number.isFinite(multWidget.value)) multWidget.value = 1.0;
        }
      }
      fetchTriggers(rowLoraNames(this)).then(() => resizeRows(this));
      this.setDirtyCanvas(true, true);
      return result;
    };
  },
});
