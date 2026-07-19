// LoRA Pair Loader (CCN) -- dynamic row widget.
//
// Rows serialize as widget values named "lora_<n>":
//   { on: bool, lora: string, strength_high: float, strength_low: float }
// The Python side (lora_pair_loader.py) harvests them from **kwargs and is
// authoritative for pair resolution; the badge here is a preview only.
// The pair-token table is mirrored from the Python file; keep in sync.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "CCN_LoraPairLoader";
const ROW_TYPE = "CCN_LORA_PAIR_ROW";
const STRENGTH_MIN = -10.0;
const STRENGTH_MAX = 10.0;
const DRAG_SCALE = 0.01; // strength units per pixel
const CLICK_TOLERANCE = 3; // px of movement still treated as a click

const PAIR_TOKENS = [
  ["high_noise", "low_noise"],
  ["high-noise", "low-noise"],
  ["high noise", "low noise"],
  ["highnoise", "lownoise"],
  ["_high", "_low"],
  ["-high", "-low"],
  [".high", ".low"],
  [" high", " low"],
  ["high_", "low_"],
  ["high-", "low-"],
  ["high.", "low."],
  ["high ", "low "],
  ["high/", "low/"],
  ["high\\", "low\\"],
  ["/high", "/low"],
  ["\\high", "\\low"],
];

// Single-letter (h/l) and abbreviated (hn/ln) markers across the delimiter
// product, matching the generated table in lora_pair_loader.py.
const ABBREV_DELIMS = ["_", "-", ".", " ", "/", "\\"];
for (const [abbrevHigh, abbrevLow] of [["hn", "ln"], ["h", "l"]]) {
  for (const left of ABBREV_DELIMS) {
    for (const right of ABBREV_DELIMS) {
      PAIR_TOKENS.push([`${left}${abbrevHigh}${right}`, `${left}${abbrevLow}${right}`]);
    }
  }
}

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
    console.warn("[CCN LoraPairLoader] failed to fetch lora list", err);
  }
  return loraListCache ?? [];
}

let loraEntriesCache = null; // [{ name, mtime|null }]
let chooserSortMode = "name"; // "name" | "date"
try {
  chooserSortMode = localStorage.getItem("ccn.lorapair.sort") || "name";
} catch (_) { /* storage unavailable */ }

function setSortMode(mode) {
  chooserSortMode = mode;
  try { localStorage.setItem("ccn.lorapair.sort", mode); } catch (_) {}
}

async function fetchLoraEntries() {
  try {
    const res = await api.fetchApi("/ccn/loras");
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data) && data.every((e) => typeof e?.name === "string")) {
        loraEntriesCache = data;
        loraListCache = data.map((e) => e.name); // keeps the pair badge fed
        return loraEntriesCache;
      }
    }
  } catch (err) {
    console.warn("[CCN LoraPairLoader] /ccn/loras unavailable, using name-only list", err);
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
  input.placeholder = "search\u2026";
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
        `\u2026 ${visible.length - CHOOSER_MAX_ITEMS} more \u2014 refine the search`;
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

function hasPartner(loraName) {
  if (!loraListCache || !loraName || loraName === "None") return false;
  const lowerSet = new Set(loraListCache.map((n) => n.toLowerCase()));
  const lower = loraName.toLowerCase();
  const search = "/" + lower; // string-start sentinel, mirrors the Python side
  for (const [hi, lo] of PAIR_TOKENS) {
    for (const [src, dst] of [[hi, lo], [lo, hi]]) {
      let idx = search.indexOf(src);
      while (idx !== -1) {
        const candidate = (search.slice(0, idx) + dst + search.slice(idx + src.length)).slice(1);
        if (candidate !== lower && lowerSet.has(candidate)) return true;
        idx = search.indexOf(src, idx + 1);
      }
    }
  }
  return false;
}

function baseName(path) {
  const cut = path.lastIndexOf("/");
  return cut === -1 ? path : path.slice(cut + 1);
}

function clampStrength(v) {
  if (!Number.isFinite(v)) return 0;
  return Math.min(STRENGTH_MAX, Math.max(STRENGTH_MIN, Math.round(v * 100) / 100));
}

function isRowValue(v) {
  return v && typeof v === "object" && typeof v.lora === "string";
}

function rowWidgets(node) {
  return node.widgets ? node.widgets.filter((w) => w.type === ROW_TYPE) : [];
}

function rowSuffix(row) {
  const i = row.name.indexOf("_");
  return i === -1 ? row.name : row.name.slice(i + 1);
}

function curveSocketName(row) {
  return `curve_${rowSuffix(row)}`;
}

function ensureCurveSocket(node, row) {
  const name = curveSocketName(row);
  if (!node.inputs?.some((inp) => inp.name === name)) {
    node.addInput(name, "CCN_CURVE");
    node.setSize(node.computeSize());
  }
}

function removeCurveSocket(node, row) {
  const name = curveSocketName(row);
  const idx = node.inputs?.findIndex((inp) => inp.name === name);
  if (idx != null && idx >= 0) {
    node.removeInput(idx);
    node.setSize(node.computeSize());
  }
}

function curveConnected(node, row) {
  return node.inputs?.some(
    (inp) => inp.name === curveSocketName(row) && inp.link != null);
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
      mode: "static",
      lora: "None",
      strength_high: 1.0,
      strength_low: 1.0,
      ...(value ?? {}),
      id,
    },
    _hit: {},
    _drag: null,

    computeSize(width) {
      return [width ?? 200, LiteGraph.NODE_WIDGET_HEIGHT];
    },

    serializeValue() {
      return this.value;
    },

    draw(ctx, drawNode, width, y, height) {
      const margin = 12;
      const midY = y + height / 2;
      const on = !!this.value.on;
      ctx.save();

      ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
      ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
      ctx.beginPath();
      ctx.roundRect(margin, y + 1, width - margin * 2, height - 2, height * 0.3);
      ctx.fill();
      ctx.stroke();

      // Toggle dot.
      const toggleX = margin + 11;
      ctx.fillStyle = on ? "#7f7" : "#666";
      ctx.beginPath();
      ctx.arc(toggleX, midY, 4.5, 0, Math.PI * 2);
      ctx.fill();
      this._hit.toggle = [margin, margin + 22];

      // Mode chip: [B]aked patches the models, [L]ane feeds lora_lanes.
      const isLane = this.value.mode === "lane";
      const chipX0 = margin + 24, chipX1 = chipX0 + 16;
      ctx.fillStyle = isLane ? "#7060c0" : "#3a3a4a";
      ctx.beginPath();
      ctx.roundRect(chipX0, y + 4, chipX1 - chipX0, height - 8, 3);
      ctx.fill();
      ctx.fillStyle = on ? "#ddd" : "#888";
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(isLane ? "L" : "B", (chipX0 + chipX1) / 2, midY);
      this._hit.mode = [chipX0 - 2, chipX1 + 2];

      // Remove [x] at far right.
      const removeX1 = width - margin - 4;
      const removeX0 = removeX1 - 14;
      ctx.fillStyle = "#a66";
      ctx.font = "11px Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("\u2715", (removeX0 + removeX1) / 2, midY);
      this._hit.remove = [removeX0 - 2, removeX1 + 2];

      // Strength fields, right-aligned before the remove button.
      ctx.font = "12px Arial";
      const fieldW = 52;
      const lowX1 = removeX0 - 6;
      const lowX0 = lowX1 - fieldW;
      const highX1 = lowX0 - 4;
      const highX0 = highX1 - fieldW;
      const textColor = on
        ? LiteGraph.WIDGET_TEXT_COLOR
        : LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;

      ctx.textAlign = "left";
      ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
      ctx.fillText("H", highX0, midY);
      ctx.fillStyle = textColor;
      ctx.fillText(this.value.strength_high.toFixed(2), highX0 + 12, midY);
      ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
      ctx.fillText("L", lowX0, midY);
      ctx.fillStyle = textColor;
      ctx.fillText(this.value.strength_low.toFixed(2), lowX0 + 12, midY);
      this._hit.high = [highX0 - 2, highX1];
      this._hit.low = [lowX0 - 2, lowX1];

      // Name + pair badge fill the remaining middle.
      const nameX0 = margin + 46;
      const nameX1 = highX0 - 8;
      let display = this.value.lora === "None" ? "click to choose\u2026" : baseName(this.value.lora);
      if (this.value.lora !== "None" && hasPartner(this.value.lora)) {
        display = "\u21c4 " + display; // pair badge
      }
      if (isLane && curveConnected(drawNode, this)) {
        display = "~ " + display; // curve attached
      }
      ctx.fillStyle = this.value.lora === "None"
        ? LiteGraph.WIDGET_SECONDARY_TEXT_COLOR
        : textColor;
      const maxW = nameX1 - nameX0;
      while (display.length > 4 && ctx.measureText(display).width > maxW) {
        display = display.slice(0, -2);
      }
      ctx.fillText(display, nameX0, midY);
      this._hit.name = [nameX0 - 4, nameX1];

      ctx.restore();
    },

    mouse(event, pos, mouseNode) {
      const x = pos[0];
      const type = event.type;
      const inZone = (zone) => zone && x >= zone[0] && x <= zone[1];

      if (type === "pointerdown" || type === "mousedown") {
        if (inZone(this._hit.toggle)) {
          this.value.on = !this.value.on;
          mouseNode.setDirtyCanvas(true, true);
          return true;
        }
        if (inZone(this._hit.mode)) {
          this.value.mode = this.value.mode === "lane" ? "static" : "lane";
          if (this.value.mode === "lane") ensureCurveSocket(mouseNode, this);
          else removeCurveSocket(mouseNode, this);
          mouseNode.setDirtyCanvas(true, true);
          return true;
        }
        if (inZone(this._hit.remove)) {
          const idx = mouseNode.widgets.indexOf(this);
          if (idx !== -1) {
            removeCurveSocket(mouseNode, this);
            mouseNode.widgets.splice(idx, 1);
            mouseNode.setSize(mouseNode.computeSize());
            mouseNode.setDirtyCanvas(true, true);
          }
          return true;
        }
        if (inZone(this._hit.high) || inZone(this._hit.low)) {
          const field = inZone(this._hit.high) ? "strength_high" : "strength_low";
          this._drag = { field, startX: x, startValue: this.value[field], moved: false };
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
          this.value[this._drag.field] = clampStrength(this._drag.startValue + delta * DRAG_SCALE);
          mouseNode.setDirtyCanvas(true, true);
        }
        return true;
      }

      if ((type === "pointerup" || type === "mouseup") && this._drag) {
        const drag = this._drag;
        this._drag = null;
        if (!drag.moved) {
          // Treat as a click: type an exact value.
          const current = this.value[drag.field];
          app.canvas.prompt(
            drag.field === "strength_high" ? "High strength" : "Low strength",
            current,
            (v) => {
              const parsed = parseFloat(v);
              if (Number.isFinite(parsed)) {
                this.value[drag.field] = clampStrength(parsed);
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
        chooserNode.setDirtyCanvas(true, true);
      });
    },
  };
  return widget;
}

function addRow(node, value) {
  const row = makeRow(node, value);
  const buttonIndex = node.widgets.findIndex((w) => w === node._ccnAddButton);
  if (buttonIndex === -1) {
    node.widgets.push(row);
  } else {
    node.widgets.splice(buttonIndex, 0, row);
  }
  node.setSize(node.computeSize());
  return row;
}

function ensureRowCount(node, count) {
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
  node.setSize(node.computeSize());
}

app.registerExtension({
  name: "CCN.LoraPairLoader",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      origOnNodeCreated?.apply(this, arguments);
      this._ccnRowCounter = 0;
      this.widgets = this.widgets ?? [];

      this._ccnAddButton = this.addWidget(
        "button",
        "\uFF0B Add LoRA",
        null,
        () => {
          addRow(this);
          this.setDirtyCanvas(true, true);
        }
      );
      this._ccnAddButton.options = this._ccnAddButton.options ?? {};
      this._ccnAddButton.options.serialize = false;

      addRow(this); // one starter row for discoverability
      fetchLoraList().then(() => this.setDirtyCanvas(true, true));
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
            mode: "static",
            lora: "None",
            strength_high: 1.0,
            strength_low: 1.0,
            ...rowValues[i],
            id,
          };
          rows[i].name = `lora_${id}`;
          maxId = Math.max(maxId, Number(id) || 0);
        }
        this._ccnRowCounter = Math.max(this._ccnRowCounter ?? 0, maxId);

        // Reconcile curve sockets AFTER links restore: lane rows get their
        // socket if a hand-edited workflow lost it; orphaned curve_* sockets
        // with no matching lane row are pruned.
        const wanted = new Set(
          rowWidgets(this)
            .filter((r) => r.value.mode === "lane")
            .map((r) => curveSocketName(r)));
        for (const r of rowWidgets(this)) {
          if (r.value.mode === "lane") ensureCurveSocket(this, r);
        }
        for (let i = (this.inputs?.length ?? 0) - 1; i >= 0; i--) {
          const nm = this.inputs[i].name;
          if (nm.startsWith("curve_") && !wanted.has(nm)) {
            this.removeInput(i);
          }
        }
      }
      this.setDirtyCanvas(true, true);
      return result;
    };
  },
});
