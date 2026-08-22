// LoRA Merge & Save (CCN) -- dynamic row widget, shared with the Multi Loader.
//
// Rows serialize as widget values named "lora_<n>":
//   { on: bool, lora: string, strength: float, id: int }
// The Python side (lora_merge.py) harvests them from **kwargs and merges
// every enabled row into a single LoRA file. The row widget itself (chooser,
// strength drag, trigger chips) is imported from lora_multi_loader.js.

import { app } from "../../scripts/app.js";
import {
  addRow,
  ensureRowCount,
  fetchLoraEntries,
  fetchTriggers,
  isRowValue,
  resizeRows,
  rowLoraNames,
  rowWidgets,
} from "./lora_multi_loader.js";

const NODE_NAME = "CCN_LoraMergeSave";
const DTYPE_OPTIONS = ["fp16", "bf16", "fp32"];
const DEVICE_OPTIONS = ["auto", "gpu", "cpu"];

app.registerExtension({
  name: "CCN.LoraMergeSave",

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

        // Recover the scalar widgets by value type in case positional
        // application stuffed a row dict into them.
        const scalars = info.widgets_values.filter((v) => !isRowValue(v));
        const dtypeWidget = this.widgets.find((w) => w.name === "output_dtype");
        if (dtypeWidget) {
          const saved = scalars.find((v) => DTYPE_OPTIONS.includes(v));
          if (saved !== undefined) dtypeWidget.value = saved;
          else if (!DTYPE_OPTIONS.includes(dtypeWidget.value)) dtypeWidget.value = "fp16";
        }
        const deviceWidget = this.widgets.find((w) => w.name === "svd_device");
        if (deviceWidget) {
          const saved = scalars.find((v) => DEVICE_OPTIONS.includes(v));
          if (saved !== undefined) deviceWidget.value = saved;
          else if (!DEVICE_OPTIONS.includes(deviceWidget.value)) deviceWidget.value = "auto";
        }
        const prefixWidget = this.widgets.find((w) => w.name === "filename_prefix");
        if (prefixWidget) {
          const saved = scalars.find(
            (v) => typeof v === "string" && !DTYPE_OPTIONS.includes(v) &&
              !DEVICE_OPTIONS.includes(v));
          if (saved !== undefined) prefixWidget.value = saved;
          else if (typeof prefixWidget.value !== "string") {
            prefixWidget.value = "loras/CCN_merged_lora";
          }
        }
        const rankWidget = this.widgets.find((w) => w.name === "new_rank");
        if (rankWidget) {
          const saved = scalars.find(
            (v) => typeof v === "number" && Number.isFinite(v));
          if (saved !== undefined) rankWidget.value = saved;
          else if (!Number.isFinite(rankWidget.value)) rankWidget.value = 0;
        }
      }
      fetchTriggers(rowLoraNames(this)).then(() => resizeRows(this));
      this.setDirtyCanvas(true, true);
      return result;
    };
  },
});
