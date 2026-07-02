import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// name -> {mtime, size}, fetched lazily. Refetched only when the combo holds a
// name we haven't seen (i.e. new latents appeared after a ComfyUI refresh).
let metadataCache = null;
let metadataPromise = null;

async function fetchMetadata() {
    const resp = await api.fetchApi("/ccn/latent_filter/list");
    metadataCache = await resp.json();
    return metadataCache;
}

async function getMetadata(names) {
    if (metadataCache && !names.some((n) => !(n in metadataCache))) {
        return metadataCache;
    }
    // Coalesce concurrent callers into a single in-flight request.
    if (!metadataPromise) {
        metadataPromise = fetchMetadata().finally(() => { metadataPromise = null; });
    }
    return metadataPromise;
}

function compare(a, b, meta, sortBy) {
    if (sortBy === "name") {
        // Plain codepoint comparison to match the alphabetical order
        // INPUT_TYPES produces (Python's sorted()).
        return a < b ? -1 : a > b ? 1 : 0;
    }
    const ma = meta[a] || { mtime: 0, size: 0 };
    const mb = meta[b] || { mtime: 0, size: 0 };
    if (sortBy === "size") return ma.size - mb.size;
    return ma.mtime - mb.mtime; // date_modified
}

async function applySort(node) {
    const latentWidget = node.widgets?.find((w) => w.name === "latent");
    const sortByWidget = node.widgets?.find((w) => w.name === "sort_by");
    const sortOrderWidget = node.widgets?.find((w) => w.name === "sort_order");
    if (!latentWidget || !sortByWidget || !sortOrderWidget) return;

    const values = latentWidget.options?.values;
    // Nothing to sort with 0 or 1 options (also skips the "no files" sentinel,
    // which isn't in the metadata and would otherwise force endless refetches).
    if (!Array.isArray(values) || values.length <= 1) return;

    const sortBy = sortByWidget.value;
    const meta = (sortBy === "name") ? {} : await getMetadata(values);

    const sorted = [...values].sort((a, b) => compare(a, b, meta, sortBy));
    if (sortOrderWidget.value === "descending") sorted.reverse();

    // Reorder in place, keeping the same array reference. Only display order
    // changes; the selected value string is untouched, so it stays valid.
    values.length = 0;
    values.push(...sorted);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "CCN.LatentLoaderFiltered",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "CCN_LatentLoaderFiltered") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;

            for (const name of ["sort_by", "sort_order"]) {
                const w = node.widgets?.find((x) => x.name === name);
                if (!w) continue;
                const cb = w.callback;
                w.callback = function () {
                    const ret = cb?.apply(this, arguments);
                    applySort(node);
                    return ret;
                };
            }

            // Fresh drop: apply the default (newest-first) immediately.
            applySort(node);
            return r;
        };

        // Loaded workflow: widget values restore during configure (after
        // onNodeCreated), so re-sort here to honor the saved settings.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            applySort(this);
            return r;
        };
    },
});
