import { app } from "../../scripts/app.js";

// Live section labels for Prompt Store Custom: when a Prompt Store Headings
// node is wired into the "headings" input, its heading values are shown as
// the labels of the five section widgets and output slots. Cosmetic only —
// the server resolves the same headings at execution time. Widget/output
// .name is never touched (serialization and link binding depend on it).

const SECTION_COUNT = 5;

function resolveUpstream(node, slot) {
    // Preferred: LiteGraph's built-in accessor
    if (typeof node.getInputNode === "function") {
        try {
            const src = node.getInputNode(slot);
            if (src) return src;
        } catch (_) {}
    }

    // Fallback: resolve via the graph containing this node
    const linkId = node.inputs[slot].link;
    if (linkId == null) return null;
    const graph = node.graph || app.graph;
    const link = graph?.links?.[linkId]
              ?? (graph?.links?.get && graph.links.get(linkId));
    if (!link) return null;
    return graph.getNodeById(link.origin_id);
}

function syncLabels(node) {
    const slot = node.inputs?.findIndex((i) => i.name === "headings");
    if (slot == null || slot < 0) return;

    let headings = null;
    if (node.inputs[slot].link != null) {
        const srcNode = resolveUpstream(node, slot);
        const srcType = srcNode?.type ?? srcNode?.comfyClass;
        if (srcType === "CCN_PromptStoreHeadings") {
            headings = [];
            for (let i = 0; i < SECTION_COUNT; i++) {
                const w = srcNode.widgets?.find((x) => x.name === `heading_${i + 1}`);
                headings.push(typeof w?.value === "string" ? w.value.trim() : "");
            }
        }
    }

    for (let i = 0; i < SECTION_COUNT; i++) {
        const label = headings?.[i] || null;
        const widget = node.widgets?.find((w) => w.name === `section_${i + 1}`);
        if (widget) {
            if (label) widget.label = label;
            else delete widget.label;
        }
        // Output 0 is the joined prompt; sections start at 1.
        const output = node.outputs?.[i + 1];
        if (output) {
            if (label) output.label = label;
            else delete output.label;
        }
    }

    node.setDirtyCanvas(true, true);
}

function pushToConsumers(node) {
    const graph = node.graph || app.graph;
    for (const linkId of node.outputs?.[0]?.links ?? []) {
        const link = graph?.links?.[linkId]
                  ?? (graph?.links?.get && graph.links.get(linkId));
        if (!link) continue;
        const target = graph.getNodeById(link.target_id);
        const targetType = target?.type ?? target?.comfyClass;
        if (targetType === "CCN_PromptStoreCustom") syncLabels(target);
    }
}

app.registerExtension({
    name: "CCN.PromptStoreCustom",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "CCN_PromptStoreCustom") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated?.apply(this, arguments);
                // Deferred: clones copy serialized labels but not links, so a
                // post-frame sync strips labels that no longer have a source.
                requestAnimationFrame(() => syncLabels(this));
                return r;
            };

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function () {
                const r = onConnectionsChange?.apply(this, arguments);
                syncLabels(this);
                return r;
            };

            // Loaded workflow: the upstream headings node's widget values are
            // restored during graph.configure, possibly after this node, so
            // defer the sync until the whole load pass has finished.
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure?.apply(this, arguments);
                requestAnimationFrame(() => syncLabels(this));
                return r;
            };
        }

        if (nodeData.name === "CCN_PromptStoreHeadings") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated?.apply(this, arguments);
                const node = this;

                // Push label updates to connected stores as headings change.
                for (let i = 0; i < SECTION_COUNT; i++) {
                    const w = node.widgets?.find((x) => x.name === `heading_${i + 1}`);
                    if (!w) continue;
                    const cb = w.callback;
                    w.callback = function () {
                        const ret = cb?.apply(this, arguments);
                        pushToConsumers(node);
                        return ret;
                    };
                }
                return r;
            };
        }
    },
});
