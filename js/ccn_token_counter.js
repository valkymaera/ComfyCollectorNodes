import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// On-node readout for CCN token counter nodes. The widget is created at
// node creation (not lazily on execution) so widgets_values serialization
// stays aligned and the last reading survives a workflow reload.

const NODE_TYPES = new Set(["CCN_TokenCounter", "CCN_ConditioningTokenCount"]);
const WIDGET_NAME = "token_info";

app.registerExtension({
    name: "CCN.TokenCounter",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_TYPES.has(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const w = ComfyWidgets.STRING(
                this,
                WIDGET_NAME,
                ["STRING", { multiline: true }],
                app
            ).widget;
            w.inputEl.readOnly = true;
            w.inputEl.style.opacity = 0.8;
            w.value = "";
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            const payload = message?.ccn_token_info;
            if (payload === undefined) return;

            const text = Array.isArray(payload) ? payload.join("\n") : String(payload);
            const w = this.widgets?.find((x) => x.name === WIDGET_NAME);
            if (w) {
                w.value = text;
            }
        };
    },
});
