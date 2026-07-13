import { app } from "../../scripts/app.js";

/*
 * CCN Int Nodes — Widget Sync Extension
 *
 * After each execution, reads the ui output from BetterInt and
 * GatedIncrement nodes and updates the "value" widget to match.
 * This lets you pause a batch, see the current value, and resume.
 *

 */

const CCN_INT_NODES = ["CCN_BetterInt", "CCN_GatedIncrement"];

app.registerExtension({
    name: "CCN.IntNodes",

    nodeCreated(node) {
        if (!CCN_INT_NODES.includes(node.comfyClass)) return;

        const origOnExecuted = node.onExecuted;

        node.onExecuted = function (data) {
            if (origOnExecuted) origOnExecuted.call(this, data);

            // Sync the value widget from Python's ui output
            if (data?.output_value?.[0] != null) {
                const widget = this.widgets?.find((w) => w.name === "value");
                if (widget) {
                    widget.value = data.output_value[0];
                    widget.callback?.(widget.value);
                }
            }

            // For GatedIncrement: show counter in title if available
            if (data?.counter_display?.[0] != null) {
                this.title =
                    this.comfyClass === "CCN_GatedIncrement"
                        ? `Gated Increment (CCN) [${data.counter_display[0]}]`
                        : this.title;
            }

            // Redraw
            if (this.graph) {
                this.graph.setDirtyCanvas(true);
            }
        };
    },
});
