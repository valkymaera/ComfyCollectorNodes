import { app } from "../../scripts/app.js";

/*
 * Live preview hook for CCN_ImageCanvas.
 *
 * The node draws nothing of its own — it only exposes ccn_image, the CCN
 * preview convention, so a downstream canvas node's Load Preview (Cropped
 * Image, Image Inset, Image Stitch, Rotate Image) can pull the solid-color
 * canvas without executing the graph. The data URL is built on read from
 * the current width/height/color widget values, so the preview always
 * matches what the next run will output. Drawing-only fallbacks on
 * malformed values (1024 / black); Python validates at queue time.
 */

app.registerExtension({
    name: "CCN.ImageCanvas",

    async nodeCreated(node) {
        if (node.comfyClass !== "CCN_ImageCanvas") return;

        function widgetValue(name) {
            return node.widgets?.find((w) => w.name === name)?.value;
        }

        // Mirrors Python's self-clamp so the preview matches the output.
        function dim(name) {
            const v = Math.floor(Number(widgetValue(name)));
            return Number.isFinite(v) ? Math.max(1, Math.min(8192, v)) : 1024;
        }

        function fillColor() {
            const v = String(widgetValue("color") ?? "").trim();
            if (/^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.test(v)) {
                return v.startsWith("#") ? v : "#" + v;
            }
            return "#000000";
        }

        Object.defineProperty(node, "ccn_image", {
            configurable: true,
            get() {
                try {
                    const off = document.createElement("canvas");
                    off.width = dim("width");
                    off.height = dim("height");
                    const ctx = off.getContext("2d");
                    ctx.fillStyle = fillColor();
                    ctx.fillRect(0, 0, off.width, off.height);
                    return off.toDataURL("image/png");
                } catch (err) {
                    console.warn("[CCN ImageCanvas] ccn_image getter failed:", err);
                    return null;
                }
            },
        });
    },
});
