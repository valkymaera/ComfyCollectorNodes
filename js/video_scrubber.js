import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "CCN.VideoScrubber",

    async nodeCreated(node) {
        if (node.comfyClass !== "CCN_VideoScrubber") return;

        const videoWidget = node.widgets.find((w) => w.name === "video");
        const scrubWidget = node.widgets.find((w) => w.name === "scrub_frame");
        if (!videoWidget || !scrubWidget) return;

        // --- state ---
        let totalFrames = 0;
        let lastFilename = "";
        let debounceTimer = null;
        let blobUrl = null;

        function releaseBlobUrl() {
            if (blobUrl) {
                URL.revokeObjectURL(blobUrl);
                blobUrl = null;
            }
        }

        // ---------------------------------------------------------------
        //  Build the scrubber UI — single container becomes one DOM widget
        // ---------------------------------------------------------------
        const container = document.createElement("div");
        container.style.cssText =
            "display:flex;flex-direction:column;gap:4px;width:100%;padding:2px 0;";

        // upload button
        const uploadBtn = document.createElement("button");
        uploadBtn.textContent = "Upload Video";
        uploadBtn.style.cssText =
            "width:100%;padding:5px 0;cursor:pointer;" +
            "border:1px solid #555;background:#333;color:#ddd;" +
            "border-radius:4px;font-size:12px;";

        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "video/*,.gif";
        fileInput.style.display = "none";

        // preview image
        const previewImg = document.createElement("img");
        previewImg.style.cssText =
            "width:100%;display:block;background:#111;" +
            "border-radius:4px;min-height:20px;max-height:320px;" +
            "object-fit:contain;";
        previewImg.draggable = false;

        // range slider
        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = "0";
        slider.max = "0";
        slider.value = "0";
        slider.style.cssText = "width:100%;margin:0;cursor:pointer;";

        // frame counter label
        const label = document.createElement("div");
        label.style.cssText =
            "text-align:center;font-size:11px;color:#999;user-select:none;";
        label.textContent = "No video loaded";

        container.appendChild(uploadBtn);
        container.appendChild(fileInput);
        container.appendChild(previewImg);
        container.appendChild(slider);
        container.appendChild(label);

        // insert as DOM widget right after the video combo
        const domWidget = node.addDOMWidget(
            "ccn_scrubber_ui",
            "div",
            container,
            { serialize: false }
        );
        const widgets = node.widgets;
        const domIdx = widgets.indexOf(domWidget);
        if (domIdx > -1) {
            widgets.splice(domIdx, 1);
            const vidIdx = widgets.indexOf(videoWidget);
            widgets.splice(vidIdx + 1, 0, domWidget);
        }

        // ---------------------------------------------------------------
        //  Frame fetching
        // ---------------------------------------------------------------

        async function fetchFrame(filename, idx) {
            if (!filename) return;
            try {
                const url =
                    `/ccn/video_scrubber/frame` +
                    `?filename=${encodeURIComponent(filename)}` +
                    `&frame=${idx}`;
                const resp = await api.fetchApi(url);
                if (!resp.ok) return;
                releaseBlobUrl();
                const blob = await resp.blob();
                blobUrl = URL.createObjectURL(blob);
                previewImg.src = blobUrl;
            } catch (e) {
                console.error("[CCN VideoScrubber] frame fetch:", e);
            }
        }

        function debouncedFetch(filename, idx) {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => fetchFrame(filename, idx), 50);
        }

        // ---------------------------------------------------------------
        //  Video change — fetch metadata and first preview
        // ---------------------------------------------------------------

        async function onVideoChange(filename) {
            if (!filename) return;
            // avoid redundant fetches on repeated calls with same file
            if (filename === lastFilename && totalFrames > 0) {
                // just refresh the current frame in case slider moved
                debouncedFetch(filename, parseInt(slider.value));
                return;
            }
            lastFilename = filename;

            try {
                const url =
                    `/ccn/video_scrubber/info` +
                    `?filename=${encodeURIComponent(filename)}`;
                const resp = await api.fetchApi(url);
                if (!resp.ok) return;

                const info = await resp.json();
                totalFrames = info.total_frames || 0;
                const maxIdx = Math.max(0, totalFrames - 1);

                slider.max = String(maxIdx);
                scrubWidget.options.max = maxIdx;

                // clamp existing value
                const clamped = Math.min(
                    Math.max(0, scrubWidget.value),
                    maxIdx
                );
                slider.value = String(clamped);
                scrubWidget.value = clamped;

                const dim =
                    info.width && info.height
                        ? ` · ${info.width}×${info.height}`
                        : "";
                label.textContent =
                    `Frame ${clamped} / ${totalFrames}${dim}`;

                fetchFrame(filename, clamped);
            } catch (e) {
                console.error("[CCN VideoScrubber] info fetch:", e);
            }
        }

        // ---------------------------------------------------------------
        //  Event wiring
        // ---------------------------------------------------------------

        // slider scrub → update widget + fetch preview
        slider.addEventListener("input", () => {
            const val = parseInt(slider.value);
            scrubWidget.value = val;
            label.textContent = `Frame ${val} / ${totalFrames}`;
            debouncedFetch(lastFilename, val);
        });

        // scrub_frame widget typed/dragged → sync slider + fetch preview
        const origScrubCb = scrubWidget.callback;
        scrubWidget.callback = function (value) {
            if (origScrubCb) origScrubCb.call(this, value);
            const val = Math.max(
                0,
                Math.min(value, totalFrames > 0 ? totalFrames - 1 : 0)
            );
            slider.value = String(val);
            label.textContent = `Frame ${val} / ${totalFrames}`;
            debouncedFetch(lastFilename, val);
        };

        // video combo selection changed
        const origVideoCb = videoWidget.callback;
        videoWidget.callback = function (value) {
            if (origVideoCb) origVideoCb.call(this, value);
            lastFilename = "";          // reset so onVideoChange re-fetches info
            onVideoChange(value);
        };

        // ---------------------------------------------------------------
        //  Upload
        // ---------------------------------------------------------------

        uploadBtn.addEventListener("click", () => fileInput.click());

        fileInput.addEventListener("change", async () => {
            const file = fileInput.files?.[0];
            if (!file) return;

            uploadBtn.textContent = "Uploading…";
            uploadBtn.disabled = true;

            try {
                const body = new FormData();
                body.append("image", file);
                body.append("overwrite", "true");

                const resp = await api.fetchApi("/upload/image", {
                    method: "POST",
                    body,
                });

                if (resp.ok) {
                    const data = await resp.json();
                    const name = data.name;

                    // add to combo dropdown if not already present
                    if (name && videoWidget.options?.values) {
                        if (!videoWidget.options.values.includes(name)) {
                            videoWidget.options.values.push(name);
                            videoWidget.options.values.sort();
                        }
                    }

                    videoWidget.value = name;
                    lastFilename = "";
                    onVideoChange(name);
                }
            } catch (e) {
                console.error("[CCN VideoScrubber] upload:", e);
            } finally {
                uploadBtn.textContent = "Upload Video";
                uploadBtn.disabled = false;
                fileInput.value = "";
            }
        });

        // ---------------------------------------------------------------
        //  Lifecycle hooks
        // ---------------------------------------------------------------

        // restore preview when loading a saved workflow
        const origOnConfigure = node.onConfigure;
        node.onConfigure = function () {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            requestAnimationFrame(() => {
                if (videoWidget.value) {
                    lastFilename = "";
                    onVideoChange(videoWidget.value);
                }
            });
        };

        // clean up blob URLs on node removal
        const origOnRemoved = node.onRemoved;
        node.onRemoved = function () {
            releaseBlobUrl();
            clearTimeout(debounceTimer);
            if (origOnRemoved) origOnRemoved.apply(this, arguments);
        };

        // set a reasonable minimum width for the preview to be useful
        node.size[0] = Math.max(node.size[0], 300);

        // initial load if a video is already selected (e.g. duplicated node)
        if (videoWidget.value) {
            requestAnimationFrame(() => onVideoChange(videoWidget.value));
        }
    },
});
