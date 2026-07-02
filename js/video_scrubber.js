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
            "display:flex;flex-direction:column;gap:4px;width:100%;" +
            "height:100%;box-sizing:border-box;padding:2px 0;";

        // upload button
        const uploadBtn = document.createElement("button");
        uploadBtn.textContent = "Upload Video";
        uploadBtn.style.cssText =
            "width:100%;flex:0 0 auto;padding:5px 0;cursor:pointer;" +
            "border:1px solid #555;background:#333;color:#ddd;" +
            "border-radius:4px;font-size:12px;";

        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "video/*,.gif";
        fileInput.style.display = "none";

        // preview image
        const previewImg = document.createElement("img");
        previewImg.style.cssText =
            "width:100%;flex:1 1 auto;min-height:0;display:block;" +
            "background:#111;border-radius:4px;object-fit:contain;";
        previewImg.draggable = false;

        // frame-step row: ◀ [slider] ▶ — arrows jump by the frame_step widget
        const sliderRow = document.createElement("div");
        sliderRow.style.cssText =
            "display:flex;align-items:center;gap:4px;flex:0 0 auto;";

        const arrowCss =
            "flex:0 0 auto;padding:2px 9px;cursor:pointer;line-height:1;" +
            "border:1px solid #555;background:#333;color:#ddd;" +
            "border-radius:4px;font-size:12px;";

        const prevBtn = document.createElement("button");
        prevBtn.textContent = "\u25C0";
        prevBtn.title = "Step back by frame_step";
        prevBtn.style.cssText = arrowCss;

        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = "0";
        slider.max = "0";
        slider.value = "0";
        slider.style.cssText =
            "flex:1 1 auto;min-width:0;margin:0;cursor:pointer;";

        const nextBtn = document.createElement("button");
        nextBtn.textContent = "\u25B6";
        nextBtn.title = "Step forward by frame_step";
        nextBtn.style.cssText = arrowCss;

        sliderRow.appendChild(prevBtn);
        sliderRow.appendChild(slider);
        sliderRow.appendChild(nextBtn);

        // frame counter label
        const label = document.createElement("div");
        label.style.cssText =
            "flex:0 0 auto;text-align:center;font-size:11px;" +
            "color:#999;user-select:none;";
        label.textContent = "No video loaded";

        // exact-frame button — decodes the current frame accurately on demand
        const exactBtn = document.createElement("button");
        exactBtn.textContent = "Load Exact Frame";
        exactBtn.style.cssText =
            "width:100%;flex:0 0 auto;padding:5px 0;cursor:pointer;" +
            "border:1px solid #555;background:#333;color:#ddd;" +
            "border-radius:4px;font-size:12px;";

        // clear-cache button — deletes all cached exact-frame PNGs from disk
        const clearBtn = document.createElement("button");
        clearBtn.textContent = "Clear Frame Cache";
        clearBtn.style.cssText =
            "width:100%;flex:0 0 auto;padding:5px 0;cursor:pointer;" +
            "border:1px solid #555;background:#333;color:#ddd;" +
            "border-radius:4px;font-size:12px;";

        container.appendChild(uploadBtn);
        container.appendChild(fileInput);
        container.appendChild(previewImg);
        container.appendChild(sliderRow);
        container.appendChild(label);
        container.appendChild(exactBtn);
        container.appendChild(clearBtn);

        // The preview reports only a MINIMUM height to ComfyUI via getMinHeight.
        // ComfyUI derives the node's min-size from that, while the element fills
        // the node's available height above it — so the preview scales up when
        // the node grows yet the node still resizes back down to the minimum.
        const MIN_BOX = 160;

        // insert as DOM widget right after the video combo
        const domWidget = node.addDOMWidget(
            "ccn_scrubber_ui",
            "div",
            container,
            { serialize: false, getMinHeight: () => MIN_BOX }
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

        // frame_step widget feeds the arrow buttons (UI-only; Python ignores it)
        const stepWidget = node.widgets.find((w) => w.name === "frame_step");

        // One path to land on a frame: clamp, sync slider + widget + label, fetch.
        function gotoFrame(val) {
            const maxIdx = totalFrames > 0 ? totalFrames - 1 : 0;
            const v = Math.max(0, Math.min(val, maxIdx));
            slider.value = String(v);
            scrubWidget.value = v;
            label.textContent = `Frame ${v} / ${totalFrames}`;
            debouncedFetch(lastFilename, v);
        }

        function stepFrames(dir) {
            const step = Math.max(1, parseInt(stepWidget?.value ?? 1) || 1);
            gotoFrame(parseInt(slider.value) + dir * step);
        }

        prevBtn.addEventListener("click", () => stepFrames(-1));
        nextBtn.addEventListener("click", () => stepFrames(1));

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
                // overwrite is appended first so the server route reads it
                // before the file part it streams to disk
                const body = new FormData();
                body.append("overwrite", "true");
                body.append("image", file);

                // Custom streaming route, not /upload/image: the stock endpoint
                // buffers the whole body in RAM and is capped by the server's
                // max-upload-size, which rejects large videos.
                const resp = await api.fetchApi("/ccn/video_scrubber/upload", {
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
                    uploadBtn.textContent = "Upload Video";
                } else {
                    const detail = await resp.text().catch(() => "");
                    console.error(
                        "[CCN VideoScrubber] upload failed:",
                        resp.status, detail
                    );
                    uploadBtn.textContent = "Upload failed";
                    setTimeout(() => {
                        uploadBtn.textContent = "Upload Video";
                    }, 2500);
                }
            } catch (e) {
                console.error("[CCN VideoScrubber] upload:", e);
                uploadBtn.textContent = "Upload failed";
                setTimeout(() => {
                    uploadBtn.textContent = "Upload Video";
                }, 2500);
            } finally {
                uploadBtn.disabled = false;
                fileInput.value = "";
            }
        });

        // ---------------------------------------------------------------
        //  Load Exact Frame
        // ---------------------------------------------------------------

        // Decode the current frame accurately (server-side, threaded) and
        // show that cached PNG instead of the fast keyframe preview. The crop
        // node's pull reads whatever <img> we display, so after this press it
        // pulls the exact frame for free. Any scrub afterward re-fetches a fast
        // frame and overwrites the preview, reverting to fast automatically.
        exactBtn.addEventListener("click", async () => {
            if (!lastFilename || totalFrames <= 0) return;
            const idx = parseInt(slider.value);

            exactBtn.textContent = "Decoding…";
            exactBtn.disabled = true;
            try {
                const url =
                    `/ccn/video_scrubber/exact` +
                    `?filename=${encodeURIComponent(lastFilename)}` +
                    `&frame=${idx}`;
                const resp = await api.fetchApi(url);
                if (!resp.ok) {
                    console.error(
                        "[CCN VideoScrubber] exact frame failed:", resp.status
                    );
                    return;
                }
                const info = await resp.json();
                // Switching from a blob preview to an on-disk /view URL — drop
                // the old blob so it isn't leaked.
                releaseBlobUrl();
                previewImg.src = api.apiURL(
                    `/view?filename=${encodeURIComponent(info.filename)}` +
                    `&type=${info.type}` +
                    `&subfolder=${encodeURIComponent(info.subfolder || "")}`
                );
                label.textContent =
                    `Frame ${info.frame} / ${totalFrames} · exact`;
            } catch (e) {
                console.error("[CCN VideoScrubber] exact frame:", e);
            } finally {
                exactBtn.textContent = "Load Exact Frame";
                exactBtn.disabled = false;
            }
        });

        clearBtn.addEventListener("click", async () => {
            // Frames are re-decoded on demand, so this is non-destructive — but
            // confirm anyway to guard against a mid-scrub misclick.
            const ok = window.confirm(
                "Clear all cached Video Scrubber frames?\n\n" +
                "They are re-decoded on demand, so nothing is lost permanently."
            );
            if (!ok) return;

            clearBtn.textContent = "Clearing\u2026";
            clearBtn.disabled = true;
            try {
                const resp = await api.fetchApi(
                    "/ccn/video_scrubber/clear_cache", { method: "POST" }
                );
                if (!resp.ok) {
                    console.error(
                        "[CCN VideoScrubber] clear cache failed:", resp.status
                    );
                    label.textContent = "Cache clear failed";
                    return;
                }
                const info = await resp.json();
                const mb = (info.bytes_freed || 0) / (1024 * 1024);
                label.textContent = info.cleared
                    ? `Cleared ${info.cleared} frame(s) \u00b7 ${mb.toFixed(1)} MB freed`
                    : "Frame cache already empty";
            } catch (e) {
                console.error("[CCN VideoScrubber] clear cache:", e);
                label.textContent = "Cache clear failed";
            } finally {
                clearBtn.textContent = "Clear Frame Cache";
                clearBtn.disabled = false;
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

        // sensible starting footprint so the preview is usefully large
        node.size[0] = Math.max(node.size[0], 300);
        node.size[1] = Math.max(node.size[1] || 0, 420);

        // initial load if a video is already selected (e.g. duplicated node)
        if (videoWidget.value) {
            requestAnimationFrame(() => onVideoChange(videoWidget.value));
        }
    },
});
