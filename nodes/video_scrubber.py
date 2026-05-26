"""
Video Scrubber - Video loader with in-node frame preview and scrubbing.

Registers two lightweight API routes (/ccn/video_scrubber/frame and /info)
so the frontend scrubber widget can fetch individual frame thumbnails
without running the full node.
"""

import os
import numpy as np
import torch
import cv2
import folder_paths

from server import PromptServer
from aiohttp import web


VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif')


def _resolve_video_path(filename):
    """Resolve a video filename to its full filesystem path."""
    if not filename:
        return None
    # get_annotated_filepath handles subfolder annotations like "sub/file.mp4 [input]"
    filepath = folder_paths.get_annotated_filepath(filename)
    if filepath and os.path.isfile(filepath):
        return filepath
    fallback = os.path.join(folder_paths.get_input_directory(), filename)
    if os.path.isfile(fallback):
        return fallback
    return None


def _get_video_info(filepath):
    """Extract frame count, dimensions, and FPS from a video file."""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return 0, 0, 0, 0.0
    info = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        cap.get(cv2.CAP_PROP_FPS),
    )
    cap.release()
    return info


def _extract_frame_jpeg(filepath, frame_idx, max_dim=384):
    """Extract a single frame as JPEG bytes, downsized for preview thumbnails."""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None

    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(
            frame, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


# ---------------------------------------------------------------------------
#  API routes — serve frame thumbnails and video metadata to the frontend
# ---------------------------------------------------------------------------

@PromptServer.instance.routes.get("/ccn/video_scrubber/frame")
async def _api_frame(request):
    """Serve a single video frame as JPEG for the scrubber preview."""
    filename = request.query.get("filename", "")
    frame_idx = int(request.query.get("frame", "0"))

    filepath = _resolve_video_path(filename)
    if not filepath:
        return web.Response(status=404, text="Video not found")

    data = _extract_frame_jpeg(filepath, frame_idx)
    if data is None:
        return web.Response(status=500, text="Frame extraction failed")

    return web.Response(body=data, content_type="image/jpeg")


@PromptServer.instance.routes.get("/ccn/video_scrubber/info")
async def _api_info(request):
    """Serve video metadata (frame count, dimensions, fps)."""
    filename = request.query.get("filename", "")

    filepath = _resolve_video_path(filename)
    if not filepath:
        return web.json_response({"error": "not found"}, status=404)

    total, w, h, fps = _get_video_info(filepath)
    return web.json_response({
        "total_frames": total,
        "width": w,
        "height": h,
        "fps": fps,
    })


# ---------------------------------------------------------------------------
#  Node
# ---------------------------------------------------------------------------

class VideoScrubber:
    """
    Video loader with in-node frame scrubbing preview.
    Browse frames visually via the scrubber widget, then queue to output
    all frames, the selected frame, and before/after splits at the
    scrub point for temporal cropping.
    """

    CATEGORY = "ComfyCollectorNodes/Loaders"

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = sorted(
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and f.lower().endswith(VIDEO_EXTENSIONS)
        )

        return {
            "required": {
                "video": (files if files else [""], {}),
                "scrub_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("frames", "selected_frame", "frames_before", "frames_after", "frame_index", "total_frames")
    FUNCTION = "load_video"

    def load_video(self, video, scrub_frame):
        filepath = _resolve_video_path(video)
        if not filepath:
            raise ValueError(f"Video not found: {video}")

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            raise ValueError(f"No frames in video: {video}")

        total_frames = len(frames)
        frame_index = max(0, min(scrub_frame, total_frames - 1))

        # (B, H, W, C) float32 [0, 1] — standard ComfyUI IMAGE format
        all_tensor = torch.from_numpy(
            np.stack(frames, axis=0).astype(np.float32) / 255.0
        )
        selected = all_tensor[frame_index].unsqueeze(0)
        before = all_tensor[:frame_index + 1]      # start through scrub point (inclusive)
        after = all_tensor[frame_index:]            # scrub point through end (inclusive)

        return (all_tensor, selected, before, after, frame_index, total_frames)

    @classmethod
    def IS_CHANGED(cls, video, scrub_frame):
        filepath = _resolve_video_path(video)
        if filepath:
            return f"{filepath}:{os.path.getmtime(filepath)}:{scrub_frame}"
        return ""

    @classmethod
    def VALIDATE_INPUTS(cls, video, scrub_frame):
        if not video:
            return "No video selected"
        return True
