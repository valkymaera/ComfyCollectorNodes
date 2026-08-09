"""
Video Scrubber - Video loader with in-node frame preview and scrubbing.

Registers API routes so the frontend scrubber widget can fetch frame
thumbnails (/frame), video metadata (/info), on-demand exact frames
(/exact), and upload large videos (/upload) without running the full node.

The /frame route is the fast scrubbing preview: a persistent per-video
decoder steps forward from the last shown frame (keyframe-seeking on
backward jumps), downsized to a JPEG. The /exact route does a
frame-accurate sequential decode of a single frame, caches it full-res as
a PNG, and hands it to both the preview
and the node at execution time so the displayed frame and the rendered frame
are the same one. Both routes offload decoding to a thread so the server
stays responsive during scrubbing and long seeks.
"""

import os
import re
import io
import sys
import asyncio
import threading
import numpy as np
import torch
from PIL import Image
import folder_paths

from server import PromptServer
from aiohttp import web

# PyAV is part of ComfyUI's own requirements; the guard only matters on
# installs old enough to predate that, where the pack should still load.
try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False


VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif')

AV_MISSING_ERROR = (
    "PyAV (av) is required for VideoScrubber. It ships with ComfyUI; "
    "update ComfyUI or install with: pip install av"
)

# Subfolder under the input directory where exact-frame PNGs are cached.
# In the input dir (not temp) for now so the files are easy to inspect,
# archive by moving into the input root, or delete by hand during dev.
CACHE_SUBFOLDER = "Video Scrubber Frames"


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


def _display_dims(width, height, sar):
    """
    Convert coded dimensions to display dimensions via the stream's sample
    aspect ratio. Anamorphic videos (SAR != 1) store non-square pixels that
    players scale at display time; upscale the short side so no coded
    resolution is discarded, matching the player/ffmpeg convention.
    """
    if not sar or sar <= 0 or sar == 1:
        return width, height
    if sar > 1:
        return round(width * sar), height
    return width, round(height / sar)


def _get_video_info(filepath):
    """Extract frame count, dimensions, and FPS from a video file."""
    try:
        with av.open(filepath) as container:
            if not container.streams.video:
                return 0, 0, 0, 0.0
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            total = stream.frames
            if not total and fps > 0:
                # Some containers (webm, gif) don't declare a frame count;
                # estimate it from the duration instead.
                if stream.duration and stream.time_base:
                    total = int(float(stream.duration * stream.time_base) * fps)
                elif container.duration:
                    total = int(container.duration / av.time_base * fps)
            width, height = _display_dims(
                stream.width, stream.height, stream.sample_aspect_ratio
            )
            return total, width, height, fps
    except (av.FFmpegError, OSError, ValueError):
        return 0, 0, 0, 0.0


def _decode_frame_at(container, stream, frame_idx):
    """
    Keyframe-seek toward frame_idx and decode forward to the target frame.
    Returns the decoded VideoFrame, or None when the position is unreachable.
    """
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    if fps <= 0 or not stream.time_base:
        return None
    target_pts = int(frame_idx / fps / stream.time_base)
    try:
        container.seek(target_pts, backward=True, stream=stream)
    except (av.FFmpegError, OSError):
        container.seek(0)
    frame = None
    for f in container.decode(stream):
        frame = f
        if f.pts is not None and f.pts >= target_pts:
            break
    return frame


def _extract_frame_jpeg(filepath, frame_idx, max_dim=512):
    """Extract a single frame as JPEG bytes, downsized for preview thumbnails."""
    try:
        with av.open(filepath) as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            # SAR is stream metadata, so it must be captured while the
            # container is open; the decoded frame outlives it.
            sar = stream.sample_aspect_ratio
            frame = _decode_frame_at(container, stream, frame_idx)
    except (av.FFmpegError, OSError, ValueError):
        return None
    if frame is None:
        return None

    img = frame.to_image()
    w, h = _display_dims(*img.size, sar)
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        w, h = int(w * scale), int(h * scale)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _extract_frame_rgb(filepath, frame_idx, accurate=False):
    """
    Decode a single frame as an RGB uint8 array (H, W, C), or None on failure.

    accurate=False seeks to the nearest keyframe and reads forward to it —
    fast and bounded by the keyframe interval, but frame accuracy is
    codec-dependent. accurate=True decodes sequentially from the start,
    which is frame-exact but costs one decode per frame up to frame_idx.
    """
    try:
        with av.open(filepath) as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            sar = stream.sample_aspect_ratio
            if accurate:
                current = -1
                frame = None
                for f in container.decode(stream):
                    current += 1
                    frame = f
                    if current >= frame_idx:
                        break
                # Overshot the real end (e.g. an overestimated frame count)
                if frame is None or current < frame_idx:
                    return None
            else:
                frame = _decode_frame_at(container, stream, frame_idx)
                if frame is None:
                    return None
    except (av.FFmpegError, OSError, ValueError):
        return None

    rgb = frame.to_ndarray(format="rgb24")
    h, w = rgb.shape[:2]
    dw, dh = _display_dims(w, h, sar)
    if (dw, dh) != (w, h):
        # Apply the SAR here so every consumer — the node's IMAGE output and
        # the exact-frame cache PNGs — gets square-pixel display dimensions.
        rgb = np.array(Image.fromarray(rgb).resize((dw, dh), Image.LANCZOS))
    return rgb


# ---------------------------------------------------------------------------
#  Scrub sessions — persistent decoders for responsive sequential preview
# ---------------------------------------------------------------------------

# Beyond this forward distance a keyframe seek likely beats sequential decode.
_SCRUB_FORWARD_WINDOW = 240
_SCRUB_MAX_SESSIONS = 4

_scrub_sessions = {}  # normalized path -> _ScrubSession (insertion order = LRU)
_scrub_sessions_lock = threading.Lock()


class _ScrubSession:
    """
    A persistently open decoder for one video, reused across /frame requests.

    Re-opening the container and decoding forward from a keyframe costs up to
    a full GOP per request; keeping the decoder open means scrubbing forward
    only decodes the frames between the previous position and the target.
    Backward or long jumps keyframe-seek on the same open container.
    """

    def __init__(self, filepath):
        self.mtime = os.path.getmtime(filepath)
        self.container = av.open(filepath)
        if not self.container.streams.video:
            self.container.close()
            raise ValueError(f"no video stream in {filepath}")
        self.stream = self.container.streams.video[0]
        self.sar = self.stream.sample_aspect_ratio
        self.fps = (
            float(self.stream.average_rate) if self.stream.average_rate else 0.0
        )
        self.time_base = self.stream.time_base
        # Index of the next frame decode() will yield; None right after a seek.
        self.next_idx = 0
        self.lock = threading.Lock()
        self.last_idx = None
        self.last_jpeg = None

    def close(self):
        try:
            self.container.close()
        except Exception:
            pass

    def frame_jpeg(self, frame_idx, max_dim):
        """Decode frame_idx and return preview JPEG bytes (caller holds lock)."""
        if frame_idx == self.last_idx and self.last_jpeg is not None:
            return self.last_jpeg
        frame = self._decode_to(frame_idx)
        if frame is None:
            return None
        w, h = _display_dims(frame.width, frame.height, self.sar)
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            w, h = int(w * scale), int(h * scale)
        # One swscale call does SAR-corrected scaling + RGB conversion —
        # far faster than to_image() + LANCZOS and ample for a preview.
        img = frame.reformat(width=w, height=h, format="rgb24").to_image()
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        self.last_idx = frame_idx
        self.last_jpeg = buf.getvalue()
        return self.last_jpeg

    def _frame_index(self, frame):
        """Best-effort index of a decoded frame; None when underivable."""
        if frame.pts is None or self.fps <= 0 or not self.time_base:
            return None
        return round(float(frame.pts * self.time_base) * self.fps)

    def _decode_to(self, frame_idx, _retried=False):
        walk = (
            self.next_idx is not None
            and self.next_idx <= frame_idx < self.next_idx + _SCRUB_FORWARD_WINDOW
        )
        if not walk:
            if self.fps <= 0 or not self.time_base:
                return None
            target_pts = int(frame_idx / self.fps / self.time_base)
            try:
                self.container.seek(
                    target_pts, backward=True, stream=self.stream
                )
            except (av.FFmpegError, OSError):
                self.container.seek(0)
            self.next_idx = None

        frame = None
        try:
            for f in self.container.decode(self.stream):
                frame = f
                idx = self._frame_index(f)
                if idx is None:
                    # No usable pts — count on from the last known index.
                    idx = (
                        self.next_idx if self.next_idx is not None
                        else frame_idx
                    )
                self.next_idx = idx + 1
                if idx >= frame_idx:
                    break
        except av.EOFError:
            # Feeding a drained decoder raises; a seek resets it below.
            frame = None

        if frame is None and not _retried:
            # Decoder exhausted (an earlier request ran to EOF) — retry once
            # via a fresh seek.
            self.next_idx = None
            return self._decode_to(frame_idx, _retried=True)
        return frame


def _session_key(filepath):
    return os.path.normcase(os.path.abspath(filepath))


def _get_scrub_session(filepath):
    """Return a cached-or-new session for filepath, LRU-capped, mtime-checked."""
    key = _session_key(filepath)
    mtime = os.path.getmtime(filepath)
    with _scrub_sessions_lock:
        session = _scrub_sessions.pop(key, None)
        if session is not None and session.mtime != mtime:
            with session.lock:
                session.close()
            session = None
        if session is None:
            session = _ScrubSession(filepath)
        _scrub_sessions[key] = session  # re-insert = most recently used
        while len(_scrub_sessions) > _SCRUB_MAX_SESSIONS:
            oldest = _scrub_sessions.pop(next(iter(_scrub_sessions)))
            with oldest.lock:
                oldest.close()
        return session


def _drop_scrub_session(filepath):
    """Close and forget the session for filepath (frees its file handle)."""
    with _scrub_sessions_lock:
        session = _scrub_sessions.pop(_session_key(filepath), None)
    if session is not None:
        with session.lock:
            session.close()


def _scrub_frame_jpeg(filepath, frame_idx, max_dim=512):
    """
    Preview JPEG via the persistent session, falling back to a stateless
    decode when the session path fails so one bad state never sticks.
    """
    try:
        session = _get_scrub_session(filepath)
        with session.lock:
            data = session.frame_jpeg(frame_idx, max_dim)
        if data is not None:
            return data
    except (av.FFmpegError, EOFError, OSError, ValueError):
        pass
    _drop_scrub_session(filepath)
    return _extract_frame_jpeg(filepath, frame_idx, max_dim)


# ---------------------------------------------------------------------------
#  Exact-frame cache — full-res PNGs keyed by source name + frame index
# ---------------------------------------------------------------------------

def _cache_dir():
    """Return the cache directory, creating it if needed."""
    path = os.path.join(folder_paths.get_input_directory(), CACHE_SUBFOLDER)
    os.makedirs(path, exist_ok=True)
    return path


def _cache_filename(video, frame_idx):
    """
    Build a human-readable, sortable cache filename for one frame.
    The video stem is reduced to ASCII word characters, keeping the names
    tidy, sortable, and safe in /view URLs regardless of the source name.
    Two source names that normalize identically would collide, which is
    unlikely with distinct files in a flat input folder.
    """
    stem = os.path.splitext(os.path.basename(video))[0]
    safe = re.sub(r'[^A-Za-z0-9_-]', '_', stem)
    return f"{safe}_{frame_idx:06d}.png"


def _cache_is_valid(cache_path, video_path):
    """
    A cache PNG is valid only if it exists and is newer than its source video.

    Comparing mtimes (rather than baking one into the filename) keeps names
    readable while still invalidating caches when a clip is re-exported under
    the same name — the new video's mtime jumps ahead of the stale PNGs.
    """
    if not os.path.isfile(cache_path):
        return False
    try:
        return os.path.getmtime(cache_path) >= os.path.getmtime(video_path)
    except OSError:
        return False


def _extract_and_cache(filepath, frame_idx, cache_path):
    """
    Decode one frame accurately and write it full-res to cache_path.
    Runs in a thread executor (blocking decode). Returns True on success.
    """
    rgb = _extract_frame_rgb(filepath, frame_idx, accurate=True)
    if rgb is None:
        return False
    # Frame is already RGB here, so PIL writes it directly — matching the
    # still-image convention used elsewhere in the pack (no BGR juggling).
    Image.fromarray(rgb).save(cache_path, compress_level=4)
    return True


# ---------------------------------------------------------------------------
#  API routes — serve frame thumbnails and video metadata to the frontend
# ---------------------------------------------------------------------------

@PromptServer.instance.routes.get("/ccn/video_scrubber/frame")
async def _api_frame(request):
    """Serve a single video frame as JPEG for the scrubber preview."""
    if not HAS_AV:
        return web.Response(status=500, text=AV_MISSING_ERROR)

    filename = request.query.get("filename", "")
    frame_idx = int(request.query.get("frame", "0"))

    filepath = _resolve_video_path(filename)
    if not filepath:
        return web.Response(status=404, text="Video not found")

    # Offload the decode to a thread so scrubbing never blocks the server's
    # event loop; the session keeps sequential forward scrubs near-instant.
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(
        None, _scrub_frame_jpeg, filepath, frame_idx
    )
    if data is None:
        return web.Response(status=500, text="Frame extraction failed")

    return web.Response(body=data, content_type="image/jpeg")


@PromptServer.instance.routes.get("/ccn/video_scrubber/info")
async def _api_info(request):
    """Serve video metadata (frame count, dimensions, fps)."""
    if not HAS_AV:
        return web.json_response({"error": AV_MISSING_ERROR}, status=500)

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


@PromptServer.instance.routes.get("/ccn/video_scrubber/exact")
async def _api_exact(request):
    """
    Frame-accurate decode of a single frame, cached full-res as a PNG.

    Returns the cached file's /view coordinates (filename/subfolder/type) so
    the frontend can show it and the crop node can pull it. A valid existing
    cache is reused, so re-pressing on the same frame costs nothing. The decode
    is offloaded to a thread so a long sequential seek doesn't stall the server.
    """
    if not HAS_AV:
        return web.json_response({"error": AV_MISSING_ERROR}, status=500)

    filename = request.query.get("filename", "")
    frame_idx = int(request.query.get("frame", "0"))

    filepath = _resolve_video_path(filename)
    if not filepath:
        return web.json_response({"error": "not found"}, status=404)

    total, _, _, _ = _get_video_info(filepath)
    if total <= 0:
        return web.json_response({"error": "no frames"}, status=500)
    frame_idx = max(0, min(frame_idx, total - 1))

    cache_name = _cache_filename(filename, frame_idx)
    cache_path = os.path.join(_cache_dir(), cache_name)

    if not _cache_is_valid(cache_path, filepath):
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(
            None, _extract_and_cache, filepath, frame_idx, cache_path
        )
        if not ok:
            return web.json_response(
                {"error": "extraction failed"}, status=500
            )

    return web.json_response({
        "filename": cache_name,
        "subfolder": CACHE_SUBFOLDER,
        "type": "input",
        "frame": frame_idx,
    })


def _unique_input_path(input_dir, filename):
    """Return a non-colliding path in input_dir, adding a numeric suffix."""
    candidate = os.path.join(input_dir, filename)
    if not os.path.exists(candidate):
        return candidate
    stem, ext = os.path.splitext(filename)
    i = 1
    while os.path.exists(os.path.join(input_dir, f"{stem}_{i}{ext}")):
        i += 1
    return os.path.join(input_dir, f"{stem}_{i}{ext}")


@PromptServer.instance.routes.post("/ccn/video_scrubber/upload")
async def _api_upload(request):
    """
    Stream a video upload to the input directory.

    ComfyUI's stock /upload/image reads the whole body into memory and is
    capped by the server's max-upload-size (100 MB by default), so large
    videos fail there with a 413. This route lifts the per-request cap and
    writes the file to disk in chunks, so neither file size nor available RAM
    is a constraint.
    """
    # Lift aiohttp's per-request body size cap — the app-wide limit that
    # otherwise rejects large videos. Must be set before the body is read.
    # Older aiohttp only boolean-checked this in read()/post(), where 0 meant
    # "no limit"; aiohttp >= 3.13 forwards it into MultipartReader as an
    # absolute byte cap, where 0 rejects every part with 413. sys.maxsize
    # disables the cap under both behaviors.
    request._client_max_size = sys.maxsize

    try:
        reader = await request.multipart()
        overwrite = False
        saved_name = None

        while True:
            part = await reader.next()
            if part is None:
                break

            # The frontend sends "overwrite" before the file part so this is
            # known by the time the filename is resolved below.
            if part.name == "overwrite":
                text = (await part.text()).strip().lower()
                overwrite = text in ("1", "true", "yes")
                continue

            if part.filename:
                # basename guards against path traversal in the supplied name
                filename = os.path.basename(part.filename)
                if not filename:
                    continue
                input_dir = folder_paths.get_input_directory()
                target = os.path.join(input_dir, filename)
                if os.path.exists(target) and not overwrite:
                    target = _unique_input_path(input_dir, filename)
                # Close any scrub session holding this file open — on Windows
                # an open read handle would fail the overwrite.
                _drop_scrub_session(target)
                with open(target, "wb") as out:
                    while True:
                        chunk = await part.read_chunk(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                saved_name = os.path.basename(target)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

    if not saved_name:
        return web.json_response({"error": "no file received"}, status=400)

    return web.json_response({"name": saved_name})


@PromptServer.instance.routes.post("/ccn/video_scrubber/clear_cache")
async def _api_clear_cache(request):
    """
    Delete cached exact-frame PNGs.

    With no 'video' query param, clears every cached frame; with ?video=<name>,
    clears only that clip's frames (matched by the sanitized stem prefix that
    _cache_filename builds). Bounded to the cache subfolder: deletes only
    top-level .png files, never recurses, and never removes the folder itself
    or anything outside it.
    """
    cache_dir = _cache_dir()

    # Optional per-video scoping (the UI button clears all; the param is here
    # so a per-clip control can reuse this route later).
    video = request.query.get("video", "")
    prefix = None
    if video:
        stem = os.path.splitext(os.path.basename(video))[0]
        prefix = re.sub(r'[^A-Za-z0-9_-]', '_', stem) + "_"

    cleared = 0
    bytes_freed = 0
    try:
        for name in os.listdir(cache_dir):
            if not name.lower().endswith(".png"):
                continue
            if prefix is not None and not name.startswith(prefix):
                continue
            path = os.path.join(cache_dir, name)
            # Top-level plain files only — never descend into or remove dirs.
            if not os.path.isfile(path):
                continue
            try:
                size = os.path.getsize(path)
                os.remove(path)
                cleared += 1
                bytes_freed += size
            except OSError:
                # Skip anything locked or already gone; keep clearing the rest.
                continue
    except OSError as e:
        return web.json_response({"error": str(e)}, status=500)

    return web.json_response({"cleared": cleared, "bytes_freed": bytes_freed})


# ---------------------------------------------------------------------------
#  Node
# ---------------------------------------------------------------------------

class VideoScrubber:
    """
    Video loader that decodes a single frame at a chosen index.
    Browse frames visually via the scrubber widget, then queue to output
    just that frame as an IMAGE — for branching a clip into a new
    generation from a chosen split point.
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
                # UI-only stepping control read by the JS arrows. Declared so
                # ComfyUI's input plumbing passes it cleanly; the node ignores it.
                "frame_step": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "frame_index", "total_frames")
    FUNCTION = "load_video"

    def load_video(self, video, scrub_frame, frame_step=1):
        if not HAS_AV:
            raise ImportError(AV_MISSING_ERROR)

        filepath = _resolve_video_path(video)
        if not filepath:
            raise ValueError(f"Video not found: {video}")

        total_frames, _, _, _ = _get_video_info(filepath)
        if total_frames <= 0:
            raise ValueError(f"Could not read frame count for video: {video}")

        frame_index = max(0, min(scrub_frame, total_frames - 1))

        # If the Load Exact Frame button has cached this frame, load that PNG
        # directly — the accurate decode is already paid for. Otherwise do a
        # fast keyframe seek, matching what the live scrubber preview shows.
        cache_path = os.path.join(
            _cache_dir(), _cache_filename(video, frame_index)
        )
        if _cache_is_valid(cache_path, filepath):
            pil = Image.open(cache_path).convert("RGB")
            rgb = np.array(pil)
        else:
            rgb = _extract_frame_rgb(filepath, frame_index, accurate=False)

        if rgb is None:
            raise ValueError(
                f"Could not extract frame {frame_index} from {video} "
                f"(reported total {total_frames}). Try a lower index."
            )

        # ComfyUI IMAGE format: (B, H, W, C) float32 in [0, 1]
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)

        return (tensor, frame_index, total_frames)

    @classmethod
    def IS_CHANGED(cls, video, scrub_frame, frame_step=1):
        filepath = _resolve_video_path(video)
        if not filepath:
            return ""
        frame_index = scrub_frame
        cache_path = os.path.join(
            _cache_dir(), _cache_filename(video, frame_index)
        )
        # Fold the cache file's state in so creating (or invalidating) an exact
        # frame retriggers execution and the output tracks the cache.
        cache_state = (
            os.path.getmtime(cache_path)
            if _cache_is_valid(cache_path, filepath) else ""
        )
        return f"{filepath}:{os.path.getmtime(filepath)}:{scrub_frame}:{cache_state}"

    @classmethod
    def VALIDATE_INPUTS(cls, video, scrub_frame, frame_step=1):
        if not video:
            return "No video selected"
        return True
