"""
Simple Track -- multi-clip crossfade timeline.

Combines an arbitrary number of video clips (IMAGE batch + optional AUDIO
pairs on dynamic inputs video_<n>/audio_<n>) into one output video with
crossfade blending. The timeline widget (js/wip/simple_track.js) shows a
scrubbable composited preview, two parallel thumbnail tracks where dragged
clip overlaps define the crossfades, and a per-clip volume curve. Because
same-track overlap is disallowed by the widget, at most two clips cover any
frame; their smoothstep weights are exact complements, so the blend is
always fully opaque. Audio gets the identical fade multiplied by the
per-clip volume curve.

State travels in the track_data widget as JSON (schema v1):

    {"version": 1,
     "clips": [{"key": "1", "track": 0, "start": 0, "frames": 81,
                "curve": [{"x":0,"y":1,"in":0,"out":0,"mirrored":true}, ...]}]}

"key" is the input suffix, "start" is in timeline frames, "frames" is the
widget's last-known clip length (the actual tensor length wins at
execution), and "curve" is CCN curve keys mapping normalized clip time to
gain (clamped 0..1). Connected inputs missing from track_data are
auto-placed with the same formula the widget uses, so a never-opened node
and the widget agree; stale entries are dropped.

After each run the ui payload ("ccn_simple_track") returns the reconciled
clip list plus a low-res preview mp4 per clip, materialized into
temp/simple_track/ and content-hash cached. The widget scrubs those files
through the existing /ccn/video_scrubber routes, which is what lets
generated (in-memory) clips get pixel previews at all. Preview audio is
encoded raw -- no curve or fade baked in.
"""

import hashlib
import itertools
import json
import os
import re
import time
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional
import torchaudio

import folder_paths
import comfy.utils

from ..nodes.curve_cfg_guider import hermite
from ..nodes.video_scrubber import HAS_AV, _drop_scrub_session
from .lora_pair_loader import AnyType, FlexibleOptionalInputType

if HAS_AV:
    import av

_ANY = AnyType("*")

_PREVIEW_SUBFOLDER = "simple_track"
_PREVIEW_MAX_DIM = 512
# ComfyUI wipes temp at startup; the TTL sweep only matters on long-running
# servers. The utime-touch on every cache hit keeps live files young.
_PREVIEW_TTL = 24 * 3600
_PART_TTL = 3600
# Folded into the content hash so encoder-parameter changes invalidate.
_PREVIEW_VERSION = "v1"
_RESIZE_CHUNK = 32

_KEY_RE = re.compile(r"^video_(.+)$")

_part_counter = itertools.count()


def _preview_dir():
    path = os.path.join(folder_paths.get_temp_directory(), _PREVIEW_SUBFOLDER)
    os.makedirs(path, exist_ok=True)
    return path


def _wire_order(key):
    """Sort key for input suffixes: numeric order when numeric, else lexical."""
    return (0, int(key)) if key.isdigit() else (1, key)


def _smoothstep(u):
    u = u.clamp(0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _sanitize_curve(curve):
    """Validate curve keys from track_data; None means constant gain 1."""
    if not isinstance(curve, list) or not curve:
        return None
    keys = []
    for k in curve:
        if not isinstance(k, dict):
            return None
        try:
            keys.append({
                "x": float(k["x"]),
                "y": float(k["y"]),
                "in": float(k.get("in", 0.0)),
                "out": float(k.get("out", 0.0)),
            })
        except (KeyError, TypeError, ValueError):
            return None
    keys.sort(key=lambda k: k["x"])
    return keys


def _reconcile(track_data_str, videos, audios, default_overlap):
    """
    Merge the widget's track_data with the actually-connected inputs.

    Wire order drives iteration so auto-placement is deterministic. For known
    keys the tensor's frame count is truth (the JSON's is advisory -- the JS
    may have last seen a different upstream video). Connected inputs missing
    from the JSON are auto-placed with the exact formula the JS uses; JSON
    entries whose input is gone are dropped.
    """
    try:
        data = json.loads(track_data_str) if track_data_str else {}
        if not isinstance(data, dict):
            data = {}
    except (json.JSONDecodeError, TypeError):
        data = {}

    saved = {}
    for entry in data.get("clips", []) if isinstance(data.get("clips"), list) else []:
        if isinstance(entry, dict) and "key" in entry:
            saved[str(entry["key"])] = entry

    clips = []
    for key in sorted(videos, key=_wire_order):
        images = videos[key]
        frames = int(images.shape[0])
        if frames <= 0:
            continue
        entry = saved.get(key)
        if entry is not None:
            track = 1 if entry.get("track") == 1 else 0
            try:
                start = max(0, int(entry.get("start", 0)))
            except (TypeError, ValueError):
                start = 0
            curve = _sanitize_curve(entry.get("curve"))
        else:
            prev = max(clips, key=lambda c: c["start"] + c["frames"], default=None)
            if prev is None:
                start, track = 0, 0
            else:
                overlap = max(0, min(int(default_overlap), prev["frames"] - 1, frames - 1))
                start = max(0, prev["start"] + prev["frames"] - overlap)
                track = 1 - prev["track"]
            curve = None
        clips.append({
            "key": key,
            "track": track,
            "start": start,
            "frames": frames,
            "images": images,
            "audio": audios.get(key),
            "curve": curve,
        })

    clips.sort(key=lambda c: (c["start"], c["start"] + c["frames"], c["key"]))
    return clips


def _resize_clip(images, W, H):
    """
    Fit-pad (letterbox) onto black rather than stretch: aspect distortion is
    most visible mid-crossfade when both clips are on screen, and black bars
    composite naturally with the black timeline base. Chunked to bound
    common_upscale's transient allocations on long clips.
    """
    B, h, w, C = images.shape
    if (h, w) == (H, W):
        return images
    scale = min(W / w, H / h)
    tw = max(1, round(w * scale))
    th = max(1, round(h * scale))
    out = torch.zeros((B, H, W, C), dtype=torch.float32)
    y0 = (H - th) // 2
    x0 = (W - tw) // 2
    for i in range(0, B, _RESIZE_CHUNK):
        chunk = images[i:i + _RESIZE_CHUNK].movedim(-1, 1)
        chunk = comfy.utils.common_upscale(chunk, tw, th, "bilinear", "disabled")
        out[i:i + _RESIZE_CHUNK, y0:y0 + th, x0:x0 + tw] = chunk.movedim(1, -1)
    return out


def _render_video(clips, W, H):
    """
    Segment sweep: boundaries are every clip start/end, so the active set is
    constant per segment -- solo segments are pure slice copies and only real
    overlaps pay for blend math. Records each clip's per-frame crossfade
    weight for the audio mix (single source of truth for both fades).
    Weights use smoothstep sampled at frame centers,
    u = (t - overlap_start + 0.5) / overlap_len; s(u) + s(1-u) = 1 exactly,
    so two-clip frames are always fully opaque.
    """
    total = max(c["start"] + c["frames"] for c in clips)
    out = torch.zeros((total, H, W, 3), dtype=torch.float32)
    resized = {c["key"]: _resize_clip(c["images"], W, H) for c in clips}
    for c in clips:
        c["weight"] = torch.ones(c["frames"], dtype=torch.float32)

    bounds = sorted({0, total, *(c["start"] for c in clips),
                     *(c["start"] + c["frames"] for c in clips)})
    for a, b in zip(bounds, bounds[1:]):
        if a < 0 or b > total or b <= a:
            continue
        active = [c for c in clips
                  if c["start"] <= a and b <= c["start"] + c["frames"]]
        if not active:
            continue

        if len(active) == 1:
            c = active[0]
            out[a:b] = resized[c["key"]][a - c["start"]:b - c["start"]]
            continue

        t = torch.arange(a, b, dtype=torch.float32)
        if len(active) == 2:
            first, second = sorted(
                active,
                key=lambda c: (c["start"], -(c["start"] + c["frames"]), c["key"]))
            ov_start = second["start"]
            ov_len = first["start"] + first["frames"] - ov_start
            w = _smoothstep((t - ov_start + 0.5) / ov_len)
            wb = w.view(-1, 1, 1, 1)
            sa = resized[first["key"]][a - first["start"]:b - first["start"]]
            sb = resized[second["key"]][a - second["start"]:b - second["start"]]
            out[a:b] = sa * (1.0 - wb) + sb * wb
            first["weight"][a - first["start"]:b - first["start"]] = 1.0 - w
            second["weight"][a - second["start"]:b - second["start"]] = w
            continue

        # Three or more concurrent clips only happens with hand-edited JSON
        # (the widget forbids same-track overlap). Degrade gracefully: tent
        # weights peaking mid-clip, normalized to sum 1 -- smooth and opaque,
        # never an error.
        weights = []
        for c in active:
            half = max(c["frames"] / 2.0, 0.5)
            edge = torch.minimum(t - c["start"] + 0.5,
                                 c["start"] + c["frames"] - t - 0.5)
            weights.append(_smoothstep(edge / half).clamp(min=1e-3))
        stack = torch.stack(weights)
        stack = stack / stack.sum(dim=0, keepdim=True)
        blended = torch.zeros((b - a, H, W, 3), dtype=torch.float32)
        for c, w in zip(active, stack):
            blended += resized[c["key"]][a - c["start"]:b - c["start"]] * w.view(-1, 1, 1, 1)
            c["weight"][a - c["start"]:b - c["start"]] = w
        out[a:b] = blended

    return out, total


def _curve_gains(curve, frames):
    """Per-frame volume gain: hermite(curve, f / (frames-1)), clamped 0..1.
    The clamp matters -- hermite can overshoot between keys, and a negative
    gain would invert audio phase."""
    if not curve:
        return torch.ones(frames, dtype=torch.float32)
    denom = max(frames - 1, 1)
    gains = [min(1.0, max(0.0, hermite(curve, f / denom))) for f in range(frames)]
    return torch.tensor(gains, dtype=torch.float32)


def _adapt_channels(wave, channels):
    """(C, N) -> (channels, N): repeat mono up, mean-downmix anything else."""
    if wave.shape[0] == channels:
        return wave
    if wave.shape[0] == 1:
        return wave.repeat(channels, 1)
    mono = wave.mean(dim=0, keepdim=True)
    return mono.repeat(channels, 1) if channels > 1 else mono


def _mix_audio(clips, total_frames, fps):
    """
    Sample rate and channel count follow the first connected audio in wire
    order (none -> 44100 mono silence). Each clip's envelope is built
    per-frame (volume curve x the crossfade weight recorded by the video
    render) then linearly upsampled to samples, so audio provably matches
    the video fade. No clamping: crossfade weights sum to 1 so overlap
    addition cannot clip on its own.
    """
    ref = next((c for c in clips if c["audio"] is not None), None)
    if ref is not None:
        sr = int(ref["audio"]["sample_rate"])
        channels = int(ref["audio"]["waveform"].shape[1])
    else:
        sr, channels = 44100, 1
    total_samples = max(1, round(total_frames / fps * sr))
    mix = torch.zeros((1, channels, total_samples), dtype=torch.float32)

    for c in clips:
        if c["audio"] is None:
            continue
        wave = c["audio"]["waveform"][0].float().cpu()
        src_sr = int(c["audio"]["sample_rate"])
        if src_sr != sr:
            wave = torchaudio.functional.resample(wave, src_sr, sr)
        wave = _adapt_channels(wave, channels)

        n = max(1, round(c["frames"] / fps * sr))
        if wave.shape[1] >= n:
            wave = wave[:, :n]
        else:
            wave = torch.nn.functional.pad(wave, (0, n - wave.shape[1]))

        env_frames = _curve_gains(c["curve"], c["frames"]) * c["weight"]
        if c["frames"] == 1:
            env = env_frames.repeat(n)
        else:
            env = torch.nn.functional.interpolate(
                env_frames.view(1, 1, -1), size=n,
                mode="linear", align_corners=True).view(-1)

        s0 = max(0, round(c["start"] / fps * sr))
        end = min(total_samples, s0 + n)
        if end > s0:
            mix[0, :, s0:end] += wave[:, :end - s0] * env[:end - s0]

    return {"waveform": mix, "sample_rate": sr}


def _clip_content_hash(images, audio, fps):
    """
    Cheap content fingerprint for the preview cache: shape plus a few strided
    frame/audio samples, not the full tensors. Content-addressed names mean
    two nodes sharing a clip share one preview file.
    """
    h = hashlib.sha1()
    h.update(f"{_PREVIEW_VERSION}|{_PREVIEW_MAX_DIM}|{fps:.4f}|{tuple(images.shape)}".encode())
    B, H, W, _ = images.shape
    stride = max(1, min(H, W) // 16)
    for f in sorted({0, B - 1, *(round(i * (B - 1) / 7) for i in range(8))}):
        sample = (images[f, ::stride, ::stride] * 255).clamp(0, 255).to(torch.uint8)
        h.update(sample.cpu().numpy().tobytes())
    if audio is not None:
        wf = audio["waveform"]
        h.update(f"a|{audio['sample_rate']}|{tuple(wf.shape)}".encode())
        astride = max(1, wf.shape[-1] // 4096)
        h.update(wf[0, :, ::astride].to(torch.float16).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _encode_preview(path, images, audio, fps):
    """
    Low-res preview encode, modeled on core VideoFromComponents.save_to.
    Dimensions are rounded down to even (yuv420p/h264 requirement). Audio is
    encoded raw -- the widget applies volume curve and fades live.
    """
    B, h, w, _ = images.shape
    scale = min(1.0, _PREVIEW_MAX_DIM / max(w, h))
    tw = max(2, int(w * scale) // 2 * 2)
    th = max(2, int(h * scale) // 2 * 2)

    with av.open(path, mode="w", format="mp4") as output:
        # Every stream must be added before the first mux writes the
        # container header; a stream added after that has no valid
        # time_base and muxing raises "Cannot rebase to zero time".
        rate = Fraction(round(fps * 1000), 1000)
        vstream = output.add_stream("h264", rate=rate)
        vstream.width = tw
        vstream.height = th
        vstream.pix_fmt = "yuv420p"
        vstream.options = {"crf": "23", "preset": "veryfast"}

        astream = None
        wave = None
        layout = None
        if audio is not None:
            sr = int(audio["sample_rate"])
            wave = audio["waveform"][0].float().cpu()
            if wave.shape[0] not in (1, 2, 6):
                wave = wave.mean(dim=0, keepdim=True)
            n = max(1, round(B / fps * sr))
            if wave.shape[1] >= n:
                wave = wave[:, :n]
            else:
                wave = torch.nn.functional.pad(wave, (0, n - wave.shape[1]))
            layout = {1: "mono", 2: "stereo", 6: "5.1"}[wave.shape[0]]
            try:
                astream = output.add_stream("aac", rate=sr, layout=layout)
            except (av.FFmpegError, ValueError):
                # AAC rejects exotic sample rates; retry once at 48 kHz.
                wave = torchaudio.functional.resample(wave, sr, 48000)
                sr = 48000
                astream = output.add_stream("aac", rate=sr, layout=layout)

        for i in range(0, B, _RESIZE_CHUNK):
            chunk = images[i:i + _RESIZE_CHUNK]
            if (th, tw) != (h, w):
                chunk = comfy.utils.common_upscale(
                    chunk.movedim(-1, 1), tw, th, "bilinear", "disabled").movedim(1, -1)
            arr = (chunk * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            for img in arr:
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
                frame = frame.reformat(format="yuv420p")
                output.mux(vstream.encode(frame))
        output.mux(vstream.encode(None))

        if astream is not None:
            aframe = av.AudioFrame.from_ndarray(
                wave.contiguous().numpy(), format="fltp", layout=layout)
            aframe.sample_rate = sr
            aframe.pts = 0
            output.mux(astream.encode(aframe))
            output.mux(astream.encode(None))


def _materialize_preview(images, audio, fps):
    """
    Ensure a preview file exists for this clip; returns the annotated file
    reference or None. Cache hit refreshes mtime so the TTL sweep sees the
    file as live. Publish is atomic (.part then os.replace) so the scrubber
    routes can never see a half-written mp4; a preview failure never fails
    the run.
    """
    if not HAS_AV:
        return None
    name = f"st_{_clip_content_hash(images, audio, fps)}.mp4"
    path = os.path.join(_preview_dir(), name)
    ref = {"filename": name, "subfolder": _PREVIEW_SUBFOLDER, "type": "temp"}
    if os.path.isfile(path):
        try:
            os.utime(path)
        except OSError:
            pass
        return ref
    part = f"{path}.{os.getpid()}-{next(_part_counter)}.part"
    try:
        _encode_preview(part, images, audio, fps)
        # Defensive Windows handle release before replacing the target.
        _drop_scrub_session(path)
        os.replace(part, path)
    except (av.FFmpegError, OSError, ValueError, EOFError):
        try:
            if os.path.isfile(part):
                os.remove(part)
        except OSError:
            pass
        return None
    return ref


def _cleanup_stale_previews():
    """
    Age-gated sweep of temp/simple_track (precedent: video_scrubber's upload
    temp cleanup). The utime-touch on every cache hit plus the long TTL means
    this can never delete a file another live node referenced recently.
    """
    try:
        directory = os.path.join(
            folder_paths.get_temp_directory(), _PREVIEW_SUBFOLDER)
        names = os.listdir(directory)
    except OSError:
        return
    now = time.time()
    for name in names:
        path = os.path.join(directory, name)
        try:
            if not os.path.isfile(path):
                continue
            mtime = os.path.getmtime(path)
            stale = (name.endswith(".part") and mtime < now - _PART_TTL) or (
                name.startswith("st_") and name.endswith(".mp4")
                and mtime < now - _PREVIEW_TTL)
            if stale:
                _drop_scrub_session(path)
                os.remove(path)
        except OSError:
            continue


_cleanup_stale_previews()


class SimpleTrack:
    """Blend any number of clips into one video on a two-track crossfade
    timeline, with per-clip audio volume curves."""

    CATEGORY = "CCN"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Output width. 0 = take it from the first clip.",
                }),
                "height": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Output height. 0 = take it from the first clip.",
                }),
                "fps": ("FLOAT", {
                    "default": 16.0, "min": 1.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Timeline timebase; every clip is interpreted at "
                               "this rate. Only affects audio timing and the "
                               "output's intended playback rate.",
                }),
                "default_overlap": ("INT", {
                    "default": 8, "min": 0, "max": 999, "step": 1,
                    "tooltip": "Crossfade length in frames used when a newly "
                               "connected clip is auto-placed on the timeline.",
                }),
                "track_data": ("STRING", {
                    "default": "{}", "multiline": False,
                    "tooltip": "Timeline state managed by the track widget; "
                               "edit there, not here.",
                }),
            },
            "optional": FlexibleOptionalInputType(_ANY),
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "combine"
    # Output node so a bare "queue to populate the timeline previews" works
    # while arranging, before any save node is attached.
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Combines any number of clips (IMAGE batch + optional AUDIO pairs) "
        "into one video. Drag clips on the two tracks to overlap them; "
        "overlaps become smooth crossfades of both video and audio, and each "
        "clip has an editable volume curve. Note: before the first run, clip "
        "lengths shown by the widget come from upstream video files and may "
        "exceed the frames actually reaching this node; the first run "
        "corrects them."
    )

    def combine(self, width, height, fps, default_overlap, track_data, **kwargs):
        videos = {}
        audios = {}
        for k, v in kwargs.items():
            m = _KEY_RE.match(k)
            if m and torch.is_tensor(v) and v.dim() == 4:
                videos[m.group(1)] = v[..., :3].float().cpu()
            elif k.startswith("audio_") and isinstance(v, dict) and "waveform" in v:
                audios[k.split("_", 1)[1]] = v

        fps = float(fps)
        if not videos:
            W, H = (width or 512), (height or 512)
            images = torch.zeros((1, H, W, 3), dtype=torch.float32)
            audio = {
                "waveform": torch.zeros(
                    (1, 1, max(1, round(44100 / fps))), dtype=torch.float32),
                "sample_rate": 44100,
            }
            payload = {
                "version": 1, "fps": fps, "width": W, "height": H,
                "total_frames": 1, "sample_rate": 44100,
                "av_missing": not HAS_AV, "clips": [],
            }
            return {"ui": {"ccn_simple_track": [payload]}, "result": (images, audio)}

        first = videos[sorted(videos, key=_wire_order)[0]]
        W = width or int(first.shape[2])
        H = height or int(first.shape[1])

        clips = _reconcile(track_data, videos, audios, default_overlap)
        images, total = _render_video(clips, W, H)
        audio = _mix_audio(clips, total, fps)

        previews = {c["key"]: _materialize_preview(c["images"], c["audio"], fps)
                    for c in clips}
        payload = {
            "version": 1, "fps": fps, "width": W, "height": H,
            "total_frames": total, "sample_rate": audio["sample_rate"],
            "av_missing": not HAS_AV,
            "clips": [{
                "key": c["key"], "track": c["track"],
                "start": c["start"], "frames": c["frames"],
                "source_width": int(c["images"].shape[2]),
                "source_height": int(c["images"].shape[1]),
                "has_audio": c["audio"] is not None,
                "preview": previews.get(c["key"]),
            } for c in clips],
        }
        _cleanup_stale_previews()
        return {"ui": {"ccn_simple_track": [payload]}, "result": (images, audio)}
