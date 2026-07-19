"""
Video Loader By Index - Load video files by their position in a folder
"""

import os
import numpy as np
import torch

# PyAV is part of ComfyUI's own requirements; the guard only matters on
# installs old enough to predate that, where the pack should still load.
try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False


VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif', '.m4v', '.wmv', '.flv')


def find_files(directory, extensions, recursive):
    """Collect files matching extensions from a directory, sorted alphabetically."""
    if recursive:
        files = []
        for root, _, filenames in os.walk(directory):
            for f in filenames:
                if f.lower().endswith(extensions):
                    rel_path = os.path.relpath(os.path.join(root, f), directory)
                    files.append(rel_path)
        return sorted(files)
    else:
        return sorted([
            f for f in os.listdir(directory)
            if f.lower().endswith(extensions)
        ])


def resolve_index(index, total):
    """Wrap index if it exceeds total. Returns (actual_index, wrapped)."""
    wrapped = index >= total
    actual_index = index % total
    return actual_index, wrapped


def load_video_frames(file_path, frame_skip, max_frames):
    """
    Extract frames from a video file using PyAV.
    Returns a list of numpy arrays in RGB format, plus metadata.
    """
    try:
        container = av.open(file_path)
    except (av.FFmpegError, OSError, ValueError) as e:
        raise ValueError(f"Could not open video: {file_path}") from e

    with container:
        if not container.streams.video:
            raise ValueError(f"No video stream in: {file_path}")
        stream = container.streams.video[0]

        fps = float(stream.average_rate) if stream.average_rate else 0.0
        total_video_frames = stream.frames
        if not total_video_frames and fps > 0:
            # Some containers (webm, gif) don't declare a frame count;
            # estimate it from the duration instead.
            if stream.duration and stream.time_base:
                total_video_frames = int(float(stream.duration * stream.time_base) * fps)
            elif container.duration:
                total_video_frames = int(container.duration / av.time_base * fps)

        frames = []
        frame_idx = 0
        collected = 0

        try:
            for frame in container.decode(stream):
                if frame_idx % (frame_skip + 1) == 0:
                    frames.append(frame.to_ndarray(format="rgb24"))
                    collected += 1

                    if max_frames > 0 and collected >= max_frames:
                        break

                frame_idx += 1
        except av.FFmpegError:
            # Trailing corruption ends extraction with whatever was decoded,
            # matching the old behavior of stopping at the first bad read.
            pass

        meta = {
            "total_video_frames": total_video_frames,
            "fps": fps,
            "width": stream.width,
            "height": stream.height,
            "extracted_frames": len(frames),
        }
    return frames, meta


class VideoLoaderByIndex:
    """
    Loads a video file from a directory based on its index position.
    Outputs all frames as a batched IMAGE tensor for use in video workflows.
    Automatically wraps index if it exceeds the number of available files.

    Requires PyAV (av), bundled with ComfyUI.
    """

    CATEGORY = "ComfyCollectorNodes/Loaders"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "placeholder": "/path/to/videos"}),
                "recursive": ("BOOLEAN", {"default": False}),
                "index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "frame_skip": ("INT", {
                    "default": 0, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Skip N frames between each extracted frame. 0 = every frame, 1 = every other, etc.",
                }),
                "max_frames": ("INT", {
                    "default": 0, "min": 0, "max": 99999, "step": 1,
                    "tooltip": "Maximum frames to extract. 0 = all frames.",
                }),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "FLOAT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("frames", "filename", "file_path", "frame_count", "fps", "total_files", "actual_index", "wrapped")
    FUNCTION = "load_video_by_index"

    def load_video_by_index(self, directory, recursive, index, frame_skip, max_frames, debug=False):
        if not HAS_AV:
            raise ImportError(
                "PyAV (av) is required for VideoLoaderByIndex. It ships with "
                "ComfyUI; update ComfyUI or install with: pip install av"
            )

        directory = directory.strip()
        if not directory:
            raise ValueError("No directory specified")
        if not os.path.isdir(directory):
            raise ValueError(f"Directory not found: {directory}")

        video_files = find_files(directory, VIDEO_EXTENSIONS, recursive)
        if not video_files:
            raise ValueError(f"No video files found in {directory}")

        total_files = len(video_files)
        actual_index, wrapped = resolve_index(index, total_files)

        if debug:
            if wrapped:
                print(f"[ComfyCollectorNodes] Index {index} exceeds {total_files} videos, wrapping to index {actual_index}")
            print(f"[ComfyCollectorNodes] Loading video {actual_index + 1}/{total_files}: {video_files[actual_index]}")

        filename = video_files[actual_index]
        file_path = os.path.join(directory, filename)

        frames, meta = load_video_frames(file_path, frame_skip, max_frames)

        if not frames:
            raise ValueError(f"No frames extracted from {file_path}")

        if debug:
            skip_info = f", skip={frame_skip}" if frame_skip > 0 else ""
            limit_info = f", limit={max_frames}" if max_frames > 0 else ""
            print(
                f"[ComfyCollectorNodes] Video: {meta['width']}x{meta['height']} @ {meta['fps']:.2f}fps, "
                f"{meta['total_video_frames']} total frames, {meta['extracted_frames']} extracted"
                f"{skip_info}{limit_info}"
            )

        # Stack frames into (B, H, W, C) float32 [0, 1]
        frames_np = np.stack(frames).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_np)

        if debug:
            print(f"[ComfyCollectorNodes] Output tensor shape: {list(frames_tensor.shape)}")

        return (
            frames_tensor,
            filename,
            file_path,
            meta["extracted_frames"],
            meta["fps"],
            total_files,
            actual_index,
            wrapped,
        )

    @classmethod
    def IS_CHANGED(cls, directory, recursive, index, frame_skip, max_frames, debug=False):
        directory = directory.strip()
        if not directory or not os.path.isdir(directory):
            return ""
        video_files = find_files(directory, VIDEO_EXTENSIONS, recursive)
        if not video_files:
            return ""
        total = len(video_files)
        actual_index = index % total
        file_path = os.path.join(directory, video_files[actual_index])
        return f"{file_path}:{os.path.getmtime(file_path)}:{frame_skip}:{max_frames}"
