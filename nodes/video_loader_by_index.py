"""
Video Loader By Index - Load video files by their position in a folder
"""

import os
import numpy as np
import torch

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


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
    Extract frames from a video file using OpenCV.
    Returns a list of numpy arrays in RGB format, plus metadata.
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {file_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    frame_idx = 0
    collected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % (frame_skip + 1) == 0:
            # OpenCV reads BGR, convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            collected += 1

            if max_frames > 0 and collected >= max_frames:
                break

        frame_idx += 1

    cap.release()

    meta = {
        "total_video_frames": total_video_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "extracted_frames": len(frames),
    }
    return frames, meta


class VideoLoaderByIndex:
    """
    Loads a video file from a directory based on its index position.
    Outputs all frames as a batched IMAGE tensor for use in video workflows.
    Automatically wraps index if it exceeds the number of available files.

    Requires OpenCV (cv2).
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
        if not HAS_CV2:
            raise ImportError(
                "OpenCV (cv2) is required for VideoLoaderByIndex. "
                "Install with: pip install opencv-python-headless"
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
