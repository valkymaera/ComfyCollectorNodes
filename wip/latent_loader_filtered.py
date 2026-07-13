"""
Latent Loader Filtered - loads .latent files from ComfyUI's input directory
with a sortable picker (by modification date, name, or size). Unlike the
built-in Load Latent, it recurses into subfolders, so latents organized under
input/latents/ (or any subfolder) are found. Text search comes for free from
the combo's native type-to-filter.
"""

import os

import safetensors.torch
import folder_paths


LATENT_EXTENSION = ".latent"

# Shown when the input directory holds no latents, so the combo renders a clear
# message instead of an empty list (which the frontend displays as "undefined").
NO_FILES = "(no .latent files in input)"


def _find_latents(input_dir):
    """Recursively collect .latent files as forward-slash relative paths, sorted."""
    found = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(LATENT_EXTENSION):
                rel = os.path.relpath(os.path.join(root, f), input_dir)
                found.append(rel.replace(os.sep, "/"))
    return sorted(found)


def _resolve(input_dir, name):
    """Turn a forward-slash relative name back into an absolute OS path."""
    return os.path.join(input_dir, *name.split("/"))


# Metadata route for the frontend. INPUT_TYPES runs without widget context and
# returns a static alphabetical list, so it cannot sort by mtime/size. The JS
# extension fetches this and reorders the picker client-side.
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/ccn/latent_filter/list")
    async def _latent_filter_list(request):
        input_dir = folder_paths.get_input_directory()
        entries = {}
        for name in _find_latents(input_dir):
            path = _resolve(input_dir, name)
            try:
                st = os.stat(path)
                entries[name] = {"mtime": st.st_mtime, "size": st.st_size}
            except OSError:
                entries[name] = {"mtime": 0, "size": 0}
        return web.json_response(entries)
except Exception:
    pass


class LatentLoaderFiltered:
    """Loads a saved .latent from the input directory, with name/date/size sorting of the picker."""

    SORT_BY = ["date_modified", "name", "size"]
    SORT_ORDER = ["descending", "ascending"]

    @classmethod
    def INPUT_TYPES(cls):
        files = _find_latents(folder_paths.get_input_directory())
        return {
            "required": {
                "latent": (files if files else [NO_FILES],),
                "sort_by": (cls.SORT_BY, {"default": "date_modified"}),
                "sort_order": (cls.SORT_ORDER, {"default": "descending"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "load"
    CATEGORY = "ComfyCollectorNodes/Loaders"
    DESCRIPTION = (
        "Load a saved .latent from ComfyUI's input directory. Recurses into "
        "subfolders and sorts the picker by date / name / size. Move latents "
        "from output/latents/ into input/ to make them selectable here."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, latent=None, **kwargs):
        if not latent or latent == NO_FILES:
            return True  # Handled at execution with a helpful message.
        input_dir = folder_paths.get_input_directory()
        if not os.path.exists(_resolve(input_dir, latent)):
            return f"Latent file not found: {latent}"
        return True

    # sort_by / sort_order are display-only: the frontend reorders the picker and
    # the backend ignores them, so selection behaves like the built-in loader.
    def load(self, latent, sort_by, sort_order):
        if not latent or latent == NO_FILES:
            raise ValueError(
                "No .latent files found in the input directory. Move a .latent "
                "from output/latents/ into ComfyUI's input folder (subfolders "
                "are fine) and refresh."
            )

        input_dir = folder_paths.get_input_directory()
        path = _resolve(input_dir, latent)
        if not os.path.exists(path):
            raise ValueError(f"Latent file not found: {path}")

        data = safetensors.torch.load_file(path, device="cpu")

        # Latents saved before the versioned format were scaled by the SD1.5
        # factor; rescale those so every format comes out in the same space.
        # The marker key is present only in newer files.
        multiplier = 1.0 if "latent_format_version_0" in data else (1.0 / 0.18215)
        samples = {"samples": data["latent_tensor"].float() * multiplier}
        return (samples,)

    @classmethod
    def IS_CHANGED(cls, latent, sort_by=None, sort_order=None):
        if not latent or latent == NO_FILES:
            return ""
        input_dir = folder_paths.get_input_directory()
        path = _resolve(input_dir, latent)
        if os.path.exists(path):
            return f"{path}:{os.path.getmtime(path)}"
        return ""
