"""
LoRA Loader Filtered - faithful clone of the built-in LoRA loader with a
sortable picker (by modification date, name, or size). Text search comes for
free from the combo's native type-to-filter.
"""

import os

import folder_paths
import comfy.sd
import comfy.utils


# Metadata route for the frontend. INPUT_TYPES runs without widget context and
# can only return a static alphabetical list, so it cannot sort by mtime/size.
# The JS extension fetches this and reorders the lora_name combo client-side.
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/ccn/lora_filter/list")
    async def _lora_filter_list(request):
        entries = {}
        for name in folder_paths.get_filename_list("loras"):
            path = folder_paths.get_full_path("loras", name)
            if not path:
                # Keep the name present with zeros so the frontend treats it as
                # "known" and doesn't refetch on every sort looking for it.
                entries[name] = {"mtime": 0, "size": 0}
                continue
            try:
                st = os.stat(path)
                entries[name] = {"mtime": st.st_mtime, "size": st.st_size}
            except OSError:
                entries[name] = {"mtime": 0, "size": 0}
        return web.json_response(entries)
except Exception:
    pass


class LoraLoaderFiltered:
    """LoRA loader matching the built-in node, with name/date/size sorting of the picker."""

    def __init__(self):
        self.loaded_lora = None

    SORT_BY = ["date_modified", "name", "size"]
    SORT_ORDER = ["descending", "ascending"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "sort_by": (cls.SORT_BY, {"default": "date_modified"}),
                "sort_order": (cls.SORT_ORDER, {"default": "descending"}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load_lora"
    CATEGORY = "ComfyCollectorNodes/Loaders"

    # sort_by / sort_order are display-only: the frontend reorders the picker and
    # the backend ignores them. Selection therefore behaves exactly like the
    # built-in loader regardless of how the list is sorted on screen.
    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, sort_by, sort_order):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)

        # Reuse the cached tensor dict when the same file is requested again,
        # mirroring the built-in loader's caching behavior.
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        return (model_lora, clip_lora)
