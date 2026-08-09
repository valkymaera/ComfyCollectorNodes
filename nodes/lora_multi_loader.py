"""
LoRA Multi Loader -- dynamic-row LoRA stack with trigger-word support.

The dynamic rows are provided by js/lora_multi_loader.js. Rows arrive as
widget values named "lora_<n>", each a dict:
    {"on": bool, "lora": str, "strength": float, "id": int}
Each enabled row patches the model (and CLIP when connected) at its
strength times the node-wide `strength_multiplier`; an effective strength
of 0 skips the row. The `enabled` toggle bypasses the whole node, passing
model/CLIP through unchanged. The `triggers` output joins the top trigger
words of every applied row.

Trigger words are ranked by kohya ss_tag_frequency counts (most-used tags
first), falling back to explicit trigger-phrase metadata keys and kohya
dataset directory names when no tag frequency is embedded. The same
extraction feeds POST /ccn/lora_triggers, which the row widget uses to
display trigger chips without executing the workflow.
"""

import json

import folder_paths
import comfy.sd
import comfy.utils
import safetensors

from .lora_metadata import LoraMetadata


class AnyType(str):
    """A type string that never fails an equality check against another type."""

    def __ne__(self, other):
        return False


class FlexibleOptionalInputType(dict):
    """Accepts arbitrarily named optional inputs (the dynamic rows), while
    still declaring explicit known sockets."""

    def __init__(self, input_type, explicit=None):
        super().__init__()
        self.input_type = input_type
        if explicit:
            self.update(explicit)

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        return (self.input_type,)

    def __contains__(self, key):
        return True


_ANY = AnyType("*")
_NONE_VALUES = ("", "none", "None")

# Explicit trigger-key values longer than this are caption/comment blobs,
# not usable as chips.
_MAX_TRIGGER_LEN = 80

# Serialized-nothing values some trainers write into metadata fields.
_JUNK_TRIGGERS = {"", "none", "null"}


def get_top_triggers(lora_name, limit=3):
    """Top trigger words for a lora rel-path, best-ranked first.

    Returns [] on any failure (missing file, non-safetensors, unreadable
    or absent metadata) rather than raising.
    """
    path = folder_paths.get_full_path("loras", lora_name)
    if path is None or not path.lower().endswith(".safetensors"):
        return []
    try:
        with safetensors.safe_open(path, framework="pt") as f:
            meta = f.metadata()
    except Exception:
        return []
    if not meta:
        return []

    # Primary: kohya tag frequency, ranked by use count across datasets.
    if "ss_tag_frequency" in meta:
        try:
            merged = {}
            for tags in json.loads(meta["ss_tag_frequency"]).values():
                if not isinstance(tags, dict):
                    continue
                for tag, count in tags.items():
                    tag = tag.strip()
                    if tag.lower() in _JUNK_TRIGGERS:
                        continue
                    try:
                        merged[tag] = merged.get(tag, 0) + int(count)
                    except (TypeError, ValueError):
                        continue
            if merged:
                ranked = sorted(merged.items(), key=lambda kv: -kv[1])
                return [tag for tag, _ in ranked[:limit]]
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: explicit trigger phrases, then kohya "N_concept" dir names.
    triggers = []
    seen = set()
    for key in LoraMetadata.TRIGGER_KEYS:
        val = meta.get(key, "").strip()
        if val.lower() in _JUNK_TRIGGERS:
            continue
        if len(val) <= _MAX_TRIGGER_LEN and val not in seen:
            seen.add(val)
            triggers.append(val)
            if len(triggers) >= limit:
                return triggers
    if "ss_dataset_dirs" in meta:
        try:
            for dirname in json.loads(meta["ss_dataset_dirs"]):
                parts = dirname.split("_", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    concept = parts[1]
                    if concept not in seen:
                        seen.add(concept)
                        triggers.append(concept)
                        if len(triggers) >= limit:
                            break
        except (json.JSONDecodeError, AttributeError):
            pass
    return triggers


class LoraMultiLoader:
    """Dynamic multi-LoRA loader with per-row strengths and trigger words.

    Rows expand as LoRAs are added (like a power-lora stack). Each enabled
    row patches the model, and CLIP too when connected, at its strength.
    LoRA patches are additive, so row order does not affect the result and
    duplicate rows effectively sum their strengths.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to patch."}),
            },
            "optional": FlexibleOptionalInputType(_ANY, {
                "clip": ("CLIP", {
                    "tooltip": "Optional; row strengths also apply to the "
                               "text encoder when connected.",
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When off, the model and CLIP pass through "
                               "unchanged and no triggers are emitted.",
                }),
                "strength_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Multiplies every row's strength; 0 disables "
                               "the whole stack.",
                }),
            }),
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "triggers")
    FUNCTION = "load"
    CATEGORY = "CCN"
    DESCRIPTION = (
        "Dynamic LoRA stack: rows expand as LoRAs are added, each patching "
        "the model (and CLIP when connected) at its strength scaled by the "
        "strength multiplier. The enabled toggle bypasses the whole stack. "
        "The triggers output joins each applied LoRA's top trigger words, "
        "ranked from embedded training metadata."
    )

    def load(self, model, clip=None, enabled=True,
             strength_multiplier=1.0, **kwargs):
        if not enabled:
            return (model, clip, "")

        try:
            strength_multiplier = float(strength_multiplier)
        except (TypeError, ValueError):
            raise ValueError(
                "LoraMultiLoader: strength_multiplier is non-numeric: "
                f"{strength_multiplier!r}"
            )

        rows = [(key, value) for key, value in kwargs.items()
                if isinstance(value, dict) and "lora" in value]
        rows.sort(key=self._row_key)

        available = set(folder_paths.get_filename_list("loras"))
        loaded = {}  # path -> tensors, so duplicate rows load a file once
        trigger_words = []
        for row_number, (key, value) in enumerate(rows, start=1):
            label = f"row {row_number}"
            if not value.get("on", True):
                continue

            lora_name = value.get("lora")
            if not isinstance(lora_name, str) or lora_name in _NONE_VALUES:
                continue

            try:
                strength = float(value.get("strength", 1.0))
            except (TypeError, ValueError):
                raise ValueError(
                    f"LoraMultiLoader: {label} ({lora_name!r}) has a "
                    f"non-numeric strength: {value.get('strength')!r}"
                )
            strength *= strength_multiplier
            if strength == 0.0:
                continue

            if lora_name not in available:
                raise ValueError(
                    f"LoraMultiLoader: {label} references {lora_name!r}, "
                    "which is not in the loras folder. Refresh the node or "
                    "fix the row; refusing to run with a silently missing "
                    "LoRA."
                )

            path = folder_paths.get_full_path_or_raise("loras", lora_name)
            lora = loaded.get(path)
            if lora is None:
                try:
                    lora = comfy.utils.load_torch_file(path, safe_load=True)
                except Exception as exc:
                    raise ValueError(
                        f"LoraMultiLoader: {label} failed to load "
                        f"{lora_name!r}: {exc}"
                    ) from exc
                loaded[path] = lora
            model, clip = comfy.sd.load_lora_for_models(
                model, clip, lora, strength, strength)

            for word in get_top_triggers(lora_name):
                if word not in trigger_words:
                    trigger_words.append(word)

        return (model, clip, ", ".join(trigger_words))

    # Numeric sort on the "lora_<n>" suffix keeps row 10 after row 2;
    # non-numeric names sort after, lexicographically.
    @staticmethod
    def _row_key(item):
        suffix = item[0].partition("_")[2]
        return (0, int(suffix), "") if suffix.isdigit() else (1, 0, item[0])


# ----------------------------------------------------------------------
#  Trigger chips: batch trigger lookup for the row widget
#  (js/lora_multi_loader.js). The route registers only when running
#  inside the ComfyUI server; the module imports cleanly without it.
# ----------------------------------------------------------------------

try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/ccn/lora_triggers")
    async def _ccn_lora_triggers(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        names = body.get("names") if isinstance(body, dict) else None
        if not isinstance(names, list):
            return web.json_response(
                {"error": 'expected {"names": [...]}'}, status=400)
        out = {}
        for name in names[:200]:
            if isinstance(name, str) and name not in _NONE_VALUES:
                out[name] = get_top_triggers(name)
        return web.json_response(out)
except Exception:
    pass
