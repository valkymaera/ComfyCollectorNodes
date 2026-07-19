"""
LoRA Pair Loader -- dynamic-row LoRA stack for Wan 2.2 dual-expert models.

Each row selects one LoRA file and carries two weights: H (applied to the
high-noise model) and L (applied to the low-noise model). At execution the
loader resolves the row's partner file by conservative filename-token
swapping (high_noise <-> low_noise and friends). A swap only counts when
the swapped filename actually exists in the lora folder, so false-positive
tokens are inert. Resolution outcomes per row:

  * Pair resolved: the high-side file is applied to model_high at weight H
    and the low-side file to model_low at weight L, regardless of which of
    the two files the row selected.
  * No partner found: the selected file is applied to both models at their
    respective weights (covers Wan 2.1-era and single-file 2.2 LoRAs).

A weight of 0 skips that side entirely. This doubles as the manual escape
hatch: any exotic routing is expressible with two rows carrying opposite
zero weights (e.g. force one file onto both experts even though a partner
exists, or cross-apply a high-trained file to the low expert).

Application is model-only (no text-encoder patching), matching how Wan
LoRAs ship and how core LoraLoaderModelOnly behaves. LoRA patches are
additive, so row order does not affect the result.

The dynamic rows are provided by js/wip/lora_pair_loader.js. Rows arrive as
widget values named "lora_<n>", each a dict:
    {"on": bool, "lora": str, "strength_high": float, "strength_low": float}
The pair-token table below is mirrored in the JS file for the row badge;
keep the two in sync.
"""

import logging
import os

import folder_paths
import comfy.sd
import comfy.utils


class AnyType(str):
    """A type string that never fails an equality check against another type."""

    def __ne__(self, other):
        return False


class FlexibleOptionalInputType(dict):
    """Accepts arbitrarily named optional inputs (dynamic rows and their
    curve sockets), while still declaring explicit known sockets."""

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

# Ordered most-specific first. Mirrored in js/lora_pair_loader.js.
_PAIR_TOKENS = [
    ("high_noise", "low_noise"),
    ("high-noise", "low-noise"),
    ("high noise", "low noise"),
    ("highnoise", "lownoise"),
    ("_high", "_low"),
    ("-high", "-low"),
    (".high", ".low"),
    (" high", " low"),
    ("high_", "low_"),
    ("high-", "low-"),
    ("high.", "low."),
    ("high ", "low "),
    ("high/", "low/"),
    ("high\\", "low\\"),
    ("/high", "/low"),
    ("\\high", "\\low"),
]

# Single-letter (h/l) and abbreviated (hn/ln) markers, generated across the
# delimiter product so mixed forms like "_h." or "-h_" all match. These sit
# after the word tokens, so the more specific spelling always wins, and the
# existence guard keeps accidental substrings inert.
_ABBREV_DELIMS = "_-. /\\"
for _abbrev_high, _abbrev_low in (("hn", "ln"), ("h", "l")):
    for _left in _ABBREV_DELIMS:
        for _right in _ABBREV_DELIMS:
            _PAIR_TOKENS.append(
                (f"{_left}{_abbrev_high}{_right}", f"{_left}{_abbrev_low}{_right}")
            )

_NONE_VALUES = ("", "none", "None")


def _resolve_pair(lora_name, lower_to_actual):
    """Find a partner file for lora_name via token swapping.

    Returns (side, partner_name) where side is "high" or "low" describing
    which side LORA_NAME itself belongs to, or (None, None) when no
    existence-guarded swap resolves.
    """
    # A leading "/" sentinel makes string-start behave like a folder
    # boundary, so delimited tokens like "/h_" can match at position 0
    # (e.g. "h_lightning.safetensors" in the loras root).
    search = "/" + lora_name.lower()
    for high_token, low_token in _PAIR_TOKENS:
        for src, dst, side in (
            (high_token, low_token, "high"),
            (low_token, high_token, "low"),
        ):
            idx = search.find(src)
            while idx != -1:
                candidate = (search[:idx] + dst + search[idx + len(src):])[1:]
                partner = lower_to_actual.get(candidate)
                if partner is not None and partner != lora_name:
                    return side, partner
                idx = search.find(src, idx + 1)
    return None, None


class LoraPairLoader:
    """Dual-purpose LoRA pair stack for Wan 2.2 dual-expert models.

    Each row is either:
      * baked (mode "static", the default): applied to model_high /
        model_low at load, exactly like a normal LoRA loader; or
      * lane (mode "lane"): emitted on the lora_lanes output for
        MoESamplerDual to schedule, with an optional per-row CCN_CURVE
        socket (curve_<row id>) multiplying its strengths over global run
        progress. Lane rows do NOT patch the models, so wiring both the
        model outputs and lora_lanes can never double-apply a row.

    Pair resolution, the zero-weight side skip, and the missing-file error
    behave identically in both modes. An optional `lanes` input chains
    another loader's lanes ahead of this one's.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high": ("MODEL", {
                    "tooltip": "High-noise expert model.",
                }),
                "model_low": ("MODEL", {
                    "tooltip": "Low-noise expert model.",
                }),
            },
            "optional": FlexibleOptionalInputType(_ANY, {
                "lanes": ("CCN_LORA_LANES", {
                    "tooltip": "Chain lanes from another LoRA Pair Loader; "
                               "prepended to this node's lane rows.",
                }),
            }),
        }

    RETURN_TYPES = ("MODEL", "MODEL", "CCN_LORA_LANES", "STRING")
    RETURN_NAMES = ("model_high", "model_low", "lora_lanes", "info")
    FUNCTION = "load"
    CATEGORY = "CCN"
    DESCRIPTION = (
        "Dynamic LoRA pair stack: baked rows patch both experts with "
        "independent high/low weights; lane rows emit schedulable lanes "
        "(with optional per-row strength curves) for MoE Sampler Dual. "
        "HIGH/LOW partners auto-resolve from filenames; weight 0 skips a side."
    )

    def load(self, model_high, model_low, **kwargs):
        lora_names = folder_paths.get_filename_list("loras")
        lower_to_actual = {name.lower(): name for name in lora_names}
        available = set(lora_names)

        rows = []
        for key in sorted(kwargs):
            value = kwargs[key]
            if isinstance(value, dict) and "lora" in value:
                rows.append((key, value))

        chain = kwargs.get("lanes")
        lanes_out = list(chain) if isinstance(chain, (list, tuple)) else []
        info_lines = []
        row_number = 0
        for key, value in rows:
            row_number += 1
            label = f"row {row_number}"
            mode = value.get("mode", "static")

            if not value.get("on", True):
                info_lines.append(f"{label}: off")
                continue

            lora_name = value.get("lora")
            if not isinstance(lora_name, str) or lora_name in _NONE_VALUES:
                info_lines.append(f"{label}: no LoRA selected, skipped")
                continue

            try:
                strength_high = float(value.get("strength_high", 1.0))
                strength_low = float(value.get("strength_low", 1.0))
            except (TypeError, ValueError):
                raise ValueError(
                    f"LoraPairLoader: {label} ({lora_name!r}) has non-numeric "
                    f"strengths: {value.get('strength_high')!r} / "
                    f"{value.get('strength_low')!r}"
                )

            if strength_high == 0.0 and strength_low == 0.0:
                info_lines.append(f"{label}: {lora_name} -- both weights 0, skipped")
                continue

            if lora_name not in available:
                raise ValueError(
                    f"LoraPairLoader: {label} references {lora_name!r}, which is "
                    "not in the loras folder. Refresh the node or fix the row; "
                    "refusing to run with a silently missing LoRA."
                )

            side, partner = _resolve_pair(lora_name, lower_to_actual)
            if partner is not None:
                if side == "high":
                    file_high, file_low = lora_name, partner
                else:
                    file_high, file_low = partner, lora_name
                pair_note = f"pair ({file_high} / {file_low})"
            else:
                file_high = file_low = lora_name
                pair_note = "single file -> both experts"

            if mode == "lane":
                suffix = key.split("_", 1)[1] if "_" in key else key
                curve = kwargs.get(f"curve_{suffix}")
                lanes_out.append({
                    "name": os.path.splitext(os.path.basename(lora_name))[0],
                    "file_high": file_high,
                    "file_low": file_low,
                    "strength_high": strength_high,
                    "strength_low": strength_low,
                    "curve": curve,
                })
                curve_note = "curved" if curve is not None else "constant"
                info_lines.append(
                    f"{label}: {lora_name} [lane] -- {pair_note}; "
                    f"high @{strength_high:g}, low @{strength_low:g} "
                    f"({curve_note})")
                continue

            applied = []
            if strength_high != 0.0:
                model_high = self._apply(model_high, file_high, strength_high, label)
                applied.append(f"high @{strength_high:g}")
            if strength_low != 0.0:
                model_low = self._apply(model_low, file_low, strength_low, label)
                applied.append(f"low @{strength_low:g}")

            info_lines.append(
                f"{label}: {lora_name} [baked] -- {pair_note}; "
                f"{', '.join(applied)}")

        if row_number == 0 and not lanes_out:
            info_lines.append("no rows -- models passed through unchanged")

        return (model_high, model_low, lanes_out, "\n".join(info_lines))

    @staticmethod
    def _apply(model, lora_name, strength, label):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        try:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        except Exception as exc:
            raise ValueError(
                f"LoraPairLoader: {label} failed to load {lora_name!r}: {exc}"
            ) from exc
        patched_model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
        if patched_model is None:
            logging.warning(
                "LoraPairLoader: %s produced no model patches from %s "
                "(no matching keys?); passing model through", label, lora_name
            )
            return model
        return patched_model


# ----------------------------------------------------------------------
#  Chooser support: name + mtime listing for the row widget's search
#  dialog (js/lora_pair_loader.js), enabling date-sorted selection.
#  The route registers only when running inside the ComfyUI server; the
#  module imports cleanly without it (tests, CLI tooling), and the JS
#  falls back to the name-only /object_info list in that case.
# ----------------------------------------------------------------------

def lora_listing():
    """Return [{"name": rel_path, "mtime": float|None}] for the loras folder."""
    entries = []
    for name in folder_paths.get_filename_list("loras"):
        path = folder_paths.get_full_path("loras", name)
        if path is None:
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
        entries.append({"name": name, "mtime": mtime})
    return entries


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/ccn/loras")
    async def _ccn_lora_listing(request):
        return web.json_response(lora_listing())
except Exception:
    pass
