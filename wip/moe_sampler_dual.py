"""
MoE Sampler Dual -- all-in-one two-expert sampler for Wan 2.2.

Splits the incoming sigma schedule at the trained expert boundary (via
MoESigmaSplit), then runs the high-noise expert over the first segment and
the low-noise expert over the remainder, chaining the latent between
phases exactly like the two-SamplerCustomAdvanced wiring -- but with the
degenerate cases handled internally: an empty phase is simply skipped, and
in the all-low case the initial noise is injected at the low phase where
it belongs. No stub schedules ever execute.

CFG per phase reuses the CurveCFG guider:
  * curve connected (CCN_CURVE): cfg = lerp(cfg_min, cfg, hermite(curve, t))
    where t is that PHASE's local progress (0 at phase start, 1 at phase
    end), measured per curve_mode ("step" or "sigma"), with optional
    sigma-proportional decay -- identical semantics to CurveCFGGuider.
  * no curve: constant cfg for that phase (cfg_min unused).

LoRA lanes (optional lora_lanes input, built by LoRA Pair Lane): each lane
is a LoRA pair with peak strengths and an optional curve multiplying them
over GLOBAL run progress. Curves compile to piecewise-constant segments:
each phase is divided into at most lane_segments chunks, every lane's
strength is evaluated at each chunk's midpoint, and adjacent chunks with
identical strengths merge. Each surviving segment samples on a freshly
patched model clone (same load_lora_for_models path as any LoRA loader).
Trade-offs by construction: repatch count is bounded by lane_segments per
phase, and multistep samplers reset history at segment boundaries -- the
same reset every Wan 2.2 workflow already accepts once at the expert
boundary. CFG progress remains phase-relative regardless of segmentation.

The optional SAMPLER input overrides the sampler_name dropdown.

The cockpit widget (js/wip/moe_sampler_dual.js) previews the schedule, split,
CFG lanes, and LoRA lane staircase live by walking upstream widget values;
after each run this node returns the authoritative schedule and compiled
lane segments in the ui payload ("ccn_moe") so the widget can true itself
up and flag divergence.
"""

import torch

import folder_paths
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import latent_preview

from .moe_sigma_split import MoESigmaSplit, _BOUNDARY_PRESETS, _CUSTOM_LABEL
from ..nodes.curve_cfg_guider import CurveCFG, hermite

# Flat curve at blend 1.0: CurveCFG then yields max_cfg at every step, which
# is how the constant-CFG (no curve connected) path is expressed.
_CONSTANT_KEYS = [
    {"x": 0.0, "y": 1.0, "in": 0.0, "out": 0.0, "mirrored": True},
    {"x": 1.0, "y": 1.0, "in": 0.0, "out": 0.0, "mirrored": True},
]

_STRENGTH_EPS = 1e-6


class MoESamplerDual:
    """Two-phase MoE sampler with per-phase curve CFG and LoRA lanes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high": ("MODEL", {"tooltip": "High-noise expert model."}),
                "model_low": ("MODEL", {"tooltip": "Low-noise expert model."}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sigmas": ("SIGMAS", {
                    "tooltip": "Full schedule from an upstream scheduler; split "
                               "internally at the expert boundary.",
                }),
                "latent_image": ("LATENT",),
                "noise_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "add_noise": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Disable only when the incoming latent is already "
                               "noised for this schedule.",
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "boundary": (list(_BOUNDARY_PRESETS) + [_CUSTOM_LABEL], {
                    "default": "i2v (0.900)",
                    "tooltip": "Trained expert boundary (t = sigma * 1000).",
                }),
                "custom_boundary": ("FLOAT", {
                    "default": 0.875, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Used only when boundary = custom.",
                }),
                "curve_mode": (["step", "sigma"], {
                    "tooltip": "How progress is measured for CFG curves "
                               "(per-phase) and LoRA lane curves (global).",
                }),
                "sigma_decay": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Attenuate effective CFG toward 1.0 as sigma falls "
                               "within each phase (matches CurveCFGGuider).",
                }),
                "cfg_high": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.001}),
                "cfg_high_min": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.001,
                    "tooltip": "Curve floor for the high phase; unused without a curve.",
                }),
                "cfg_low": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.001}),
                "cfg_low_min": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.001,
                    "tooltip": "Curve floor for the low phase; unused without a curve.",
                }),
                "lane_segments": ("INT", {
                    "default": 4, "min": 1, "max": 8,
                    "tooltip": "Max LoRA lane segments per phase (repatch budget). "
                               "Used only when lora_lanes is connected.",
                }),
            },
            "optional": {
                "curve_high": ("CCN_CURVE",),
                "curve_low": ("CCN_CURVE",),
                "lora_lanes": ("CCN_LORA_LANES", {
                    "tooltip": "Schedulable LoRA lanes from LoRA Pair Lane.",
                }),
                "sampler": ("SAMPLER", {
                    "tooltip": "Overrides sampler_name when connected.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "sample"
    CATEGORY = "CCN"
    DESCRIPTION = (
        "All-in-one Wan 2.2 MoE sampler: splits the schedule at the trained "
        "expert boundary, runs high then low expert with per-phase curve-driven "
        "CFG and curve-scheduled LoRA lanes, and handles empty phases and noise "
        "placement internally."
    )

    def sample(self, model_high, model_low, positive, negative, sigmas,
               latent_image, noise_seed, add_noise, sampler_name, boundary,
               custom_boundary, curve_mode, sigma_decay, cfg_high, cfg_high_min,
               cfg_low, cfg_low_min, lane_segments, curve_high=None,
               curve_low=None, lora_lanes=None, sampler=None):

        (sigmas_high, sigmas_low, high_steps, low_steps,
         switch_sigma, split_info) = MoESigmaSplit().split(
            sigmas, boundary, custom_boundary)

        lanes = self._validated_lanes(lora_lanes)
        sampler_obj = sampler if sampler is not None \
            else comfy.samplers.sampler_object(sampler_name)

        latent = latent_image
        x = comfy.sample.fix_empty_latent_channels(model_high, latent["samples"])
        noise_mask = latent.get("noise_mask", None)

        if add_noise:
            pending_noise = comfy.sample.prepare_noise(
                x, noise_seed, latent.get("batch_index", None))
        else:
            pending_noise = torch.zeros(
                x.size(), dtype=x.dtype, layout=x.layout, device="cpu")
        zero_noise = torch.zeros(
            x.size(), dtype=x.dtype, layout=x.layout, device="cpu")

        flat = sigmas.flatten()
        g_smax, g_smin = float(flat[0]), float(flat[-1])
        total_steps = flat.numel() - 1
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        info_lines = [split_info]
        lora_cache = {}
        ui_lanes = {"names": [lane["name"] for lane in lanes],
                    "high": [], "low": []} if lanes else None

        phases = []
        if high_steps > 0:
            phases.append(("high", model_high, sigmas_high, high_steps, 0,
                           cfg_high, cfg_high_min, curve_high, 0))
        if low_steps > 0:
            phases.append(("low", model_low, sigmas_low, low_steps,
                           high_steps, cfg_low, cfg_low_min, curve_low, 1))

        for (phase, base_model, phase_sigmas, steps, offset,
             cfg_max, cfg_min, cfg_curve, side_idx) in phases:

            info_lines.append(self._cfg_line(
                phase, cfg_max, cfg_min, cfg_curve, curve_mode, sigma_decay))

            if lanes:
                segments = self._compile_segments(
                    steps, offset, total_steps, phase_sigmas,
                    g_smax, g_smin, lanes, lane_segments, curve_mode)
                ui_lanes[phase] = [
                    {"start": s["start"], "end": s["end"],
                     "strengths": [list(v) for v in s["strengths"]]}
                    for s in segments]
                info_lines.append(self._segment_line(phase, lanes, segments))
            else:
                segments = [{"start": 0, "end": steps, "strengths": None}]

            for seg in segments:
                seg_sigmas = phase_sigmas[seg["start"]: seg["end"] + 1]
                seg_steps = seg["end"] - seg["start"]
                if seg_steps <= 0:
                    continue

                if seg["strengths"] is None:
                    model = base_model
                else:
                    model = self._lane_model(
                        base_model, lanes, seg["strengths"], side_idx,
                        lora_cache, phase)

                guider = self._build_guider(
                    model, positive, negative, phase_sigmas,
                    cfg_max, cfg_min, cfg_curve, curve_mode, sigma_decay,
                    phase)
                callback = latent_preview.prepare_callback(
                    guider.model_patcher, seg_steps)
                x = guider.sample(
                    pending_noise, x, sampler_obj, seg_sigmas,
                    denoise_mask=noise_mask, callback=callback,
                    disable_pbar=disable_pbar, seed=noise_seed)
                pending_noise = zero_noise

        x = x.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = x
        info = "\n".join(info_lines)

        ui_payload = {
            "sigmas": [round(float(s), 6) for s in flat.tolist()],
            "boundary": self._boundary_value(boundary, custom_boundary),
            "high_steps": int(high_steps),
            "low_steps": int(low_steps),
            "switch_sigma": float(switch_sigma),
        }
        if ui_lanes is not None:
            ui_payload["lanes"] = ui_lanes
        return {"ui": {"ccn_moe": [ui_payload]}, "result": (out, info)}

    # ------------------------------------------------------------------
    #  LoRA lane compilation
    # ------------------------------------------------------------------

    def _validated_lanes(self, lora_lanes):
        if not lora_lanes:
            return []
        if not isinstance(lora_lanes, (list, tuple)):
            raise ValueError(
                "MoESamplerDual: lora_lanes must be a CCN_LORA_LANES list; "
                f"got {type(lora_lanes).__name__}")
        lanes = []
        for i, lane in enumerate(lora_lanes):
            if not isinstance(lane, dict) or "file_high" not in lane \
                    or "file_low" not in lane:
                raise ValueError(
                    f"MoESamplerDual: lane {i + 1} is not a valid lane entry: "
                    f"{lane!r}")
            entry = dict(lane)
            entry["strength_high"] = float(lane.get("strength_high", 1.0))
            entry["strength_low"] = float(lane.get("strength_low", 1.0))
            entry["name"] = str(lane.get("name", f"lane {i + 1}"))
            if lane.get("curve") is not None:
                entry["curve"] = self._validated_keys(
                    lane["curve"], f"lane '{entry['name']}'")
            else:
                entry["curve"] = None
            lanes.append(entry)
        return lanes

    @staticmethod
    def _lane_strengths(lanes, t):
        out = []
        for lane in lanes:
            mult = 1.0 if lane["curve"] is None else hermite(lane["curve"], t)
            out.append((round(lane["strength_high"] * mult, 4),
                        round(lane["strength_low"] * mult, 4)))
        return out

    def _compile_segments(self, phase_len, phase_offset, total_steps,
                          phase_sigmas, g_smax, g_smin, lanes, seg_count,
                          mode):
        k = max(1, min(int(seg_count), phase_len))
        bounds = sorted({round(i * phase_len / k) for i in range(k + 1)})
        if bounds[0] != 0:
            bounds.insert(0, 0)
        if bounds[-1] != phase_len:
            bounds.append(phase_len)

        segments = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b <= a:
                continue
            if mode == "sigma":
                mid_sigma = float(phase_sigmas[(a + b) // 2])
                rng = g_smax - g_smin
                t = (g_smax - mid_sigma) / rng if rng > 1e-10 else 0.0
            else:
                t = (phase_offset + (a + b) / 2.0) / max(total_steps, 1)
            t = max(0.0, min(1.0, t))
            segments.append({
                "start": a, "end": b,
                "strengths": self._lane_strengths(lanes, t),
            })

        merged = [segments[0]]
        for seg in segments[1:]:
            if seg["strengths"] == merged[-1]["strengths"]:
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(seg)
        return merged

    def _lane_model(self, base_model, lanes, strengths, side_idx, lora_cache,
                    phase):
        model = base_model
        for lane, pair in zip(lanes, strengths):
            strength = pair[side_idx]
            if abs(strength) < _STRENGTH_EPS:
                continue
            fname = lane["file_high"] if side_idx == 0 else lane["file_low"]
            if not fname:
                continue
            path = folder_paths.get_full_path_or_raise("loras", fname)
            lora = lora_cache.get(path)
            if lora is None:
                try:
                    lora = comfy.utils.load_torch_file(path, safe_load=True)
                except Exception as exc:
                    raise ValueError(
                        f"MoESamplerDual: lane {lane['name']!r} failed to load "
                        f"{fname!r} for the {phase} phase: {exc}") from exc
                lora_cache[path] = lora
            model, _ = comfy.sd.load_lora_for_models(
                model, None, lora, strength, 0)
        return model

    @staticmethod
    def _segment_line(phase, lanes, segments):
        parts = []
        for seg in segments:
            vals = ", ".join(
                f"{lane['name']} {seg['strengths'][i][0 if phase == 'high' else 1]:g}"
                for i, lane in enumerate(lanes))
            parts.append(f"steps {seg['start']}-{seg['end'] - 1}: [{vals}]")
        return f"{phase} lanes: {len(segments)} segment(s) -- " + "; ".join(parts)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _boundary_value(boundary, custom_boundary):
        if boundary == _CUSTOM_LABEL:
            return float(custom_boundary)
        return _BOUNDARY_PRESETS.get(boundary, float(custom_boundary))

    @staticmethod
    def _validated_keys(curve_keys, label):
        if not isinstance(curve_keys, (list, tuple)) or len(curve_keys) == 0:
            raise ValueError(
                f"MoESamplerDual: {label} must be a CCN_CURVE key list; "
                f"got {type(curve_keys).__name__}")
        keys = []
        for k in curve_keys:
            if not isinstance(k, dict) or "x" not in k or "y" not in k:
                raise ValueError(
                    f"MoESamplerDual: {label} contains an invalid key "
                    f"entry: {k!r}")
            keys.append({
                "x": float(k["x"]), "y": float(k["y"]),
                "in": float(k.get("in", 0.0)), "out": float(k.get("out", 0.0)),
                "mirrored": bool(k.get("mirrored", True)),
            })
        keys.sort(key=lambda key: key["x"])
        return keys

    def _build_guider(self, model, positive, negative, phase_sigmas,
                      cfg_max, cfg_min, curve_keys, mode, sigma_decay, phase):
        if curve_keys is None:
            keys = [dict(k) for k in _CONSTANT_KEYS]
            min_cfg = max_cfg = float(cfg_max)
        else:
            keys = self._validated_keys(curve_keys, f"curve_{phase}")
            min_cfg = float(cfg_min)
            max_cfg = float(cfg_max)

        guider = CurveCFG(model)
        guider.set_conds(positive, negative)
        guider.set_curve_params(
            keys, min_cfg, max_cfg, mode, phase_sigmas, sigma_decay)
        return guider

    @staticmethod
    def _cfg_line(phase, cfg_max, cfg_min, curve_keys, mode, sigma_decay):
        if curve_keys is None:
            base = f"{phase} cfg: constant {float(cfg_max):g}"
        else:
            base = (f"{phase} cfg: curve [{float(cfg_min):g} .. "
                    f"{float(cfg_max):g}], mode={mode}")
        if sigma_decay:
            base += ", sigma_decay"
        return base
