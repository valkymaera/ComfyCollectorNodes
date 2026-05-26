"""
#q #cD #d
Neutral Prompt Guider for ComfyUI.
Implements perpendicular (Perp-Neg), salient blending,  
and top-k filtering as a native Guider, compatible      
with SamplerCustomAdvanced.

Extends CurveCFG so every feature of the Curve CFG Guider is
available — set min_cfg == max_cfg for flat/fixed CFG.

All conditioning (positive, negative, and every aux entry) is
batched into a single calc_cond_batch call and prepared by the
standard guider pipeline, so ControlNet, IP-Adapter, area
conditioning, timestep ranges, hooks, etc. all work correctly.

Architecture
------------
NeutralPromptEntry  —  packages one CONDITIONING + strategy config
                       into an NP_ENTRIES list.  Chainable: connect
                       the 'entries' output of one to the 'entries'
                       input of the next to accumulate multiple aux
                       conditionings with independent strategies.

NeutralPromptGuider —  creates the compound Guider.  Without any
                       NP_ENTRIES connected it behaves identically
                       to CurveCFGGuider.

References:
  Perp-Neg          https://perp-neg.github.io/
  Magic Fusion      https://magicfusion.github.io/
  Semantic Guidance  https://arxiv.org/abs/2301.12247
  CFG Rescale        https://arxiv.org/abs/2305.08891
"""

import json
import torch
import comfy.samplers
from .curve_cfg_guider import CurveCFG, hermite, DEFAULT_CURVE


# ---------------------------------------------------------------------------
# Core math  (ported from sd-webui-neutral-prompt by ljleb)
# ---------------------------------------------------------------------------

def get_perpendicular_component(normal: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Return the component of *vector* orthogonal to *normal*.

    The parallel component is projected out so that the aux conditioning
    contributes nothing in directions already covered by the main prompt.
    """
    if (normal == 0).all():
        return vector
    return vector - normal * (torch.sum(normal * vector) / torch.norm(normal) ** 2)


def salient_blend(normal: torch.Tensor, vectors: list) -> torch.Tensor:
    """Per-element salience competition.

    For every element the source with the largest salience (softmax of
    absolute value) wins.  Aux vectors only replace the main signal where
    they activate more strongly.
    """
    salience_maps = [_get_salience(normal)] + [_get_salience(v) for v, _ in vectors]
    mask = torch.argmax(torch.stack(salience_maps, dim=0), dim=0)

    result = torch.zeros_like(normal)
    for mask_i, (vector, weight) in enumerate(vectors, start=1):
        vector_mask = (mask == mask_i).float()
        result += weight * vector_mask * (vector - normal)
    return result


def _get_salience(vector: torch.Tensor) -> torch.Tensor:
    return torch.softmax(torch.abs(vector).flatten(), dim=0).reshape_as(vector)


def filter_abs_top_k(vector: torch.Tensor, k_ratio: float) -> torch.Tensor:
    """Zero out everything except the top *k_ratio* fraction by abs value."""
    k = int(torch.numel(vector) * (1 - k_ratio))
    threshold, _ = torch.kthvalue(torch.abs(vector.flatten()), k)
    return vector * (torch.abs(vector) >= threshold).to(vector.dtype)


def apply_cfg_rescale(
    cfg_result: torch.Tensor,
    cond_result: torch.Tensor,
    rescale: float,
) -> torch.Tensor:
    """Std-dev based CFG rescaling (arxiv 2305.08891 §3.4)."""
    if rescale <= 0:
        return cfg_result
    cfg_mean = cfg_result.mean()
    rescale_mean = (1 - rescale) * cfg_mean + rescale * cond_result.mean()
    rescale_factor = rescale * (cond_result.std() / cfg_result.std() - 1) + 1
    return rescale_mean + (cfg_result - cfg_mean) * rescale_factor


# ---------------------------------------------------------------------------
# Guider
# ---------------------------------------------------------------------------

class CurveNeutralCFG(CurveCFG):
    """CFGGuider with curve-scheduled CFG + neutral prompt strategies.

    Extends CurveCFG so all curve/sigma-decay features are inherited.
    When no NP entries are present, falls back to the standard CFGGuider
    path (via CurveCFG.predict_noise → sampling_function).
    """

    def __init__(self, model):
        super().__init__(model)
        self.np_entries = []
        self.np_cfg_rescale = 0.0
        self.np_debug = False

    def set_neutral_params(self, entries, cfg_rescale=0.0, debug=False):
        # Shallow-copy each entry dict so we don't mutate the node output
        self.np_entries = [dict(e) for e in (entries or [])]
        self.np_cfg_rescale = cfg_rescale
        self.np_debug = debug

        # Register aux conditionings with the guider's conds system.
        # This is the key advantage over the old sampler_cfg_function
        # approach: prepare_sampling will process these through the
        # same pipeline as positive/negative (areas, masks, timestep
        # ranges, encode_model_conds, hooks, etc.).
        aux_conds = {}
        for i, entry in enumerate(self.np_entries):
            key = f"np_aux_{i}"
            aux_conds[key] = entry["conditioning"]
            entry["_cond_key"] = key
        if aux_conds:
            self.inner_set_conds(aux_conds)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        # ── 1. Curve-based CFG ──────────────────────────────────────
        t, sigma = self._current_progress(timestep)
        blend = hermite(self.curve_keys, t)
        cfg = self.min_cfg + (self.max_cfg - self.min_cfg) * blend
        if self.sigma_decay and self.sigma_max > 1e-10:
            s = sigma / self.sigma_max
            cfg = 1.0 + (cfg - 1.0) * s

        # ── 2. No aux entries → standard CurveCFG path ─────────────
        if not self.np_entries:
            self.cfg = cfg
            # Skip CurveCFG.predict_noise (which would recompute the
            # curve) and go straight to CFGGuider.predict_noise
            return comfy.samplers.sampling_function(
                self.inner_model, x, timestep,
                self.conds.get("negative"),
                self.conds.get("positive"),
                cfg, model_options=model_options, seed=seed,
            )

        # ── 3. Batch ALL conditionings in one forward pass ──────────
        neg = self.conds.get("negative")
        pos = self.conds.get("positive")
        batch = [neg, pos]
        for entry in self.np_entries:
            batch.append(self.conds.get(entry["_cond_key"]))

        out = comfy.samplers.calc_cond_batch(
            self.inner_model, batch, x, timestep, model_options
        )

        uncond_pred = out[0]   # denoised x₀ for negative conditioning
        cond_pred   = out[1]   # denoised x₀ for positive conditioning
        cond_delta  = cond_pred - uncond_pred

        # ── 4. Strategy modifications ───────────────────────────────
        pos_mod = torch.zeros_like(cond_delta)
        neg_mod = torch.zeros_like(uncond_pred)
        pos_salient_pairs = []
        neg_salient_pairs = []

        for i, entry in enumerate(self.np_entries):
            aux_pred  = out[2 + i]
            aux_delta = aux_pred - uncond_pred

            strat  = entry["strategy"]
            w      = entry["weight"]
            is_neg = entry["side"] == "negative"
            ref    = -cond_delta if is_neg else cond_delta

            if strat == "perpendicular":
                comp = w * get_perpendicular_component(ref, aux_delta)
                if is_neg:
                    neg_mod -= comp
                else:
                    pos_mod += comp

            elif strat == "salient":
                target = neg_salient_pairs if is_neg else pos_salient_pairs
                target.append((aux_delta, w))

            elif strat == "top_k":
                filt = w * filter_abs_top_k(aux_delta, entry.get("k_ratio", 0.05))
                if is_neg:
                    neg_mod -= filt
                else:
                    pos_mod += filt

            if self.np_debug:
                print(
                    f"[NeutralPromptGuider] {strat}/{entry['side']} "
                    f"| w={w:.2f} "
                    f"| aux_norm={aux_delta.norm().item():.4f} "
                    f"| ref_norm={ref.norm().item():.4f}"
                )

        if pos_salient_pairs:
            pos_mod += salient_blend(cond_delta, pos_salient_pairs)
        if neg_salient_pairs:
            neg_mod -= salient_blend(-cond_delta, neg_salient_pairs)

        # ── 5. CFG combination ──────────────────────────────────────
        enhanced_cond   = cond_pred   + pos_mod
        enhanced_uncond = uncond_pred + neg_mod
        result = enhanced_uncond + cfg * (enhanced_cond - enhanced_uncond)

        # ── 6. CFG rescale ──────────────────────────────────────────
        if self.np_cfg_rescale > 0:
            result = apply_cfg_rescale(result, enhanced_cond, self.np_cfg_rescale)

        # ── 7. Post-CFG hooks (compatibility with other nodes) ──────
        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": result,
                "cond": pos, "uncond": neg,
                "cond_scale": cfg,
                "model": self.inner_model,
                "uncond_denoised": uncond_pred,
                "cond_denoised": cond_pred,
                "sigma": timestep,
                "model_options": model_options,
                "input": x,
            }
            result = fn(args)

        return result


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class NeutralPromptEntry:
    """Packages one auxiliary CONDITIONING with a strategy selection.

    Chain multiple entries: connect the *entries* output of one node
    to the *entries* input of the next.  The accumulated list feeds
    into a NeutralPromptGuider.
    """

    STRATEGIES = ["perpendicular", "salient", "top_k"]
    SIDES      = ["positive", "negative"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "strategy": (cls.STRATEGIES, {
                    "tooltip": (
                        "perpendicular — Perp-Neg: removes the component "
                        "that conflicts with the main prompt.\n"
                        "salient — per-element competition: aux wins only "
                        "where it activates more strongly.\n"
                        "top_k — keeps only the strongest fraction of the "
                        "aux contribution."
                    ),
                }),
                "side": (cls.SIDES, {
                    "tooltip": (
                        "positive — strategy merges with the positive prompt.\n"
                        "negative — strategy merges with the negative prompt "
                        "(e.g. orthogonal negatives that don't interfere)."
                    ),
                }),
                "weight": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Strength of this auxiliary conditioning.",
                }),
            },
            "optional": {
                "k_ratio": ("FLOAT", {
                    "default": 0.05, "min": 0.001, "max": 1.0, "step": 0.001,
                    "tooltip": "Top-K only: fraction of elements to keep (0.05 = top 5%).",
                }),
                "entries": ("NP_ENTRIES", {
                    "tooltip": "Chain from another NeutralPromptEntry to accumulate multiple strategies.",
                }),
            },
        }

    RETURN_TYPES  = ("NP_ENTRIES",)
    RETURN_NAMES  = ("entries",)
    FUNCTION      = "build"
    CATEGORY      = "CCN"
    DESCRIPTION   = (
        "Neutral Prompt Entry — pair an auxiliary conditioning with a "
        "strategy (perpendicular / salient / top_k).  Chain multiples "
        "and connect to a Neutral Prompt Guider."
    )

    def build(self, conditioning, strategy, side, weight, k_ratio=0.05, entries=None):
        entry = {
            "conditioning": conditioning,
            "strategy": strategy,
            "side": side,
            "weight": weight,
            "k_ratio": k_ratio,
        }
        result = list(entries) if entries else []
        result.append(entry)
        return (result,)


class NeutralPromptEmpty:
    """Outputs an empty NP_ENTRIES list (no-op).

    Use as the disabled branch of a switch so the guider always
    receives a valid NP_ENTRIES input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES  = ("NP_ENTRIES",)
    RETURN_NAMES  = ("entries",)
    FUNCTION      = "build"
    CATEGORY      = "CCN"
    DESCRIPTION   = "Neutral Prompt Empty — outputs an empty entry list (no-op bypass)."

    def build(self):
        return ([],)


class NeutralPromptGuider:
    """CFG Guider with curve-scheduled CFG and neutral prompt strategies.

    Without NP_ENTRIES connected this behaves identically to
    CurveCFGGuider.  With entries it applies the selected strategy
    (perpendicular / salient / top_k) during the CFG step.

    Set min_cfg == max_cfg for flat (non-curved) CFG.

    Use with SamplerCustomAdvanced.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sigmas": ("SIGMAS",),
                "min_cfg": ("FLOAT", {
                    "default": 1.0, "step": 0.001,
                    "tooltip": "CFG value when the curve outputs 0.",
                }),
                "max_cfg": ("FLOAT", {
                    "default": 7.0, "step": 0.001,
                    "tooltip": "CFG value when the curve outputs 1.",
                }),
                "mode": (["step", "sigma"], {
                    "tooltip": "How denoising progress is measured for the curve.",
                }),
                "sigma_decay": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Attenuate guidance toward 1.0 as sigma decreases.",
                }),
                "cfg_rescale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Std-dev based CFG rescaling (0 = disabled).  Reduces over-exposure at high CFG.",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print per-step strategy diagnostics to the console.",
                }),
                "curve_data": ("STRING", {
                    "default": DEFAULT_CURVE, "multiline": False,
                }),
            },
            "optional": {
                "curve": ("CURVE",),
                "np_entries": ("NP_ENTRIES", {
                    "tooltip": "Neutral prompt entries from NeutralPromptEntry nodes.",
                }),
            },
        }

    RETURN_TYPES  = ("GUIDER", "SIGMAS")
    RETURN_NAMES  = ("guider", "sigmas")
    FUNCTION      = "get_guider"
    CATEGORY      = "CCN"
    DESCRIPTION   = (
        "Neutral Prompt Guider — curve-scheduled CFG with optional "
        "perpendicular / salient / top-k conditioning strategies.  "
        "Drop-in superset of CurveCFGGuider.  Use with SamplerCustomAdvanced."
    )

    def get_guider(
        self,
        model, positive, negative, sigmas,
        min_cfg, max_cfg, mode, sigma_decay,
        cfg_rescale, debug, curve_data,
        curve=None, np_entries=None,
    ):
        # ── Parse curve ─────────────────────────────────────────────
        if curve is not None:
            keys = curve
        else:
            try:
                keys = json.loads(curve_data)
            except (json.JSONDecodeError, TypeError):
                keys = json.loads(DEFAULT_CURVE)

        keys = sorted(keys, key=lambda k: k["x"])

        # ── Build guider ────────────────────────────────────────────
        guider = CurveNeutralCFG(model)
        guider.set_conds(positive, negative)
        guider.set_curve_params(keys, min_cfg, max_cfg, mode, sigmas, sigma_decay)

        if np_entries:
            guider.set_neutral_params(np_entries, cfg_rescale, debug)

        return (guider, sigmas)
