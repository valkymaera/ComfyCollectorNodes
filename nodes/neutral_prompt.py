"""
Neutral Prompt node for ComfyUI.

Ported from sd-webui-neutral-prompt (A1111 extension by ljleb).
Implements perpendicular (Perp-Neg), salient blending, and top-k
filtering strategies for multi-conditioning combination during CFG.

Works with any model type including video models — all operations
are tensor-shape agnostic.

References:
  Perp-Neg          https://perp-neg.github.io/
  Magic Fusion       https://magicfusion.github.io/
  Semantic Guidance   https://arxiv.org/abs/2301.12247
  CFG Rescale         https://arxiv.org/abs/2305.08891
"""

import torch
import comfy.samplers
import comfy.sampler_helpers


# ---------------------------------------------------------------------------
# Conditioning preparation
# ---------------------------------------------------------------------------

def _prepare_cond(raw_cond, model, x, model_options):
    """
    Convert raw node conditioning  [(tensor, dict), ...]  into the fully
    processed list-of-dicts that calc_cond_batch expects.

    This replicates the essential steps of the normal pipeline:
      convert_cond  →  resolve areas  →  timestep ranges  →  encode_model_conds
    """
    converted = comfy.sampler_helpers.convert_cond(raw_cond)

    comfy.samplers.resolve_areas_and_cond_masks_multidim(
        converted, x.shape[2:], x.device
    )
    comfy.samplers.calculate_start_end_timesteps(model, converted)

    if hasattr(model, 'extra_conds'):
        converted = comfy.samplers.encode_model_conds(
            model.extra_conds, converted, x, x.device, "positive"
        )

    return converted


# ---------------------------------------------------------------------------
# Core math  (ported from the A1111 extension by ljleb)
# ---------------------------------------------------------------------------

def get_perpendicular_component(normal: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """
    Project out the component of *vector* that lies along *normal*.
    Returns only the perpendicular remainder so that the aux conditioning
    contributes nothing in directions already covered by the main prompt.
    """
    if (normal == 0).all():
        return vector
    return vector - normal * (torch.sum(normal * vector) / torch.norm(normal) ** 2)


def salient_blend(normal: torch.Tensor, vectors: list) -> torch.Tensor:
    """
    Per-element salience competition.  For every element the source with
    the highest salience (softmax of absolute value) wins.  Aux vectors
    only replace the main signal where they activate more strongly.
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
    """Zero out everything except the top *k_ratio* fraction by absolute value."""
    k = int(torch.numel(vector) * (1 - k_ratio))
    threshold, _ = torch.kthvalue(torch.abs(vector.flatten()), k)
    return vector * (torch.abs(vector) >= threshold).to(vector.dtype)


def apply_cfg_rescale(
    cfg_result: torch.Tensor,
    cond_result: torch.Tensor,
    rescale: float,
) -> torch.Tensor:
    """Standard-deviation based CFG rescaling (arxiv 2305.08891 §3.4)."""
    if rescale <= 0:
        return cfg_result
    cfg_mean = cfg_result.mean()
    rescale_mean = (1 - rescale) * cfg_mean + rescale * cond_result.mean()
    rescale_factor = rescale * (cond_result.std() / cfg_result.std() - 1) + 1
    return rescale_mean + (cfg_result - cfg_mean) * rescale_factor


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class NeutralPrompt:
    """
    Patches a model to apply an advanced conditioning-combination strategy
    during the CFG step of sampling.

    Strategies
    ----------
    perpendicular  –  Perp-Neg orthogonalisation.  Removes the component of
        the aux prediction that conflicts with the main prompt so that only
        novel, non-contradicting information is added.
    salient  –  Per-pixel salience competition.  The aux conditioning only
        wins in spatial (or spatio-temporal) regions where it activates
        more strongly than the main prompt.
    top_k  –  Keeps only the top *k_ratio* fraction of the aux prediction
        (by absolute value) and adds them to the result.  Good for small
        details and targeted adjustments.

    Side
    ----
    positive  –  Strategy is applied to the positive (conditional) side of
        the CFG equation.  Aux conditioning merges with the main prompt.
    negative  –  Strategy is applied to the negative (unconditional) side.
        Aux conditioning merges with the negative prompt, allowing
        advanced negative prompt composition (e.g. perpendicular negatives
        that don't interfere with each other).

    Chaining
    --------
    Multiple NeutralPrompt nodes can be chained model → model.  Each node
    appends its entry; the final CFG function processes all of them.
    Positive and negative side entries can be freely mixed.
    """

    STRATEGIES = ["perpendicular", "salient", "top_k"]
    SIDES = ["positive", "negative"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "main_conditioning": ("CONDITIONING",),
                "aux_conditioning": ("CONDITIONING",),
                "strategy": (cls.STRATEGIES,),
                "side": (cls.SIDES, {
                    "tooltip": (
                        "Which side of the CFG equation to apply the "
                        "strategy to.  'positive' merges with the main "
                        "prompt; 'negative' merges with the negative prompt."
                    ),
                }),
                "weight": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Strength of the auxiliary conditioning effect",
                }),
            },
            "optional": {
                "k_ratio": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": (
                        "Top-K only: fraction of elements to keep "
                        "(0.05 = strongest 5%%)"
                    ),
                }),
                "cfg_rescale": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": (
                        "Std-dev based CFG rescaling (0 = disabled).  "
                        "Reduces over-exposure artifacts at high CFG."
                    ),
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print per-step diagnostics to the console",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "conditioning")
    FUNCTION = "apply"
    CATEGORY = "CCN"
    DESCRIPTION = (
        "Neutral Prompt — apply perpendicular, salient, or top-k "
        "conditioning strategies to positive or negative prompts "
        "during sampling.  Chain multiple nodes for multi-strategy setups."
    )

    # ------------------------------------------------------------------ #

    def apply(
        self,
        model,
        main_conditioning,
        aux_conditioning,
        strategy,
        side,
        weight,
        k_ratio=0.05,
        cfg_rescale=0.0,
        debug=False,
    ):
        patched = model.clone()

        # Accumulate entries when nodes are chained (model → model)
        prev_entries = list(
            patched.model_options.get("neutral_prompt_entries", [])
        )
        prev_rescale = patched.model_options.get(
            "neutral_prompt_cfg_rescale", 0.0
        )
        prev_debug = patched.model_options.get(
            "neutral_prompt_debug", False
        )

        entry = {
            "aux_cond": aux_conditioning,
            "strategy": strategy,
            "side": side,
            "weight": weight,
            "k_ratio": k_ratio,
        }
        all_entries = prev_entries + [entry]
        final_rescale = max(prev_rescale, cfg_rescale)
        final_debug = prev_debug or debug

        # ------ closure: called by the sampler on every step ---------- #

        # Cache for prepared conditioning (populated on first call)
        _prep_cache = {}

        def _cfg_function(args):
            # args["cond"] and args["uncond"] are DENOISED predictions
            # (i.e. x - model_output).  Our return value must also be
            # denoised — ComfyUI converts back via: cfg_result = x - ret.
            d_cond = args["cond"]
            d_uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            inner_model = args["model"]
            x = args["input"]
            sigma = args["sigma"]
            mopts = args["model_options"]

            # Positive side: strategies modify the cond delta
            cond_delta = d_cond - d_uncond
            pos_aux = torch.zeros_like(cond_delta)
            pos_salient_pairs = []

            # Negative side: strategies modify the uncond prediction
            neg_aux = torch.zeros_like(d_uncond)
            neg_salient_pairs = []

            for idx, e in enumerate(all_entries):
                # Prepare conditioning once, then cache for subsequent steps
                if idx not in _prep_cache:
                    _prep_cache[idx] = _prepare_cond(
                        e["aux_cond"], inner_model, x, mopts
                    )

                prepared = _prep_cache[idx]
                aux_out = comfy.samplers.calc_cond_batch(
                    inner_model, [prepared], x, sigma, mopts
                )
                # calc_cond_batch returns raw model predictions;
                # convert to denoised space to match d_cond / d_uncond
                d_aux = x - aux_out[0]
                aux_delta = d_aux - d_uncond

                strat = e["strategy"]
                w = e["weight"]
                is_neg = e["side"] == "negative"

                if is_neg:
                    ref_delta = -cond_delta
                else:
                    ref_delta = cond_delta

                if strat == "perpendicular":
                    component = w * get_perpendicular_component(
                        ref_delta, aux_delta
                    )
                    if is_neg:
                        neg_aux = neg_aux - component
                    else:
                        pos_aux = pos_aux + component

                elif strat == "salient":
                    if is_neg:
                        neg_salient_pairs.append((aux_delta, w))
                    else:
                        pos_salient_pairs.append((aux_delta, w))

                elif strat == "top_k":
                    filtered = w * filter_abs_top_k(aux_delta, e["k_ratio"])
                    if is_neg:
                        neg_aux = neg_aux - filtered
                    else:
                        pos_aux = pos_aux + filtered

                if final_debug:
                    print(
                        f"[NeutralPrompt] {strat}/{e['side']} "
                        f"| weight={w:.2f} "
                        f"| aux_norm={aux_delta.norm().item():.4f} "
                        f"| ref_norm={ref_delta.norm().item():.4f}"
                    )

            if pos_salient_pairs:
                pos_aux = pos_aux + salient_blend(
                    cond_delta, pos_salient_pairs
                )

            if neg_salient_pairs:
                neg_aux = neg_aux - salient_blend(
                    -cond_delta, neg_salient_pairs
                )

            # Standard CFG in denoised space:
            #   d_uncond + scale * (d_cond - d_uncond)
            # Enhanced with strategy modifications on both sides:
            enhanced_uncond = d_uncond + neg_aux
            enhanced_delta = cond_delta + pos_aux
            result = enhanced_uncond + cond_scale * enhanced_delta

            if final_rescale > 0:
                result = apply_cfg_rescale(
                    result,
                    enhanced_uncond + enhanced_delta,
                    final_rescale,
                )

            return result

        # -------------------------------------------------------------- #

        patched.set_model_sampler_cfg_function(
            _cfg_function, disable_cfg1_optimization=True
        )
        patched.model_options["neutral_prompt_entries"] = all_entries
        patched.model_options["neutral_prompt_cfg_rescale"] = final_rescale
        patched.model_options["neutral_prompt_debug"] = final_debug

        return (patched, main_conditioning)
