"""
MoE Sigma Split -- splits a flow-matching sigma schedule at the Wan 2.2
expert boundary.

Wan 2.2 A14B routes denoising by diffusion timestep: the high-noise expert
handles t >= boundary * 1000 and the low-noise expert handles the remainder
(boundary 0.875 for T2V, 0.900 for I2V, over 1000 training timesteps).
ComfyUI samples Wan as flow matching with sigma in [0, 1] and
t = sigma * 1000, so the correct expert hand-off is the first step whose
sigma drops below the boundary. That index depends on scheduler, step
count, shift, and denoise -- it is not a fixed step ratio.

Standard wiring (two SamplerCustomAdvanced nodes):
    sigmas_high -> HIGH-noise model sampler, add_noise enabled
    sigmas_low  -> LOW-noise model sampler, add_noise disabled,
                   latent chained from the high phase

Degenerate schedules are handled explicitly. An empty side receives an
inert single-value [0.0] stub: through ComfyUI's CONST (flow) noise
scaling, a zero-step pass over [0.0] returns the latent unchanged whether
add_noise is enabled or not. In the all-low case (high_steps == 0), enable
add_noise on the LOW phase sampler instead -- noise normally enters through
the high phase, and the inert stub cannot carry it.
"""

import torch

_BOUNDARY_PRESETS = {
    "t2v (0.875)": 0.875,
    "i2v (0.900)": 0.900,
}
_CUSTOM_LABEL = "custom"

# Flow-matching schedules live in [0, 1]. Anything materially beyond that
# is an EDM/eps-style schedule (SD family) where a [0, 1] timestep boundary
# is meaningless, so we refuse rather than mis-split silently.
_SIGMA_MAX_TOLERANCE = 1.0 + 1e-4


class MoESigmaSplit:
    """Split a SIGMAS schedule at the Wan 2.2 MoE expert boundary."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS", {
                    "tooltip": "Flow-matching sigma schedule, e.g. from BasicScheduler "
                               "reading a shift-patched Wan model.",
                }),
                "boundary": (list(_BOUNDARY_PRESETS) + [_CUSTOM_LABEL], {
                    "default": "i2v (0.900)",
                    "tooltip": "Trained expert boundary in timestep space (t = sigma * 1000). "
                               "Steps evaluated at sigma >= boundary run on the high-noise expert.",
                }),
                "custom_boundary": ("FLOAT", {
                    "default": 0.875,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Boundary used only when 'boundary' is set to custom.",
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS", "SIGMAS", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("sigmas_high", "sigmas_low", "high_steps", "low_steps", "switch_sigma", "info")
    FUNCTION = "split"
    CATEGORY = "CCN"
    DESCRIPTION = (
        "Splits a flow-matching sigma schedule at the Wan 2.2 MoE expert boundary -- "
        "the first step whose sigma drops below the boundary -- emitting per-expert "
        "schedules that share the hand-off sigma."
    )

    def split(self, sigmas, boundary, custom_boundary):
        sigmas = self._validate_sigmas(sigmas)
        bound, bound_label, warnings = self._resolve_boundary(boundary, custom_boundary)

        n_values = int(sigmas.numel())
        total_steps = n_values - 1

        below = sigmas < bound
        if not bool(torch.all(sigmas[1:] <= sigmas[:-1])):
            warnings.append(
                "schedule is not monotonically decreasing (restart/custom schedule?); "
                "splitting at the first boundary crossing only"
            )
        crossings = int((below[1:] != below[:-1]).sum())
        if crossings > 1:
            warnings.append(
                f"schedule crosses the boundary {crossings} times; only the first crossing is used"
            )

        first_below = torch.nonzero(below, as_tuple=False)
        stub = sigmas.new_zeros(1)  # inert: exact identity through CONST noise scaling

        if first_below.numel() == 0:
            # Every sigma >= boundary: all steps belong to the high expert.
            sigmas_high = sigmas.clone()
            sigmas_low = stub
            high_steps, low_steps = total_steps, 0
            switch_sigma = 0.0
            handoff_line = "hand-off: none (boundary never crossed; low phase is an inert stub)"
            warnings.append(
                "low phase is empty -- its sampler performs zero steps and passes the latent "
                f"through; the final latent is only denoised to sigma {float(sigmas[-1]):.4f}"
            )
        else:
            k = int(first_below[0])
            if k == 0:
                # First sigma is already below the boundary: all steps belong to the low expert.
                sigmas_high = stub
                sigmas_low = sigmas.clone()
                high_steps, low_steps = 0, total_steps
                switch_sigma = float(sigmas[0])
                handoff_line = (
                    f"hand-off: step 0, sigma {switch_sigma:.4f} (t={switch_sigma * 1000.0:.1f}) -- "
                    "low expert runs the entire schedule"
                )
                warnings.append(
                    "high phase is empty -- ENABLE add_noise on the LOW phase sampler "
                    "(noise normally enters through the high phase)"
                )
            else:
                sigmas_high = sigmas[: k + 1].clone()
                sigmas_low = sigmas[k:].clone()
                high_steps, low_steps = k, total_steps - k
                switch_sigma = float(sigmas[k])
                handoff_line = (
                    f"hand-off: step {k}, sigma {switch_sigma:.4f} (t={switch_sigma * 1000.0:.1f})"
                )

        info_lines = [
            f"boundary: {bound_label} -> sigma {bound:.4f} (t={bound * 1000.0:.1f})",
            f"schedule: {total_steps} steps, sigma {float(sigmas[0]):.4f} -> {float(sigmas[-1]):.4f}",
            f"split: {high_steps} high / {low_steps} low",
            handoff_line,
        ]
        info_lines.extend(f"warning: {w}" for w in warnings)
        info = "\n".join(info_lines)

        return (sigmas_high, sigmas_low, int(high_steps), int(low_steps), float(switch_sigma), info)

    @staticmethod
    def _validate_sigmas(sigmas):
        if not isinstance(sigmas, torch.Tensor):
            raise TypeError(
                f"MoESigmaSplit: expected a SIGMAS tensor, got {type(sigmas).__name__}"
            )
        sigmas = sigmas.detach()
        if sigmas.dim() != 1:
            raise ValueError(
                f"MoESigmaSplit: expected a 1-D sigma schedule, got shape {tuple(sigmas.shape)}"
            )
        if sigmas.numel() < 2:
            raise ValueError(
                "MoESigmaSplit: schedule must contain at least two sigma values "
                f"(one sampling step); got {sigmas.numel()}"
            )
        sigma_min = float(sigmas.min())
        sigma_max = float(sigmas.max())
        if sigma_max > _SIGMA_MAX_TOLERANCE or sigma_min < 0.0:
            raise ValueError(
                f"MoESigmaSplit: sigma range [{sigma_min:.4f}, {sigma_max:.4f}] is outside [0, 1]. "
                "This node expects a flow-matching schedule (Wan-style, t = sigma * 1000); "
                "EDM/eps schedules from SD-family models cannot be split by a timestep boundary."
            )
        return sigmas

    @staticmethod
    def _resolve_boundary(boundary, custom_boundary):
        warnings = []
        if boundary == _CUSTOM_LABEL:
            bound = float(custom_boundary)
            label = f"custom ({bound:.3f})"
            if not (0.0 < bound < 1.0):
                warnings.append(
                    f"custom boundary {bound:.3f} is outside (0, 1); "
                    "the split is degenerate by construction"
                )
        elif boundary in _BOUNDARY_PRESETS:
            bound = _BOUNDARY_PRESETS[boundary]
            label = boundary
        else:
            raise ValueError(
                f"MoESigmaSplit: unknown boundary preset {boundary!r}; "
                f"expected one of {list(_BOUNDARY_PRESETS) + [_CUSTOM_LABEL]}"
            )
        return bound, label, warnings
