import json
import torch
import comfy.samplers


def hermite(keys, t):
    """Evaluate cubic Hermite spline at position t (0..1)."""
    n = len(keys)
    if n == 0:
        return 0.0
    if n == 1:
        return keys[0]["y"]
    if t <= keys[0]["x"]:
        return keys[0]["y"]
    if t >= keys[-1]["x"]:
        return keys[-1]["y"]

    idx = 0
    for i in range(n - 1):
        if keys[i]["x"] <= t <= keys[i + 1]["x"]:
            idx = i
            break

    k0, k1 = keys[idx], keys[idx + 1]
    dt = k1["x"] - k0["x"]
    if dt < 1e-10:
        return k0["y"]

    lt = (t - k0["x"]) / dt
    lt2 = lt * lt
    lt3 = lt2 * lt

    h00 = 2.0 * lt3 - 3.0 * lt2 + 1.0
    h10 = lt3 - 2.0 * lt2 + lt
    h01 = -2.0 * lt3 + 3.0 * lt2
    h11 = lt3 - lt2

    return h00 * k0["y"] + h10 * k0["out"] * dt + h01 * k1["y"] + h11 * k1["in"] * dt


class CurveCFG(comfy.samplers.CFGGuider):
    """CFGGuider that dynamically adjusts CFG scale over the course of
    inference based on a user-defined Hermite curve.
    
    The curve maps denoising progress (0..1) to a blend factor between
    min_cfg and max_cfg:  cfg = lerp(min_cfg, max_cfg, curve(progress))
    """

    def set_curve_params(self, curve_keys, min_cfg, max_cfg, mode, sigmas, sigma_decay):
        self.curve_keys = curve_keys
        self.min_cfg = float(min_cfg)
        self.max_cfg = float(max_cfg)
        self.mode = mode
        self.sigma_decay = sigma_decay

        # Cache the schedule as a list of scheduled sigmas (one per step
        # boundary).  Step index = position in this list.
        self.scheduled_sigmas = [float(s) for s in sigmas.tolist()]
        self.sigma_max = self.scheduled_sigmas[0]
        self.sigma_min = self.scheduled_sigmas[-1]
        self.total_steps = max(len(self.scheduled_sigmas) - 1, 1)

    def _current_progress(self, timestep):
        """Compute normalized progress 0..1 from the current sigma.
        
        Robust to multi-evaluation samplers (Heun, DPM++ 2M etc.) because
        progress is derived directly from sigma rather than a call counter.
        """
        if isinstance(timestep, torch.Tensor):
            sigma = float(timestep.flatten()[0].item())
        else:
            sigma = float(timestep)

        if self.mode == "step":
            # Find which step this sigma most closely corresponds to by
            # locating the nearest scheduled sigma and using its index.
            # Intermediate sigmas (from 2nd-order samplers) snap to the
            # surrounding step.
            best_i = 0
            best_d = abs(self.scheduled_sigmas[0] - sigma)
            for i, s in enumerate(self.scheduled_sigmas):
                d = abs(s - sigma)
                if d < best_d:
                    best_d = d
                    best_i = i
            t = best_i / self.total_steps
        else:
            rng = self.sigma_max - self.sigma_min
            t = (self.sigma_max - sigma) / rng if rng > 1e-10 else 0.0

        return max(0.0, min(1.0, t)), sigma

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        t, sigma = self._current_progress(timestep)

        # Evaluate curve and lerp between min/max CFG
        blend = hermite(self.curve_keys, t)
        cfg = self.min_cfg + (self.max_cfg - self.min_cfg) * blend

        # Optional sigma-proportional guidance decay.  When enabled,
        # the effective CFG is attenuated toward 1.0 as sigma decreases,
        # reducing guidance strength in the low-noise phase of sampling.
        if self.sigma_decay and self.sigma_max > 1e-10:
            scale = sigma / self.sigma_max
            cfg = 1.0 + (cfg - 1.0) * scale

        # Set cfg so the parent CFGGuider's sampling path uses it
        self.cfg = cfg
        return super().predict_noise(x, timestep, model_options, seed)


DEFAULT_CURVE = '[{"x":0,"y":1,"in":0,"out":-1,"mirrored":true},{"x":1,"y":0,"in":-1,"out":0,"mirrored":true}]'


class CurveCFGGuider:
    """Node that creates a CFG guider with a visually editable curve
    controlling how CFG scale changes over the course of inference.
    
    The curve maps denoising progress to a 0..1 blend factor:
      - curve output 0 → min_cfg
      - curve output 1 → max_cfg
    
    Mode controls how progress is measured:
      - step:  current_step / total_steps (linear in step count)
      - sigma: normalized position in the sigma schedule
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sigmas": ("SIGMAS",),
                "min_cfg": ("FLOAT", {"default": 1.0, "step": 0.001}),
                "max_cfg": ("FLOAT", {"default": 7.0, "step": 0.001}),
                "mode": (["step", "sigma"],),
                "sigma_decay": ("BOOLEAN", {"default": False}),
                "curve_data": ("STRING", {"default": DEFAULT_CURVE, "multiline": False}),
            },
            "optional": {
                "curve": ("CCN_CURVE",),
            }
        }

    RETURN_TYPES = ("GUIDER", "SIGMAS")
    RETURN_NAMES = ("guider", "sigmas")
    FUNCTION = "get_guider"
    CATEGORY = "CCN"

    def get_guider(self, model, positive, negative, sigmas, min_cfg, max_cfg, mode, sigma_decay, curve_data, curve=None):
        if curve is not None:
            keys = curve
        else:
            try:
                keys = json.loads(curve_data)
            except (json.JSONDecodeError, TypeError):
                keys = json.loads(DEFAULT_CURVE)

        keys = sorted(keys, key=lambda k: k["x"])

        guider = CurveCFG(model)
        guider.set_conds(positive, negative)
        guider.set_curve_params(keys, min_cfg, max_cfg, mode, sigmas, sigma_decay)

        return (guider, sigmas)
