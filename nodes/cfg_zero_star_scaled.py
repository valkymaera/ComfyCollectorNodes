"""
CFG-Zero* with configurable strength and scaled init.
Based on https://github.com/WeichenFan/CFG-Zero-star

Vanilla CFG:        uncond + scale * (cond - uncond)
CFG-Zero* (optimized scale): uncond * α + scale * (cond - uncond * α)
    where α = dot(cond, uncond) / ||uncond||²

Strength lerps between vanilla CFG (0.0) and CFG-Zero* (1.0). Unclamped.
Init scale attenuates the prediction during early steps (0.0 = original zero-init).
"""

import torch


class CFGZeroStarScaled:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "strength": ("FLOAT", {
                "default": 1.0,
                "step": 0.001,
                "tooltip": "Lerp between vanilla CFG (0.0) and CFG-Zero* (1.0). Unclamped for experimentation."
            }),
            "use_scaled_init": ("BOOLEAN", {"default": True}),
            "init_scale": ("FLOAT", {
                "default": 0.0,
                "step": 0.001,
                "tooltip": "Multiplier for init steps. 0.0 = original zero-init, 1.0 = no init effect."
            }),
            "init_steps": ("INT", {
                "default": 0,
                "min": 0,
                "tooltip": "Init applies from step 0 through this step index (inclusive)."
            }),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    DESCRIPTION = "CFG-Zero* with configurable strength and scaled init. Based on https://github.com/WeichenFan/CFG-Zero-star"
    CATEGORY = "CCN/experimental"
    EXPERIMENTAL = True

    def patch(self, model, strength, use_scaled_init, init_scale, init_steps):
        def cfg_zerostar(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]

            # Vanilla CFG
            normal_pred = uncond + cond_scale * (cond - uncond)

            if strength == 0.0:
                noise_pred = normal_pred
            else:
                batch_size = cond.shape[0]

                positive_flat = cond.view(batch_size, -1)
                negative_flat = uncond.view(batch_size, -1)

                dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
                squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
                alpha = dot_product / squared_norm
                alpha = alpha.view(batch_size, *([1] * (len(cond.shape) - 1)))

                zerostar_pred = uncond * alpha + cond_scale * (cond - uncond * alpha)

                noise_pred = normal_pred + strength * (zerostar_pred - normal_pred)

            if use_scaled_init:
                timestep = args["timestep"]
                sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
                matched_step_index = (sigmas == timestep[0]).nonzero()
                if len(matched_step_index) > 0:
                    current_step_index = matched_step_index.item()
                else:
                    for i in range(len(sigmas) - 1):
                        if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                            current_step_index = i
                            break
                    else:
                        current_step_index = 0

                if current_step_index <= init_steps:
                    noise_pred = noise_pred * init_scale

            return noise_pred

        m = model.clone()
        m.set_model_sampler_cfg_function(cfg_zerostar)
        return (m,)
