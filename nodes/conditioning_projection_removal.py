"""
Conditioning Projection Removal — Suppress concepts by projecting
them out of conditioning embeddings.

Takes positive and negative conditioning and removes the directional
component of the negative from the positive.  This is a lightweight,
pre-attention approach to negative guidance — rather than modifying
the sampling process (like NAG), it modifies the conditioning itself.

The negative conditioning is mean-pooled across its sequence to
extract a single concept direction vector, then that direction is
projected out of every token in the positive conditioning.

How it works:

  1. Mean-pool the negative embedding across its token sequence to
     get a single direction vector representing the "concept axis."
  2. For each token in the positive conditioning, compute its
     component along that negative direction.
  3. Subtract that component (scaled by strength).

At scale=1.0 you remove exactly the projection (the positive
conditioning loses only the part that points in the same direction
as the negative).  Above 1.0 you overcorrect, actively pushing
away from the negative direction.  Below 1.0 is partial removal.

Limitations:
  - Mean-pooling collapses multi-concept negatives into a single
    averaged direction, which dilutes each concept.  For complex
    negatives, use multiple nodes in sequence with focused negatives.
  - Operates in embedding space before attention, so it cannot
    leverage the model's own semantic interpretation the way
    post-attention methods (NAG) can.
"""

import torch
import logging

logger = logging.getLogger("CCN.ConditioningProjectionRemoval")


class ConditioningProjectionRemoval:
    """
    Remove the directional influence of a negative conditioning from
    a positive conditioning via vector projection.

    Connect any two conditionings — the negative's concept direction
    is projected out of the positive.  Useful as a lightweight
    alternative to negative prompts for flow-based models (Flux, SD3)
    that don't support CFG-based negatives.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": (
                        "Projection removal strength.  "
                        "1.0 = remove exactly the negative's directional "
                        "component.  >1.0 = overcorrect, pushing away "
                        "from the negative.  0 = no effect."
                    ),
                }),
                "pooling": (["mean", "max", "weighted_norm"], {
                    "default": "mean",
                    "tooltip": (
                        "How to collapse the negative's token sequence "
                        "into a single concept direction.  "
                        "mean = average all tokens (good general default).  "
                        "max = take the token with the largest norm "
                        "(picks the most 'opinionated' token).  "
                        "weighted_norm = weight each token by its norm "
                        "before averaging (emphasizes stronger tokens)."
                    ),
                }),
            },
            "optional": {
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print diagnostics to console",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "CCN/conditioning"
    DESCRIPTION = (
        "Remove the directional influence of a negative conditioning "
        "from a positive via vector projection.  A lightweight "
        "pre-attention approach to concept suppression for models "
        "that lack native negative prompt support (Flux, SD3)."
    )

    def _pool_direction(self, tensor, method, debug=False):
        """
        Collapse a conditioning tensor (batch, seq_len, dim) into a
        single normalized direction vector (batch, 1, dim).
        """
        if method == "max":
            # Pick the token with the largest L2 norm
            norms = tensor.norm(dim=-1)  # (batch, seq_len)
            max_idx = norms.argmax(dim=-1, keepdim=True)  # (batch, 1)
            max_idx = max_idx.unsqueeze(-1).expand(-1, -1, tensor.shape[-1])
            direction = torch.gather(tensor, 1, max_idx)  # (batch, 1, dim)

            if debug:
                print(
                    f"[ProjectionRemoval] Pooling: max | "
                    f"selected token index: {max_idx[0, 0, 0].item():.0f} | "
                    f"norm: {norms.max().item():.4f}"
                )

        elif method == "weighted_norm":
            # Weight each token by its L2 norm before averaging
            norms = tensor.norm(dim=-1, keepdim=True)  # (batch, seq_len, 1)
            total = norms.sum(dim=1, keepdim=True).clamp(min=1e-8)
            weights = norms / total
            direction = (tensor * weights).sum(dim=1, keepdim=True)

            if debug:
                w_squeezed = weights.squeeze(-1)
                print(
                    f"[ProjectionRemoval] Pooling: weighted_norm | "
                    f"weight range: [{w_squeezed.min().item():.4f}, "
                    f"{w_squeezed.max().item():.4f}]"
                )

        else:
            # Mean pool across sequence
            direction = tensor.mean(dim=1, keepdim=True)  # (batch, 1, dim)

            if debug:
                print(
                    f"[ProjectionRemoval] Pooling: mean | "
                    f"seq_len: {tensor.shape[1]} | "
                    f"direction norm: {direction.norm().item():.4f}"
                )

        # Normalize to unit direction
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        return direction

    def apply(self, positive, negative, scale, pooling="mean", debug=False):
        if scale == 0.0:
            return (positive,)

        out = []

        for i, (pos_tensor, pos_dict) in enumerate(positive):
            # Pair with corresponding negative, or reuse the first
            neg_tensor = negative[i][0] if i < len(negative) else negative[0][0]
            neg_tensor = neg_tensor.to(
                device=pos_tensor.device, dtype=pos_tensor.dtype
            )

            # Pool the negative into a single concept direction
            neg_dir = self._pool_direction(neg_tensor, pooling, debug=debug)

            # Compute projection of each positive token onto neg direction
            # pos_tensor: (batch, seq_len, dim)
            # neg_dir:    (batch, 1, dim)
            dot = (pos_tensor * neg_dir).sum(dim=-1, keepdim=True)  # (batch, seq_len, 1)
            projection = dot * neg_dir  # (batch, seq_len, dim)

            # Remove the projection
            modified = pos_tensor - scale * projection

            if debug:
                proj_norms = projection.norm(dim=-1)  # (batch, seq_len)
                pos_norms = pos_tensor.norm(dim=-1)
                mod_norms = modified.norm(dim=-1)

                # How much of each token pointed in the negative direction
                alignment = dot.squeeze(-1)  # (batch, seq_len)
                aligned_mask = alignment > 0
                aligned_count = aligned_mask.sum().item()
                total_count = alignment.numel()

                print(
                    f"[ProjectionRemoval] Entry {i} | "
                    f"scale: {scale:.2f} | "
                    f"aligned tokens: {aligned_count}/{total_count} | "
                    f"proj norm — "
                    f"mean: {proj_norms.mean().item():.4f}, "
                    f"max: {proj_norms.max().item():.4f} | "
                    f"pos norm — "
                    f"mean: {pos_norms.mean().item():.4f} → "
                    f"{mod_norms.mean().item():.4f}"
                )

            out.append((modified, pos_dict.copy()))

        return (out,)
