"""
Token Remap — Remap token embeddings at the conditioning level.
Fix ambiguous words (e.g. "ship" → "starship") by blending embeddings
toward your intended meaning with controllable blend.


"""

import re
import torch
import logging

logger = logging.getLogger("CCN.TokenRemap")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def parse_remappings(text):
    """
    Parse a multiline remapping string.
    Supports formats:
        ship -> starship
        ship => starship
        ship : starship
        ship, starship
    Lines starting with # are comments. Blank lines are skipped.
    Returns list of (source, target) tuples.
    """
    pairs = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for pattern in [r'\s*->\s*', r'\s*=>\s*', r'\s*:\s*', r'\s*,\s*']:
            parts = re.split(pattern, line, maxsplit=1)
            if len(parts) == 2:
                source, target = parts[0].strip(), parts[1].strip()
                if source and target:
                    pairs.append((source, target))
                break
        else:
            logger.warning(f"TokenRemap: Could not parse remapping line: '{line}'")
    return pairs


def apply_text_remappings(text, pairs):
    """Apply word-boundary text replacements for each (source, target) pair."""
    for source, target in pairs:
        text = re.sub(
            rf'\b{re.escape(source)}\b',
            target,
            text,
            flags=re.IGNORECASE
        )
    return text


# ---------------------------------------------------------------------------
#  TokenRemap — replaces CLIPTextEncode with embedding-level blending
# ---------------------------------------------------------------------------

class TokenRemap:
    """
    Encodes a text prompt with token-level embedding remapping.
    Encodes the prompt twice — once as-is, once with word replacements —
    then blends the conditioning at affected token positions.

    When token counts match between original and remapped prompts,
    blending is applied only at the positions that changed (precise mode).
    When counts differ, the full conditioning tensors are lerped (global mode).

    Use this in place of CLIPTextEncode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "remappings": ("STRING", {
                    "multiline": True,
                    "default": "# source -> target\n# ship -> starship\n# craft -> spacecraft",
                    "dynamicPrompts": False,
                }),
                "blend": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.05,
                    "tooltip": "0.0 = original prompt, 1.0 = fully remapped. "
                               "Values beyond 0-1 extrapolate in embedding space."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "text_out")
    FUNCTION = "encode"
    CATEGORY = "CCN/conditioning"
    DESCRIPTION = ("Encode a prompt with token-level embedding remapping. "
                   "Remap words like 'ship' to 'starship' at the embedding level "
                   "with controllable blending.")

    def encode(self, clip, text, remappings, blend):
        pairs = parse_remappings(remappings)

        # If no valid remappings or blend is 0, just do a normal encode
        if not pairs or blend == 0.0:
            tokens = clip.tokenize(text)
            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = output.pop("cond")
            return ([[cond, output]], text)

        # Build the remapped text
        remapped_text = apply_text_remappings(text, pairs)

        # If the text didn't actually change, just do a normal encode
        if remapped_text == text:
            tokens = clip.tokenize(text)
            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = output.pop("cond")
            return ([[cond, output]], text)

        # Encode both versions
        orig_tokens = clip.tokenize(text)
        remap_tokens = clip.tokenize(remapped_text)

        orig_output = clip.encode_from_tokens(
            orig_tokens, return_pooled=True, return_dict=True
        )
        remap_output = clip.encode_from_tokens(
            remap_tokens, return_pooled=True, return_dict=True
        )

        orig_cond = orig_output.pop("cond")
        remap_cond = remap_output.pop("cond")

        # Determine blending strategy based on tensor shapes
        if orig_cond.shape == remap_cond.shape:
            blended = orig_cond.clone()
            changed_mask = self._build_change_mask(orig_tokens, remap_tokens, orig_cond.shape)

            if changed_mask is not None:
                logger.info(
                    f"TokenRemap: Precise blend at "
                    f"{changed_mask.sum().item():.0f}/{changed_mask.numel()} positions"
                )
                blended = torch.where(
                    changed_mask.unsqueeze(-1).expand_as(blended),
                    torch.lerp(orig_cond, remap_cond, blend),
                    orig_cond
                )
            else:
                logger.info("TokenRemap: Global blend (could not determine changed positions)")
                blended = torch.lerp(orig_cond, remap_cond, blend)
        else:
            logger.warning(
                f"TokenRemap: Conditioning shape mismatch "
                f"({orig_cond.shape} vs {remap_cond.shape}). "
                f"Using global blend with shape matching."
            )
            min_seq = min(orig_cond.shape[1], remap_cond.shape[1])
            blended = torch.lerp(
                orig_cond[:, :min_seq, :],
                remap_cond[:, :min_seq, :],
                blend
            )

        return ([[blended, orig_output]], text)

    def _build_change_mask(self, orig_tokens, remap_tokens, cond_shape):
        """
        Compare two token dicts and return a boolean mask of shape
        (batch, seq_len) indicating which positions differ.
        Returns None if comparison isn't possible.
        """
        try:
            key = next(iter(orig_tokens))
            if key not in remap_tokens:
                return None

            orig_ids = []
            for chunk in orig_tokens[key]:
                for token_id, _w in chunk:
                    orig_ids.append(token_id)

            remap_ids = []
            for chunk in remap_tokens[key]:
                for token_id, _w in chunk:
                    remap_ids.append(token_id)

            if len(orig_ids) != len(remap_ids):
                return None

            seq_len = cond_shape[1]
            mask = torch.zeros(1, seq_len, dtype=torch.bool, device='cpu')

            for i in range(min(len(orig_ids), seq_len)):
                if orig_ids[i] != remap_ids[i]:
                    mask[0, i] = True

            return mask.to(device=torch.device('cpu'))

        except (StopIteration, IndexError, KeyError):
            return None


# ---------------------------------------------------------------------------
#  ClipRemap — patches CLIP object for automatic remapping
# ---------------------------------------------------------------------------

class ClipRemap:
    """
    Wraps a CLIP model to automatically remap words during tokenization.
    Place before any CLIPTextEncode node. All prompts encoded with the
    output CLIP will have the remappings applied automatically.

    Hard remap (no blending). Use TokenRemap if you need blend control.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "remappings": ("STRING", {
                    "multiline": True,
                    "default": "# source -> target\n# ship -> starship",
                    "dynamicPrompts": False,
                }),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "remap"
    CATEGORY = "CCN/conditioning"
    DESCRIPTION = ("Modify a CLIP model to automatically remap words during "
                   "tokenization. Hard remap — no blending. Place before "
                   "CLIPTextEncode nodes.")

    def remap(self, clip, remappings, enabled):
        if not enabled:
            return (clip,)

        pairs = parse_remappings(remappings)
        if not pairs:
            return (clip,)

        clip_clone = clip.clone()
        _original_tokenize = clip_clone.tokenize
        _pairs = pairs

        def patched_tokenize(text, *args, **kwargs):
            remapped = apply_text_remappings(text, _pairs)
            if remapped != text:
                logger.info(f"ClipRemap: '{text}' → '{remapped}'")
            return _original_tokenize(remapped, *args, **kwargs)

        clip_clone.tokenize = patched_tokenize
        return (clip_clone,)


# ---------------------------------------------------------------------------
#  TokenInspector — debugging utility
# ---------------------------------------------------------------------------

class TokenInspector:
    """
    Shows how a CLIP model tokenizes a given prompt.
    Outputs a human-readable string showing each token and its ID.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inspect"
    CATEGORY = "CCN/conditioning"
    OUTPUT_NODE = True
    DESCRIPTION = "Visualize how a prompt is tokenized by the CLIP model."

    def inspect(self, clip, text):
        tokens = clip.tokenize(text)
        lines = [f"Prompt: \"{text}\"", ""]

        for encoder_key, chunks in tokens.items():
            lines.append(f"═══ Encoder: {encoder_key} ═══")
            for chunk_idx, chunk in enumerate(chunks):
                lines.append(f"  Chunk {chunk_idx} ({len(chunk)} tokens):")
                for pos, (token_id, weight) in enumerate(chunk):
                    weight_str = f"  (weight: {weight:.2f})" if weight != 1.0 else ""
                    lines.append(f"    [{pos:3d}] ID {token_id:6d}{weight_str}")
            lines.append("")

        for encoder_key, chunks in tokens.items():
            all_ids = []
            for chunk in chunks:
                for token_id, _w in chunk:
                    all_ids.append(token_id)
            if len(all_ids) >= 2:
                bos = all_ids[0]
                eos = all_ids[-1]
                content = [t for t in all_ids if t != bos and t != eos]
                lines.append(f"Content tokens ({encoder_key}): {len(content)}")

        result = "\n".join(lines)
        logger.info(f"\n{result}")
        return (result,)
