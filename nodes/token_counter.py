"""
Token Counter - Count tokens in text for various encoders
"""


class TokenCounter:
    """
    Count tokens in a text string.
    
    Useful for checking prompt length against token budgets.
    
    Note: This uses a character-based estimate when CLIP is not provided.
    For accurate counts, connect a CLIP model.
    
    Common token limits:
      - SD 1.5 / SDXL CLIP: 77 tokens
      - T5 (Wan, Flux): 512 tokens
    """
    
    CATEGORY = "ComfyCollectorNodes/Utils"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("token_count", "info")
    FUNCTION = "count_tokens"

    def count_tokens(self, text, clip=None):
        if not text.strip():
            info = "Empty text: 0 tokens"
            print(f"[ComfyCollectorNodes] TokenCounter: {info}")
            return (0, info)
        
        if clip is not None:
            # Use actual tokenizer
            try:
                tokens = clip.tokenize(text)
                
                # tokens is a dict like {'umt5xxl': [[(token_id, weight), ...]]}
                # or {'clip_l': [...], 'clip_g': [...]} for SDXL
                encoder_name = list(tokens.keys())[0]
                token_list = tokens[encoder_name][0]
                
                # Count non-padding tokens (id > 1 typically)
                # Stop at EOS (id == 1) or padding (id == 0)
                real_count = 0
                for token_id, weight in token_list:
                    if token_id > 1:
                        real_count += 1
                    else:
                        break
                
                total_capacity = len(token_list)
                remaining = total_capacity - real_count
                percentage = (real_count / total_capacity) * 100
                
                if real_count >= total_capacity - 1:
                    warning = " ⚠️ AT LIMIT - text may be truncated!"
                elif percentage > 80:
                    warning = " (nearing limit)"
                else:
                    warning = ""
                
                info = f"{real_count}/{total_capacity} tokens ({percentage:.1f}%){warning}"
                print(f"[ComfyCollectorNodes] TokenCounter [{encoder_name}]: {info}")
                
                return (real_count, info)
                
            except Exception as e:
                info = f"Tokenizer error: {e}, falling back to estimate"
                print(f"[ComfyCollectorNodes] TokenCounter: {info}")
        
        # Fallback: character-based estimate
        # Rough estimate: ~4 characters per token for English
        char_count = len(text)
        estimated_tokens = max(1, char_count // 4)
        
        info = f"~{estimated_tokens} tokens (estimate from {char_count} chars, connect CLIP for accuracy)"
        print(f"[ComfyCollectorNodes] TokenCounter: {info}")
        
        return (estimated_tokens, info)


class ConditioningTokenCount:
    """
    Get token count from conditioning tensor.
    
    This reads the sequence length from the conditioning,
    which corresponds to the number of token positions.
    """
    
    CATEGORY = "ComfyCollectorNodes/Utils"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("sequence_length", "info")
    FUNCTION = "count"

    def count(self, conditioning):
        if not conditioning or len(conditioning) == 0:
            info = "Empty conditioning"
            print(f"[ComfyCollectorNodes] ConditioningTokenCount: {info}")
            return (0, info)
        
        cond = conditioning[0][0]
        shape = cond.shape
        
        # Shape is typically [batch, seq_len, hidden_dim]
        if len(shape) >= 2:
            seq_len = shape[1]
            hidden_dim = shape[2] if len(shape) > 2 else "?"
            info = f"Sequence length: {seq_len}, hidden dim: {hidden_dim}, shape: {list(shape)}"
        else:
            seq_len = shape[0]
            info = f"Sequence length: {seq_len}, shape: {list(shape)}"
        
        print(f"[ComfyCollectorNodes] ConditioningTokenCount: {info}")
        
        return (seq_len, info)
