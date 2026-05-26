"""
Emphasis Encode - Text encoding with emphasis support for Wan/T5 models
EXPERIMENTAL - may need adjustment based on model behavior
"""

import re
import torch


class EmphasisEncode:
    """
    EXPERIMENTAL: Text encoder with (word:weight) emphasis support.
    
    Attempts to bring A1111-style emphasis to Wan/T5 models.
    
    Syntax:
      - (word:1.2) = increase emphasis on word by 1.2x
      - (word:0.5) = decrease emphasis
      - (multiple words:1.3) = emphasize phrase
      - ((word)) = shorthand for (word:1.1)
      - nested: ((word:1.2)) = 1.2 * 1.1 = 1.32x
    
    Note: Token boundaries may not perfectly match word boundaries,
    so results are approximate. Works best with simple emphasis on
    individual words or short phrases.
    """
    
    CATEGORY = "ComfyCollectorNodes/Conditioning"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "parsed_info", "debug_output")
    FUNCTION = "encode_with_emphasis"

    def parse_emphasis(self, text):
        """
        Parse emphasis markers from text.
        Returns: (clean_text, list of {text, weight, start, end})
        """
        emphasis_regions = []
        
        # Handle ((word)) shorthand first - convert to (word:1.1)
        while "((" in text:
            text = re.sub(r'\(\(([^()]+)\)\)', r'(\1:1.1)', text)
        
        # Parse (content:weight) patterns
        pattern = r'\(([^()]+):([0-9.]+)\)'
        
        offset = 0
        for match in re.finditer(pattern, text):
            content = match.group(1)
            weight = float(match.group(2))
            
            start_in_original = match.start()
            original_len = len(match.group(0))
            clean_start = start_in_original - offset
            clean_end = clean_start + len(content)
            
            emphasis_regions.append({
                'text': content,
                'weight': weight,
                'start': clean_start,
                'end': clean_end,
            })
            
            offset += original_len - len(content)
        
        # Remove all emphasis markers from clean text
        clean_text = re.sub(r'\(([^()]+):([0-9.]+)\)', r'\1', text)
        
        return clean_text, emphasis_regions

    def count_real_tokens(self, token_list):
        """Count non-padding tokens (token_id > 1, since 0 and 1 are typically padding/special)"""
        count = 0
        for token_id, weight in token_list:
            if token_id > 1:
                count += 1
            else:
                break  # Stop at first padding token
        return count

    def apply_emphasis_to_tokens(self, tokens, emphasis_regions, clean_text, debug_lines):
        """
        Modify token weights based on emphasis regions.
        """
        # Get the encoder key (e.g., 'umt5xxl', 'clip_l', etc.)
        encoder_key = list(tokens.keys())[0]
        token_list = tokens[encoder_key][0]  # First (and usually only) batch
        
        # Count real (non-padding) tokens
        real_token_count = self.count_real_tokens(token_list)
        debug_lines.append(f"Real token count: {real_token_count}")
        
        if real_token_count == 0 or len(clean_text) == 0:
            return tokens
        
        # Estimate character-to-token mapping
        # This is still approximate but much better than before
        chars_per_token = len(clean_text) / real_token_count
        debug_lines.append(f"Chars per token estimate: {chars_per_token:.2f}")
        
        # Apply emphasis to each region
        for region in emphasis_regions:
            # Estimate which tokens this region covers
            start_token = int(region['start'] / chars_per_token)
            end_token = int(region['end'] / chars_per_token)
            
            # Clamp to valid range
            start_token = max(0, min(start_token, real_token_count - 1))
            end_token = max(start_token + 1, min(end_token + 1, real_token_count))
            
            debug_lines.append(f"Emphasis '{region['text']}' (chars {region['start']}-{region['end']}) -> tokens {start_token}-{end_token} @ {region['weight']:.2f}x")
            
            # Modify the weights in the token list
            for i in range(start_token, end_token):
                token_id, old_weight = token_list[i]
                new_weight = old_weight * region['weight']
                token_list[i] = (token_id, new_weight)
                debug_lines.append(f"  Token {i}: id={token_id}, weight {old_weight:.2f} -> {new_weight:.2f}")
        
        # Update the tokens dict
        tokens[encoder_key][0] = token_list
        return tokens

    def encode_with_emphasis(self, clip, text, debug):
        debug_lines = []
        
        # Parse emphasis from text
        clean_text, emphasis_regions = self.parse_emphasis(text)
        
        # Build info string
        if emphasis_regions:
            info_parts = [f"'{r['text']}' @ {r['weight']:.2f}x" for r in emphasis_regions]
            parsed_info = f"Emphasis: {', '.join(info_parts)}\nClean: {clean_text[:100]}..."
        else:
            parsed_info = "No emphasis markers found"
        
        # Tokenize
        tokens = clip.tokenize(clean_text)
        
        # Debug: show token structure
        debug_lines.append(f"=== TOKENIZER DEBUG ===")
        debug_lines.append(f"Clean text: {clean_text}")
        debug_lines.append(f"Token keys: {list(tokens.keys())}")
        
        encoder_key = list(tokens.keys())[0]
        token_list = tokens[encoder_key][0]
        
        # Show first few actual tokens
        debug_lines.append(f"First 10 tokens:")
        for i, (tid, tw) in enumerate(token_list[:10]):
            debug_lines.append(f"  [{i}]: id={tid}, weight={tw}")
        
        # Apply emphasis by modifying token weights
        if emphasis_regions:
            tokens = self.apply_emphasis_to_tokens(tokens, emphasis_regions, clean_text, debug_lines)
        
        # Encode with (possibly modified) tokens
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        debug_lines.append(f"=== CONDITIONING DEBUG ===")
        debug_lines.append(f"Conditioning shape: {cond.shape}")
        
        debug_output = "\n".join(debug_lines)
        
        if debug:
            print("\n" + debug_output + "\n")
        
        return ([[cond, {"pooled_output": pooled}]], parsed_info, debug_output)


class EmphasisEncodeAdvanced:
    """
    EXPERIMENTAL: Advanced emphasis encoding with normalization options.
    
    Same as EmphasisEncode but with post-emphasis normalization
    (like A1111's emphasis modes).
    """
    
    CATEGORY = "ComfyCollectorNodes/Conditioning"
    
    NORMALIZATION_METHODS = [
        "none",
        "mean_restore",
        "max_norm",
        "std_norm",
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "normalization": (cls.NORMALIZATION_METHODS, {"default": "mean_restore"}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "parsed_info", "debug_output")
    FUNCTION = "encode_with_emphasis"

    def parse_emphasis(self, text):
        emphasis_regions = []
        
        while "((" in text:
            text = re.sub(r'\(\(([^()]+)\)\)', r'(\1:1.1)', text)
        
        pattern = r'\(([^()]+):([0-9.]+)\)'
        offset = 0
        
        for match in re.finditer(pattern, text):
            content = match.group(1)
            weight = float(match.group(2))
            
            start_in_original = match.start()
            original_len = len(match.group(0))
            clean_start = start_in_original - offset
            clean_end = clean_start + len(content)
            
            emphasis_regions.append({
                'text': content,
                'weight': weight,
                'start': clean_start,
                'end': clean_end,
            })
            
            offset += original_len - len(content)
        
        clean_text = re.sub(r'\(([^()]+):([0-9.]+)\)', r'\1', text)
        return clean_text, emphasis_regions

    def count_real_tokens(self, token_list):
        count = 0
        for token_id, weight in token_list:
            if token_id > 1:
                count += 1
            else:
                break
        return count

    def apply_emphasis_to_tokens(self, tokens, emphasis_regions, clean_text, debug_lines):
        encoder_key = list(tokens.keys())[0]
        token_list = tokens[encoder_key][0]
        
        real_token_count = self.count_real_tokens(token_list)
        debug_lines.append(f"Real token count: {real_token_count}")
        
        if real_token_count == 0 or len(clean_text) == 0:
            return tokens
        
        chars_per_token = len(clean_text) / real_token_count
        debug_lines.append(f"Chars per token estimate: {chars_per_token:.2f}")
        
        for region in emphasis_regions:
            start_token = int(region['start'] / chars_per_token)
            end_token = int(region['end'] / chars_per_token)
            
            start_token = max(0, min(start_token, real_token_count - 1))
            end_token = max(start_token + 1, min(end_token + 1, real_token_count))
            
            debug_lines.append(f"Emphasis '{region['text']}' -> tokens {start_token}-{end_token} @ {region['weight']:.2f}x")
            
            for i in range(start_token, end_token):
                token_id, old_weight = token_list[i]
                new_weight = old_weight * region['weight']
                token_list[i] = (token_id, new_weight)
        
        tokens[encoder_key][0] = token_list
        return tokens

    def encode_with_emphasis(self, clip, text, normalization, debug):
        debug_lines = []
        
        clean_text, emphasis_regions = self.parse_emphasis(text)
        
        if emphasis_regions:
            info_parts = [f"'{r['text']}' @ {r['weight']:.2f}x" for r in emphasis_regions]
            parsed_info = f"Emphasis: {', '.join(info_parts)}\nNorm: {normalization}\nClean: {clean_text[:80]}..."
        else:
            parsed_info = "No emphasis markers found"
        
        tokens = clip.tokenize(clean_text)
        
        debug_lines.append(f"=== TOKENIZER DEBUG (Advanced) ===")
        debug_lines.append(f"Clean text: {clean_text}")
        
        encoder_key = list(tokens.keys())[0]
        token_list = tokens[encoder_key][0]
        
        debug_lines.append(f"First 10 tokens:")
        for i, (tid, tw) in enumerate(token_list[:10]):
            debug_lines.append(f"  [{i}]: id={tid}, weight={tw}")
        
        # Store original conditioning stats for normalization
        # We need to encode once without emphasis to get baseline
        if emphasis_regions and normalization != "none":
            baseline_cond, _ = clip.encode_from_tokens(clip.tokenize(clean_text), return_pooled=True)
            original_mean = baseline_cond.mean()
            original_max = baseline_cond.abs().max()
            original_std = baseline_cond.std()
        
        # Apply emphasis
        if emphasis_regions:
            tokens = self.apply_emphasis_to_tokens(tokens, emphasis_regions, clean_text, debug_lines)
        
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        # Apply normalization
        if emphasis_regions and normalization != "none":
            if normalization == "mean_restore":
                new_mean = cond.mean()
                if new_mean != 0:
                    cond = cond * (original_mean / new_mean)
                    debug_lines.append(f"Mean restore: {new_mean:.4f} -> {original_mean:.4f}")
                    
            elif normalization == "max_norm":
                new_max = cond.abs().max()
                if new_max > 0:
                    cond = cond * (original_max / new_max)
                    debug_lines.append(f"Max norm: {new_max:.4f} -> {original_max:.4f}")
                    
            elif normalization == "std_norm":
                new_std = cond.std()
                if new_std > 0:
                    cond = cond * (original_std / new_std)
                    debug_lines.append(f"Std norm: {new_std:.4f} -> {original_std:.4f}")
        
        debug_lines.append(f"=== CONDITIONING DEBUG ===")
        debug_lines.append(f"Conditioning shape: {cond.shape}")
        debug_lines.append(f"Applied normalization: {normalization}")
        
        debug_output = "\n".join(debug_lines)
        
        if debug:
            print("\n" + debug_output + "\n")
        
        return ([[cond, {"pooled_output": pooled}]], parsed_info, debug_output)
