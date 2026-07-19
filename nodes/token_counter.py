"""
Token Counter - Count tokens in text for various encoders
"""


def _chunk_ids(chunk):
    # Entries may be (id, weight) or (id, weight, word_id) depending on
    # tokenizer options, so index rather than unpack.
    return [entry[0] for entry in chunk]


def _resolve_sub_tokenizer(clip, encoder_key):
    # ComfyUI wrapper tokenizers store per-encoder tokenizers under either
    # the raw key (T5 family) or a "clip_" prefixed key (SD/SDXL family).
    tok = getattr(clip, "tokenizer", None)
    if tok is None:
        return None
    for attr in (encoder_key, "clip_" + encoder_key):
        sub = getattr(tok, attr, None)
        if sub is not None:
            return sub
    return tok


def _resolve_special_tokens(sub_tokenizer):
    """
    Resolve BOS/EOS/pad ids from a tokenizer object.
    Returns a dict with any of "start"/"end"/"pad", or None if nothing
    usable was found (caller falls back to structural detection).
    """
    if sub_tokenizer is None:
        return None

    # Attribute name pairs cover ComfyUI's SDTokenizer family and
    # HF-style tokenizer wrappers respectively.
    found = {}
    for name, attrs in (
        ("start", ("start_token", "bos_token_id")),
        ("end", ("end_token", "eos_token_id")),
        ("pad", ("pad_token", "pad_token_id")),
    ):
        for attr in attrs:
            val = getattr(sub_tokenizer, attr, None)
            if isinstance(val, int):
                found[name] = val
                break

    # Without at least an end or pad id we can't separate content from
    # structure, so report unusable rather than half-trusting.
    if "end" not in found and "pad" not in found:
        return None
    return found


def _count_chunk_with_specials(ids, specials):
    """
    Count content tokens and remaining pad slots in one chunk using
    resolved special ids. Returns (content_count, pad_slots).
    pad_slots is None when no pad id was resolved.
    """
    special_ids = set(specials.values())
    content = sum(1 for t in ids if t not in special_ids)

    pad_id = specials.get("pad")
    if pad_id is None:
        return content, None

    pad_slots = sum(1 for t in ids if t == pad_id)
    # In pad-with-end schemes the genuine EOS shares the pad id, so one
    # occurrence is structure, not free space.
    if specials.get("end") == pad_id and pad_slots > 0:
        pad_slots -= 1

    return content, pad_slots


def _count_chunk_fallback(ids, strip_leading_bos):
    """
    Structural content count for tokenizers whose special ids couldn't
    be resolved. Approximate by design; callers must label it as such.
    """
    if not ids:
        return 0

    # The trailing run of the final id is padding (or pad-with-end,
    # in which case the run includes the EOS).
    end = len(ids)
    run_id = ids[-1]
    while end > 0 and ids[end - 1] == run_id:
        end -= 1

    # Zero-padding schemes place a distinct EOS before the pad run;
    # non-zero runs are assumed pad-with-end and already consumed EOS.
    if run_id == 0 and end > 0:
        end -= 1

    start = 1 if strip_leading_bos else 0
    return max(0, end - start)


def _analyze_tokens(clip, tokens, debug):
    """
    Analyze a clip.tokenize() result across all encoders and chunks.
    Returns (per_encoder_counts, info_lines).
    """
    counts = []
    lines = []

    for key, chunks in tokens.items():
        sub = _resolve_sub_tokenizer(clip, key)
        specials = _resolve_special_tokens(sub)
        chunk_len = max((len(c) for c in chunks), default=0)
        chunk_count = len(chunks)
        chunk_word = "chunk" if chunk_count == 1 else "chunks"

        if debug:
            src = type(sub).__name__ if sub is not None else "None"
            print(f"[CCN TokenCounter] encoder '{key}': tokenizer={src}, "
                  f"specials={specials}, chunks={chunk_count}, chunk_len={chunk_len}")

        if specials is not None:
            total = 0
            pad_total = 0
            pad_known = True
            for c in chunks:
                content, pads = _count_chunk_with_specials(_chunk_ids(c), specials)
                total += content
                if pads is None:
                    pad_known = False
                else:
                    pad_total += pads

            if pad_known:
                capacity = total + pad_total
                pct = (100.0 * total / capacity) if capacity > 0 else 0.0
                warn = ""
                if chunk_count == 1:
                    if pad_total == 0:
                        warn = " \u26a0 chunk full"
                    elif pct > 80:
                        warn = " \u00b7 nearing capacity"
                line = (f"{key}: {total} tokens \u00b7 {chunk_count} {chunk_word} "
                        f"of {chunk_len} \u00b7 {pct:.0f}%{warn}")
            else:
                line = (f"{key}: {total} tokens \u00b7 {chunk_count} {chunk_word} "
                        f"of {chunk_len}")
        else:
            # Structural fallback: leading BOS is only detectable when
            # multiple chunks share the same first id.
            shared_bos = (
                chunk_count > 1
                and len({_chunk_ids(c)[0] for c in chunks if len(c) > 0}) == 1
            )
            total = sum(
                _count_chunk_fallback(_chunk_ids(c), shared_bos) for c in chunks
            )
            line = (f"{key}: ~{total} tokens (approx) \u00b7 {chunk_count} "
                    f"{chunk_word} of {chunk_len}")

        counts.append(total)
        lines.append(line)

    return counts, lines


class TokenCounter:
    """
    Count tokens in text across all of a model's text encoders,
    displaying the result on the node. token_count is the max across
    encoders. Falls back to a character estimate without CLIP.
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
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("token_count", "info")
    FUNCTION = "count_tokens"
    OUTPUT_NODE = True

    def _result(self, token_count, info):
        return {"ui": {"ccn_token_info": [info]}, "result": (token_count, info)}

    def count_tokens(self, text, clip=None, debug=False):
        if not text.strip():
            return self._result(0, "0 tokens (empty text)")

        if clip is not None:
            try:
                tokens = clip.tokenize(text)
                counts, lines = _analyze_tokens(clip, tokens, debug)

                token_count = max(counts) if counts else 0
                if len(counts) > 1:
                    lines.append(f"token_count (max): {token_count}")

                info = "\n".join(lines)
                if debug:
                    print(f"[CCN TokenCounter]\n{info}")

                return self._result(token_count, info)

            except Exception as e:
                print(f"[CCN TokenCounter] Warning: tokenizer failed "
                      f"({type(e).__name__}: {e}), falling back to estimate")

        # Rough English average of ~4 characters per token.
        char_count = len(text)
        estimated = max(1, char_count // 4)
        info = (f"~{estimated} tokens (estimate from {char_count} chars, "
                f"connect CLIP for accuracy)")
        if debug:
            print(f"[CCN TokenCounter] {info}")

        return self._result(estimated, info)


class ConditioningTokenCount:
    """
    Read the sequence length (token positions) from a conditioning
    tensor and display it on the node.
    """

    CATEGORY = "ComfyCollectorNodes/Utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("sequence_length", "info")
    FUNCTION = "count"
    OUTPUT_NODE = True

    def _result(self, seq_len, info):
        return {"ui": {"ccn_token_info": [info]}, "result": (seq_len, info)}

    def count(self, conditioning, debug=False):
        if not conditioning or len(conditioning) == 0:
            return self._result(0, "Empty conditioning")

        cond = conditioning[0][0]
        shape = cond.shape

        if len(shape) >= 2:
            seq_len = shape[1]
            hidden_dim = shape[2] if len(shape) > 2 else "?"
            info = (f"Sequence length: {seq_len} \u00b7 hidden dim: {hidden_dim} \u00b7 "
                    f"shape: {list(shape)}")
        else:
            seq_len = shape[0]
            info = f"Sequence length: {seq_len} \u00b7 shape: {list(shape)}"

        if debug:
            print(f"[CCN ConditioningTokenCount] {info}")

        return self._result(seq_len, info)
