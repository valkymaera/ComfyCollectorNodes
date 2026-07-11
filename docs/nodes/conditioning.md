# Conditioning

Nodes for creating, shaping, editing, combining, and inspecting `CONDITIONING`.

## At a glance

| Node | Summary |
|------|---------|
| [Token Remap](#token-remap) | Encodes a prompt while blending specific words toward alternate meanings at the embedding level. |
| [CLIP Remap](#clip-remap) | Wraps a CLIP model so every downstream encode has your word replacements applied automatically. |
| [Emphasis Encode](#emphasis-encode) | Encodes text with A1111-style `(word:weight)` emphasis on Wan/T5-style encoders. *(Experimental)* |
| [Emphasis Encode Advanced](#emphasis-encode-advanced) | Emphasis Encode plus a post-emphasis normalization step. *(Experimental)* |
| [Concept Remap](#concept-remap) | Shifts already-encoded conditioning along the direction from one concept to another, targeted to where the concept lives. |
| [Hyper Remap](#hyper-remap) | All-in-one remapper: text replacement, token blending, concept nudging, and delta residuals in a single node. |
| [Hyper Remap Slim](#hyper-remap-slim) | Wire-oriented Hyper Remap with a single output and fewer widgets. |
| [Conditioning Projection Removal](#conditioning-projection-removal) | Suppresses a concept by projecting it out of the positive — a negative prompt substitute for flow models. |
| [Conditioning Scale](#conditioning-scale) | Multiplies the conditioning tensor by a constant to strengthen or weaken it. |
| [Conditioning Normalizer](#conditioning-normalizer) | Applies one of several normalization methods to the tensor, blended by strength. |
| [Conditioning Clamp](#conditioning-clamp) | Clamps every tensor value to a min/max range to tame extremes. |
| [Conditioning Lerp](#conditioning-lerp) | Linearly interpolates between two conditionings. |
| [Conditioning Subtract](#conditioning-subtract) | Subtracts one conditioning from another to conceptually remove an idea. |
| [Conditioning Stats](#conditioning-stats) | Prints shape/min/max/mean/std to the console and passes conditioning through. |
| [Token Inspector](#token-inspector) | Reports how a CLIP model tokenizes a prompt, token by token. |

## How conditioning is structured

In ComfyUI, a `CONDITIONING` value is a list of `[tensor, dict]` pairs. The
tensor holds per-token embeddings with shape `[batch, sequence_length,
embedding_dim]` — one vector per token position. The dict carries metadata,
most commonly `pooled_output` (a single summary vector) plus any timestep or
area hints attached upstream.

The nodes on this page fall into five groups by what they do to that structure:

- **Encoding** — turn text into conditioning (replacements for `CLIPTextEncode`).
- **Concept & direction editing** — shift *meaning* in embedding space.
- **Magnitude & distribution shaping** — do math on the tensor values.
- **Combining** — merge two conditionings into one.
- **Inspection** — report on conditioning without changing it.

!!! info "Related"
    The **Neutral Prompt** family and the CFG guiders act on conditioning too,
    but at sampling time — they live under
    [Sampling & Guidance](sampling.md). **Conditioning Token Count** is under
    [Utilities](utilities.md).

---

## Encoding

These produce conditioning directly from text, in place of a standard
`CLIPTextEncode`.

### Token Remap

**Encodes a prompt while blending specific words toward alternate meanings at
the embedding level — a soft, controllable word swap.**

Useful for disambiguating a word (e.g. nudging "ship" toward "starship")
without a hard text swap. It encodes the prompt twice, once as written and once
with your word replacements applied, then blends the two. When the token counts
match, blending happens only at the positions that changed (precise mode); when
they differ, it falls back to a global blend of the whole tensor. Use this in
place of `CLIPTextEncode`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | CLIP | — | The CLIP model to encode with. |
| `text` | STRING | — | The prompt. |
| `remappings` | STRING | — | One `source -> target` per line. Accepts `->`, `=>`, `:`, or `,` as the separator; lines starting with `#` are comments. |
| `blend` | FLOAT | 1.0 | `0.0` = original prompt, `1.0` = fully remapped. Values outside 0–1 extrapolate in embedding space. |

**Outputs:** `conditioning`, `text_out` (the original text, passed through).

<!-- TODO: screenshot — Token Remap wired in place of CLIPTextEncode -->

### CLIP Remap

**Wraps a CLIP model so that *every* prompt encoded with it has your word
replacements applied automatically during tokenization.**

Place it before any `CLIPTextEncode` (or other encoder) and all downstream
encodes inherit the remap. This is a **hard** remap with no blending — reach
for Token Remap instead when you want a controllable blend.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | CLIP | — | The CLIP model to wrap. |
| `remappings` | STRING | — | One `source -> target` per line (same formats as Token Remap). |
| `enabled` | BOOLEAN | true | When off, the CLIP passes through untouched. |

**Outputs:** `CLIP` (a patched clone; the original is left intact).

### Emphasis Encode

**Encodes text with A1111-style `(word:weight)` emphasis markers on Wan/T5-style
encoders that don't natively support them.**

!!! note "Experimental"
    Token boundaries don't map cleanly to word boundaries on these models, so
    emphasis positions are estimated and results are approximate. It works best
    with simple emphasis on individual words or short phrases.

Supported syntax:

- `(word:1.2)` — scale a word's weight by 1.2×
- `(word:0.5)` — reduce emphasis
- `(several words:1.3)` — emphasize a phrase
- `((word))` — shorthand for `(word:1.1)`
- nesting compounds, e.g. `((word:1.2))` → 1.2 × 1.1

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | CLIP | — | The encoder. |
| `text` | STRING | — | Prompt with optional emphasis markers. |
| `debug` | BOOLEAN | false | Print token-mapping diagnostics to the console. |

**Outputs:** `conditioning`, `parsed_info` (a summary of detected emphasis),
`debug_output` (the full diagnostic string).

### Emphasis Encode Advanced

**Emphasis Encode plus a post-emphasis normalization step modeled on A1111's
emphasis modes.**

!!! note "Experimental"
    Same emphasis parsing as Emphasis Encode; the same caveats apply.

Adds a `normalization` choice applied after emphasis is baked in, to counteract
the way weighting shifts the overall distribution of the conditioning.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | CLIP | — | The encoder. |
| `text` | STRING | — | Prompt with optional emphasis markers. |
| `normalization` | choice | `mean_restore` | One of `none`, `mean_restore`, `max_norm`, `std_norm`. Restores the chosen statistic to its pre-emphasis value. |
| `debug` | BOOLEAN | false | Print diagnostics to the console. |

**Outputs:** `conditioning`, `parsed_info`, `debug_output`.

---

## Concept & direction editing

These edit the *meaning* carried by conditioning rather than its raw magnitude —
operating on the direction of embedding vectors.

### Concept Remap

**Shifts already-encoded conditioning along the direction from one concept to
another (e.g. `water -> fire`), targeted to the positions where the source
concept is most influential.**

Because the shift follows *influence* rather than exact token positions, it
also catches contextual bleed from attention — reflections, color, mood. It
has two modes for locating that influence:

- **Cosine** (default) — encodes the source word separately and uses cosine
  similarity to estimate where the concept lives. Fast and approximate; doesn't
  need the original prompt.
- **Differential** — when `prompt_text` is provided, encodes the prompt with and
  without the source word and measures the actual per-position difference. Much
  more precise. Falls back to cosine per-pair if it can't apply (word absent
  from the prompt, token-count mismatch, etc.).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conditioning` | CONDITIONING | — | The conditioning to edit. |
| `clip` | CLIP | — | Used to encode concept directions. |
| `remappings` | STRING | — | One `source -> target` per line. |
| `blend` | FLOAT | 1.0 | Overall strength of the shift. `1.0` = full direction vector; `>1` overshoots; negative pushes *away* from the target. |
| `sharpness` | FLOAT | 1.0 | How selectively the effect targets matching positions. `0` = uniform shift everywhere; higher = concentrated on matching positions; negative inverts (affects everything *except* the source concept). |
| `threshold` | FLOAT | 0.0 | Minimum influence weight for a position to be affected at all. |
| `prompt_text` | STRING | *(optional)* | The original prompt. Supplying it enables differential mode. |
| `debug` | BOOLEAN | false | Print per-concept diagnostics. |

**Outputs:** `conditioning`, `prompt_text_out`.

<!-- TODO: screenshot — cosine vs differential result comparison -->

### Hyper Remap

**One node, four remap operators: literal text replacement, token-level
embedding blending, concept-direction nudging, and additive delta residuals —
each line of the remappings box picks its operator by separator.**

Hyper Remap replaces a chain of Token Remap + Concept Remap (and more) with a
single node. It parses each remapping line, routes it to one of four phases by
its separator, and runs the phases in a fixed cascade — each phase consuming
the previous one's output:

1. **String replacement** (`find, replace`) — plain-text substitution on the
   prompt *before* encoding. Substring match, no embedding math.
2. **Token remap** (`source -> target`) — encodes the prompt with the word
   swapped and blends the two encodings, exactly like the standalone
   [Token Remap](#token-remap) (precise per-position blend when token counts
   match, global blend otherwise).
3. **Concept remap** (`source => target`) — nudges the conditioning along the
   source→target direction, weighted by where the source concept is most
   influential, like the standalone [Concept Remap](#concept-remap).
   Differential mode is used automatically (the prompt text is known);
   individual pairs fall back to cosine mode when differential can't apply.
4. **Delta remap** (`base ~~ subtracted`) — encodes two arbitrary prompts,
   subtracts them, and adds the weighted difference into the conditioning as a
   residual. This injects the semantic *difference* between two prompts rather
   than replacing anything.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | CLIP | — | Used for all tokenizing and encoding (prompt, remapped variants, concepts, delta pairs). |
| `text` | STRING | — | The main prompt (dynamic prompts supported). |
| `remappings` | STRING | — | One entry per line or semicolon-separated; `#` comments allowed. Separator picks the operator: `,` / `->` / `=>` / `~~`. |
| `blend` | FLOAT | 1.0 | Default strength for all operators. For `->`: lerp between original and remapped. For `=>` and `~~`: magnitude of the nudge/delta vector. `>1` overshoots, negative inverts. |
| `sharpness` | FLOAT | 1.0 | For `=>`/`~~`: how sharply positions are weighted by similarity to the source/base concept. `0` = uniform; higher = concentrated; negative favors *least*-similar positions. Ignored by `->`. |
| `threshold` | FLOAT | 0.0 | For `=>`/`~~`: masks out positions whose similarity weight falls below this value (only values > 0 activate masking). Ignored by `->`. |
| `normalize_delta` | BOOLEAN | true | L2-normalize the `~~` delta before blending, so `blend` has a consistent magnitude regardless of how different the two prompts are. |
| `case_sensitive` | BOOLEAN | true | Case sensitivity for `,` string-replacement pairs only. |
| `debug` | BOOLEAN | false | Print per-phase diagnostics to the console. |

**Outputs:** `conditioning` (after all phases), `untouched_conditioning` (the
original text encoded verbatim, for A/B comparison), `original_prompt`,
`modified_prompt` (the text after string replacement).

**Per-line overrides.** Any `->`, `=>`, or `~~` line can end with a
parenthetical that overrides the node-level defaults for just that pair:

```text
ship -> starship (1.5)                      # bare number = blend
water => fire (b:0.8, s:2.0, t:0.1)         # blend, sharpness, threshold
bee in wild ~~ insect (b:0.5, sx:2.0, tx:0.1)
```

`b`/`s`/`t` mirror the node's blend/sharpness/threshold. `~~` lines also accept
`sx`/`tx` — an *intrinsic* sharpness/threshold that weights positions by how
strongly they contribute to the delta itself, multiplied with the
similarity-based weighting. String-replacement (`,`) lines take no overrides.

!!! tip "Performance"
    Every `->` pair costs an extra CLIP encode, every `~~` pair two, and a
    differential `=>` pair up to four. Long remap lists multiply encoding time
    accordingly.

**Matching rules.** `,` replacement is a plain substring replace on the text;
`->`, `=>`, and differential weighting match whole words, case-insensitively.

<!-- TODO: screenshot — Hyper Remap with a multi-operator remappings box and A/B outputs -->

### Hyper Remap Slim

**Hyper Remap with a wire-only prompt input and a single conditioning output —
same engine, smaller node.**

Slim runs the identical four-phase pipeline (it calls the same internal
functions as [Hyper Remap](#hyper-remap)), with the same remapping syntax and
per-line overrides. Differences:

- `text` is input-only (connect it from an upstream STRING output; no text box,
  no dynamic-prompt expansion).
- Only the remapped `conditioning` is output — no untouched baseline or prompt
  strings (this also saves one full encode).
- No `debug` or `case_sensitive` options; string replacement is always
  case-sensitive.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | CLIP | — | Used for all encoding. |
| `text` | STRING (input) | — | The prompt, wired from upstream. |
| `remappings` | STRING | — | Same syntax as Hyper Remap. |
| `blend` | FLOAT | 1.0 | Same as Hyper Remap. |
| `sharpness` | FLOAT | 1.0 | Same as Hyper Remap. |
| `threshold` | FLOAT | 0.0 | Same as Hyper Remap. |
| `normalize_delta` | BOOLEAN | true | Same as Hyper Remap. |

**Outputs:** `conditioning`.

### Conditioning Projection Removal

**Suppresses a concept by projecting it out of the positive conditioning — a
lightweight, pre-attention alternative to a negative prompt.**

Aimed at flow-based models (Flux, SD3) that don't support CFG-based negatives.
It collapses the negative conditioning into a single direction vector, then
removes each positive token's component along that direction.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `positive` | CONDITIONING | — | The conditioning to clean up. |
| `negative` | CONDITIONING | — | The concept to remove. |
| `scale` | FLOAT | 1.0 | `1.0` removes exactly the negative's directional component; `>1` overcorrects, actively pushing away; `0` = no effect. |
| `pooling` | choice | `mean` | How to collapse the negative's tokens into one direction: `mean` (average), `max` (the single highest-norm token), or `weighted_norm` (norm-weighted average, emphasizing stronger tokens). |
| `debug` | BOOLEAN | false | Print projection diagnostics. |

**Outputs:** `conditioning`.

!!! tip "Multi-concept negatives"
    Pooling averages the whole negative into one direction, which dilutes each
    concept when the negative contains several. For complex negatives, chain
    multiple nodes in sequence, each with a single focused negative.

---

## Magnitude & distribution shaping

These do arithmetic on the conditioning tensor's values, leaving token meaning
in place but changing how strongly or how evenly it drives generation.

### Conditioning Scale

**Multiplies the conditioning tensor by a constant to strengthen or weaken the
prompt's influence.**

Unlike normalization, this simply amplifies or reduces magnitude with no
reshaping.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conditioning` | CONDITIONING | — | Input. |
| `scale` | FLOAT | 1.0 | Multiplier, `0.0`–`10.0`. `1.0` passes through unchanged. |

**Outputs:** `conditioning`.

### Conditioning Normalizer

**Applies a normalization method to the conditioning tensor, then blends the
result back toward the original by strength.**

Inspired by A1111's emphasis normalization; even without explicit emphasis
weights, changing the value distribution can subtly shift generation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conditioning` | CONDITIONING | — | Input. |
| `method` | choice | `none` | See methods below. |
| `strength` | FLOAT | 1.0 | Blend between original (`0`) and fully normalized (`1`). Values outside 0–1 extrapolate. |

**Methods:** `max_norm` (divide by max absolute value), `std_norm` (divide by
standard deviation), `std_half` (gentler, divide by 2×std), `zscore` (subtract
mean, divide by std), `zscore_avg` (average of z-score and max-norm),
`zscore_half` (gentler z-score), `slight_z` (20% z-score, 80% max-norm),
`mean_restore` (normalize but restore the original mean), `range` (rescale to
`[-1, 1]`), and hard clamps `clamp_1`, `clamp_1.5`, `clamp_2`.

**Outputs:** `conditioning`.

### Conditioning Clamp

**Clamps every value in the conditioning tensor to a min/max range.**

Useful for taming extreme values that produce "burn" or artifacts.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conditioning` | CONDITIONING | — | Input. |
| `min_value` | FLOAT | -4.0 | Lower bound. |
| `max_value` | FLOAT | 4.0 | Upper bound. |

**Outputs:** `conditioning`.

---

## Combining

These merge two conditionings into one.

### Conditioning Lerp

**Linearly interpolates between two conditionings.**

`result = a·(1 − blend) + b·blend`. If the two have different sequence lengths,
both are truncated to the shorter before blending. `pooled_output` is
interpolated too when both inputs carry it.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conditioning_a` | CONDITIONING | — | Endpoint at `blend = 0`. |
| `conditioning_b` | CONDITIONING | — | Endpoint at `blend = 1`. |
| `blend` | FLOAT | 0.5 | `0.0` = all A, `0.5` = even mix, `1.0` = all B. |

**Outputs:** `conditioning`.

### Conditioning Subtract

**Subtracts one conditioning from another — a lightweight way to conceptually
remove an idea.**

`result = a − b·strength`, e.g. a full-scene conditioning minus a "snow"
conditioning. Sequence lengths are truncated to the shorter; pooled outputs are
subtracted when present.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conditioning_a` | CONDITIONING | — | The base. |
| `conditioning_b` | CONDITIONING | — | What to subtract. |
| `strength` | FLOAT | 1.0 | How much of B to remove, `0.0`–`5.0`. |

**Outputs:** `conditioning`.

---

## Inspection

These report on conditioning and pass it through unchanged, so they can sit
inline in a graph without altering the result.

### Conditioning Stats

**Prints each entry's shape, min, max, mean, standard deviation, and pooled
keys to the console, and passes the conditioning through untouched.**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conditioning` | CONDITIONING | — | Input (passed through). |
| `label` | STRING | `Conditioning` | Heading printed above the stats. |

**Outputs:** `conditioning` (unchanged).

### Token Inspector

**Shows how a CLIP model tokenizes a prompt — every token, its ID, and a
per-encoder content-token count — as a human-readable string.**

Handy for understanding why a remap or emphasis landed where it did.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | CLIP | — | The tokenizer to inspect. |
| `text` | STRING | — | The prompt to tokenize. |

**Outputs:** `STRING` (the tokenization report).
