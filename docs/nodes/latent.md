# Latent

Nodes that operate on `LATENT` tensors: value shaping, per-channel adjustments,
and inspection.

## At a glance

| Node | Summary |
|------|---------|
| [Latent Clamp](#latent-clamp) | Clamps every latent value to a min/max range to tame "burn" artifacts. |
| [Latent Scale](#latent-scale) | Multiplies all latent values by a scalar to raise or lower overall energy. |
| [Latent Normalize](#latent-normalize) | Applies one of 15 normalization methods, blended by strength, whole-tensor or per-channel. |
| [Latent Stats](#latent-stats) | Prints shape/min/max/mean/std to the console and passes the latent through. |
| [Latent Channel Offset](#latent-channel-offset) | Adds a per-channel bias to the first 4 latent channels. |
| [Latent Channel Offset x16](#latent-channel-offset-x16) | The 16-channel version, for Wan/Flux/SD3-class latents. |
| [Latent Channel Scale](#latent-channel-scale) | Multiplies each of the first 4 channels by its own factor (negative inverts). |
| [Latent Channel Scale x16](#latent-channel-scale-x16) | The 16-channel multiplicative version. |

---

## Value shaping

### Latent Clamp

**Hard-limits every value in a latent to a `[min, max]` window.**

Diffusion latents can develop a few extreme outliers that decode as "burn" —
blown-out highlights or oversaturated blotches. Clamping the tails back into a
sane range tames those artifacts without reshaping the rest of the
distribution. The before/after min–max is logged to the console.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent` | LATENT | — | Input (cloned, never mutated). |
| `min_value` | FLOAT | -4.0 | Lower bound (`-20`–`0`). |
| `max_value` | FLOAT | 4.0 | Upper bound (`0`–`20`). |

**Outputs:** `latent`.

### Latent Scale

**Multiplies all latent values by a single scalar.**

Amplifying (>1) or attenuating (<1) latent magnitude changes the overall
contrast/energy of the decoded result — e.g. nudging latent energy before a
second sampling pass. A scale of exactly `1.0` passes the input through
untouched.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent` | LATENT | — | Input. |
| `scale` | FLOAT | 1.0 | Multiplier (`0`–`10`). |

**Outputs:** `latent`.

### Latent Normalize

**Rescales or re-centers a latent with one of 15 normalization methods, then
blends the result back toward the original by strength.**

Useful for balancing latent distributions, reducing artifacts, or matching
statistics between stages. The method list is deliberately identical to
[Conditioning Normalizer](conditioning.md#conditioning-normalizer) for
consistency across latent and conditioning workflows.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent` | LATENT | — | Input. |
| `method` | choice | `none` | See methods below. |
| `strength` | FLOAT | 1.0 | Blend between original (`0`) and fully normalized (`1`). Values outside 0–1 extrapolate. |
| `per_channel` | BOOLEAN | false | Normalize each channel independently instead of using whole-tensor statistics. |

**Methods:** `max_norm` (divide by max absolute value), `std_norm` (divide by
standard deviation), `std_half` (gentler, divide by 2×std), `zscore` (subtract
mean, divide by std), `zscore_avg` (average of z-score and max-norm),
`zscore_half` (gentler z-score), `slight_z` (20% z-score, 80% max-norm),
`mean_restore` (normalize but restore the original mean), `range` (rescale to
`[-1, 1]`), and hard clamps `clamp_1`, `clamp_1.5`, `clamp_2`, `clamp_3`,
`clamp_4`.

**Outputs:** `latent`.

### Latent Stats

**Prints a latent's shape, min, max, mean, and standard deviation to the
console and passes it through completely unchanged.**

Drop it inline anywhere in a graph to inspect values.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent` | LATENT | — | Input (true pass-through). |
| `label` | STRING | `Latent` | Header printed above the stats, to tell multiple stat nodes apart. |

**Outputs:** `latent` (unchanged).

---

## Channel adjustment

Latent channels carry loosely semantic content. For SD 1.5/SDXL's 4-channel
latents the rough correlations are: ch0 ≈ cyan–red / brightness-openness,
ch1 ≈ magenta–green / structure, ch2 ≈ yellow–blue tones, ch3 ≈
luminance/contrast — but the effects are abstract and semantic, not literal
color shifts. The **x16** variants exist for 16-channel latents (Wan and other
newer-generation VAEs), whose channel meanings are less documented —
experimentation encouraged. All four nodes safely ignore channels the latent
doesn't actually have.

<!-- TODO: screenshot — same seed decoded with a few different channel offsets, side by side -->

### Latent Channel Offset

**Adds a per-channel constant bias to the first four latent channels.**

Positive values push a channel toward one end of its spectrum, negative toward
the other, shifting the abstract content that channel encodes. Only channels
with a non-zero offset are touched.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent` | LATENT | — | Input. |
| `channel_0`–`channel_3` | FLOAT | 0.0 | Amount added to each channel. |

**Outputs:** `latent`.

### Latent Channel Offset x16

**The same additive offset with 16 channel controls (`ch_00`–`ch_15`), for
16-channel latents.**

Lets you probe or adjust individual channels of the wider latent. Controls
beyond the latent's real channel count are harmlessly ignored.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent` | LATENT | — | Input. |
| `ch_00`–`ch_15` | FLOAT | 0.0 | Amount added to each channel. |

**Outputs:** `latent`.

### Latent Channel Scale

**Multiplies each of the first four channels by its own factor.**

Unlike offset (which shifts), scaling changes a channel's amplitude: `>1`
amplifies its contribution, `<1` reduces it, and negative values invert its
polarity. The default of `1.0` is a true no-op.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent` | LATENT | — | Input. |
| `channel_0`–`channel_3` | FLOAT | 1.0 | Multiplier for each channel. |

**Outputs:** `latent`.

### Latent Channel Scale x16

**The 16-channel multiplicative counterpart, for Wan-style latents.**

Same amplify/reduce/invert semantics per channel.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent` | LATENT | — | Input. |
| `ch_00`–`ch_15` | FLOAT | 1.0 | Multiplier for each channel. |

**Outputs:** `latent`.
