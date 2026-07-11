# Sampling & Guidance

Nodes that change how the sampler is guided: curve-scheduled CFG, CFG-Zero*,
and the Neutral Prompt family for combining auxiliary prompts at the CFG step.

## At a glance

| Node | Summary |
|------|---------|
| [Curve](#curve) | Defines a reusable Hermite curve in a visual editor and outputs it as a `CCN_CURVE`. |
| [Curve Sample](#curve-sample) | Evaluates a curve at a position and returns the float — drive any value from a drawn curve. |
| [Curve CFG Guider](#curve-cfg-guider) | A guider whose CFG scale follows a drawn curve across the sampling steps. |
| [CFG-Zero* Scaled](#cfg-zero-scaled) | CFG-Zero* guidance with a continuous strength blend and configurable early-step attenuation. *(Experimental)* |
| [Neutral Prompt](#neutral-prompt) | Patches a model so an auxiliary prompt merges into CFG via perpendicular, salient, or top-k strategies. |
| [Neutral Prompt Entry](#neutral-prompt-entry) | Packages one auxiliary conditioning + strategy into a chainable entry list for the guider. |
| [Neutral Prompt Empty](#neutral-prompt-empty) | Outputs an empty entry list — the "disabled" branch for switches. |
| [Neutral Prompt Guider](#neutral-prompt-guider) | Curve-scheduled CFG guider that also applies Neutral Prompt strategies natively. |

---

## Curves

A **curve** in this pack (`CCN_CURVE` socket type) is a list of Hermite
keyframes — position, value, and tangents in normalized 0–1 space — evaluated
with cubic Hermite interpolation, the same basis as Unity's AnimationCurves.
Nodes that take a curve replace their `curve_data` text field with an
interactive graphical editor:

- **Drag** keyframes to move them; endpoints are pinned at x=0 and x=1 (their
  height stays free).
- **Drag tangent handles** to change slope; **Shift+drag** breaks the tangent
  pair to adjust one side only.
- **Double-click** empty curve area to add a keyframe; double-click a keyframe
  to delete it.
- **Right-click** a keyframe for break/mirror/delete options, or the widget
  for "Reset Curve".

The identical evaluation code runs in Python and in the editor, so the preview
matches sampling exactly. When an external `curve` input is connected, the
editor becomes a read-only live view of the connected curve.

<!-- TODO: screenshot — the curve editor widget with keyframes and tangent handles -->

### Curve

**Defines a reusable curve in a visual editor and outputs it as a `CCN_CURVE`
value.**

Wire the output into [Curve Sample](#curve-sample),
[Curve CFG Guider](#curve-cfg-guider), or
[Neutral Prompt Guider](#neutral-prompt-guider) so one editor drives several
consumers.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `curve_data` | curve editor | linear 0→1 | The curve (stored as JSON, edited visually). |

**Outputs:** `curve` (`CCN_CURVE`).

### Curve Sample

**Evaluates a curve at a single position and returns the resulting float.**

Turns a drawn curve into a usable scalar — a strength, a weight, any float
input. With `x_scale` you can map an input range `0..x_scale` onto the curve's
0–1 domain (e.g. sample by frame number), and `y_scale` multiplies the output.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `t` | FLOAT | 0.0 | Sample position. |
| `x_scale` | FLOAT | 1.0 | `t` is divided by this before sampling (then clamped to 0–1). |
| `y_scale` | FLOAT | 1.0 | Output multiplier. |
| `curve_data` | curve editor | linear 0→1 | Built-in curve. |
| `curve` | CCN_CURVE | *(optional)* | External curve; overrides `curve_data` when connected. |

**Outputs:** `FLOAT` — the sampled value × `y_scale`.

### Curve CFG Guider

**A guider whose CFG scale follows a drawn curve across the sampling steps,
instead of staying constant.**

Draw the shape — e.g. high guidance early, low late — and the guider maps the
curve's 0–1 output onto `min_cfg`–`max_cfg` at every step. Use it with
`SamplerCustomAdvanced`; the `sigmas` pass straight through so the same
schedule feeds the sampler. Progress is derived from the current sigma rather
than a call counter, so it stays correct with multi-evaluation samplers like
Heun or DPM++ 2M.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | — | The diffusion model. |
| `positive` | CONDITIONING | — | Positive conditioning. |
| `negative` | CONDITIONING | — | Negative conditioning. |
| `sigmas` | SIGMAS | — | The sampling schedule (also passed through). |
| `min_cfg` | FLOAT | 1.0 | CFG when the curve outputs 0. |
| `max_cfg` | FLOAT | 7.0 | CFG when the curve outputs 1. |
| `mode` | choice | `step` | `step` snaps progress to the nearest scheduled step; `sigma` measures progress linearly in sigma. |
| `sigma_decay` | BOOLEAN | false | Additionally attenuate CFG toward 1.0 as noise decreases (weaker guidance in the low-noise phase). |
| `curve_data` | curve editor | descending 1→0 | The CFG curve. |
| `curve` | CCN_CURVE | *(optional)* | External curve; overrides the editor. |

**Outputs:** `guider` (for `SamplerCustomAdvanced`), `sigmas` (passed
through).

<!-- TODO: screenshot — Curve CFG Guider wired into SamplerCustomAdvanced with a falling curve -->

### CFG-Zero* Scaled

**Implements CFG-Zero\* guidance — rescaling the unconditional term by an
optimal projection coefficient — with a continuous strength blend and a
configurable early-step attenuation.**

!!! note "Experimental"
    Based on the [CFG-Zero* paper](https://github.com/WeichenFan/CFG-Zero-star).

CFG-Zero* computes, per batch item, the projection of the conditional
prediction onto the unconditional one and uses it to rescale the unconditional
term, reducing guidance over/under-shoot. This node adds two conveniences over
the reference implementation: `strength` lerps continuously between plain CFG
(0.0) and full CFG-Zero* (1.0), and the paper's "zero-init" of the first step
is generalized to an `init_scale` multiplier over the first `init_steps`
steps. The model is cloned; the original is untouched.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | — | Model to patch. |
| `strength` | FLOAT | 1.0 | `0` = vanilla CFG, `1` = full CFG-Zero*. Unclamped for experimentation. |
| `use_scaled_init` | BOOLEAN | true | Enable the early-step attenuation. |
| `init_scale` | FLOAT | 0.0 | Multiplier applied to the prediction during init steps. `0.0` = the paper's zero-init; `1.0` = no effect. |
| `init_steps` | INT | 0 | Attenuation applies from step 0 through this index, inclusive. |

**Outputs:** `MODEL` (patched clone).

---

## Neutral Prompt family

These combine a main prompt with one or more *auxiliary* prompts at the CFG
step, using strategies ported from ljleb's A1111 `sd-webui-neutral-prompt`
extension. "Neutral" refers to the core idea: the auxiliary conditioning is
made neutral toward the main prompt — its component parallel to the main
direction is projected out — so it contributes only novel, non-contradicting
information instead of fighting the main prompt.

**The three strategies:**

- **perpendicular** — keep only the part of the aux direction orthogonal to
  the main prompt (Perp-Neg style). Adds a concept without contradicting the
  main prompt.
- **salient** — the aux only wins at positions where it activates more
  strongly than the main signal; elsewhere it contributes nothing.
- **top_k** — keep only the strongest `k_ratio` fraction of the aux's
  elements and add those.

Each entry also picks a **side**: `positive` merges the aux into the
conditional side (adding a concept), `negative` merges into the
unconditional side (e.g. composing several negatives that don't interfere
with each other).

There are two ways to use the strategies:

1. **[Neutral Prompt](#neutral-prompt)** — patches the model's CFG function;
   works with a standard KSampler; chain nodes to stack entries.
2. **[Neutral Prompt Guider](#neutral-prompt-guider)** — a native GUIDER for
   `SamplerCustomAdvanced`; aux conditionings go through the standard
   preparation pipeline (areas, masks, hooks, ControlNet, IP-Adapter) and are
   evaluated in a single batched forward pass, plus curve-scheduled CFG.
   Prefer this when your workflow already uses custom sampling.

<!-- TODO: screenshot — comparison grid: base prompt vs aux via perpendicular / salient / top_k -->

### Neutral Prompt

**Patches a MODEL so that during sampling an auxiliary conditioning is merged
into the CFG step with a perpendicular, salient, or top-k strategy — usable
with a standard KSampler.**

Wire the model through the node into the sampler, feed `main_conditioning` in
and its pass-through `conditioning` output to the sampler's positive input.
Chain several Neutral Prompt nodes to stack multiple aux entries; each clone
accumulates onto the previous ones.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | — | Model to patch (cloned). Chain other Neutral Prompt outputs here. |
| `main_conditioning` | CONDITIONING | — | The main positive prompt (passed through to the output). |
| `aux_conditioning` | CONDITIONING | — | The auxiliary conditioning for this entry. |
| `strategy` | choice | — | `perpendicular`, `salient`, or `top_k`. |
| `side` | choice | — | `positive` or `negative`. |
| `weight` | FLOAT | 1.0 | Strength of the aux effect (`-10`–`10`); negative inverts. |
| `k_ratio` | FLOAT | 0.05 | `top_k` only: fraction of elements kept (0.05 = strongest 5%). |
| `cfg_rescale` | FLOAT | 0.0 | Optional std-rescale toward the conditional prediction to curb over-exposure at high CFG (0 = off). Combined across a chain via max. |
| `debug` | BOOLEAN | false | Print per-step diagnostics. |

**Outputs:** `model` (patched clone), `conditioning` (the main conditioning,
passed through for wiring convenience).

### Neutral Prompt Entry

**Packages one auxiliary conditioning plus its strategy settings into an
`NP_ENTRIES` list item for the guider.**

Pure data — no math happens here. Chain entries by wiring one Entry's output
into the next Entry's `entries` input, then feed the final list to
[Neutral Prompt Guider](#neutral-prompt-guider).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conditioning` | CONDITIONING | — | The aux conditioning. |
| `strategy` | choice | — | `perpendicular`, `salient`, or `top_k`. |
| `side` | choice | — | `positive` or `negative`. |
| `weight` | FLOAT | 1.0 | Strength (`-10`–`10`). |
| `k_ratio` | FLOAT | 0.05 | `top_k` only. |
| `entries` | NP_ENTRIES | *(optional)* | Upstream list to append to. |

**Outputs:** `entries` (the accumulated list).

### Neutral Prompt Empty

**Outputs an empty `NP_ENTRIES` list.**

Use it as the "disabled" branch of a switch so the guider always receives a
valid input — with an empty list the guider behaves exactly like a plain
[Curve CFG Guider](#curve-cfg-guider).

**Inputs:** none. **Outputs:** `entries` (empty).

### Neutral Prompt Guider

**A GUIDER that combines curve-scheduled CFG with the Neutral Prompt
strategies, evaluated natively through the guider pipeline.**

A drop-in superset of [Curve CFG Guider](#curve-cfg-guider): the CFG scale
follows the curve between `min_cfg` and `max_cfg`, and every connected entry's
aux conditioning is applied at the CFG step with its strategy. All
conditionings — positive, negative, and every aux — go through standard
preparation (areas, masks, timestep ranges, hooks, ControlNet, IP-Adapter) and
are evaluated in one batched forward pass. Set `min_cfg == max_cfg` for a
flat, non-curved CFG.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | — | The model. |
| `positive` | CONDITIONING | — | Main positive prompt. |
| `negative` | CONDITIONING | — | Negative prompt. |
| `sigmas` | SIGMAS | — | The sampling schedule (also passed through). |
| `min_cfg` | FLOAT | 1.0 | CFG when the curve outputs 0. |
| `max_cfg` | FLOAT | 7.0 | CFG when the curve outputs 1. |
| `mode` | choice | `step` | Progress measurement: `step` or `sigma` (see Curve CFG Guider). |
| `sigma_decay` | BOOLEAN | false | Attenuate CFG toward 1.0 as noise decreases. |
| `cfg_rescale` | FLOAT | 0.0 | Std-rescale toward the conditional prediction (0 = off; applies when entries are present). |
| `debug` | BOOLEAN | false | Print per-entry, per-step diagnostics. |
| `curve_data` | curve editor | descending 1→0 | The CFG curve. |
| `curve` | CCN_CURVE | *(optional)* | External curve; overrides the editor. |
| `np_entries` | NP_ENTRIES | *(optional)* | Entry chain from Neutral Prompt Entry. Empty/absent = plain curve CFG. |

**Outputs:** `guider` (for `SamplerCustomAdvanced`), `sigmas` (passed
through).

<!-- TODO: screenshot — Entry → Entry → Neutral Prompt Guider → SamplerCustomAdvanced wiring -->
