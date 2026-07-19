# Utilities

Workflow plumbing: counters, variables, timers, inspection, JSON loading, and
small math helpers.

## At a glance

| Node | Summary |
|------|---------|
| [Better Int](#better-int) | An integer that increments or randomizes itself before or after each run, with the widget kept in sync. |
| [Gated Increment](#gated-increment) | An integer that only changes once every N runs — the counter survives restarts. |
| [Property](#property) | A named variable: wire a value in to set it, leave it unwired to get it — teleport values across a graph. |
| [Property Clear](#property-clear) | Clears stored properties by scope and/or key. |
| [Property List](#property-list) | Lists all stored properties with value summaries. |
| [Random Select](#random-select) | Randomly forwards one of up to 5 connected inputs. |
| [Float Lerp](#float-lerp) | Unclamped linear interpolation between two floats. |
| [Timer Start / Timer Stop](#timer-start-timer-stop) | Measure wall-clock time between two points in a graph. |
| [Print](#print) | Prints a custom message to the console when it executes. |
| [Inspect Tensor](#inspect-tensor) | Type-aware inspection of any value, with pass-through. |
| [Load JSON File](#load-json-file) | Loads a JSON file from the input directory via a dropdown. |
| [Load JSON File Path](#load-json-file-path) | Loads a JSON file from an arbitrary path. |
| [Token Counter](#token-counter) | Counts a prompt's tokens against the model's budget. |
| [Conditioning Token Count](#conditioning-token-count) | Reads the sequence length straight from a conditioning tensor. |

---

## Counters & values

### Better Int

**An integer constant that can change its own value — by a fixed step or by
randomizing — before or after each run, with the on-node widget kept in
sync.**

Unlike the built-in `control_after_generate` (which only mutates the widget
after execution), the `*_before` modes apply the change first, so the output
reflects the new value on the current run. After each run the widget shows the
current number, so you can pause a batch, read it, and resume.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value` | INT | 0 | The current integer (base for increments). |
| `mode` | choice | `fixed` | `fixed`, `increment_before`, `increment_after`, `random_before`, `random_after`. |
| `step` | INT | 1 | Amount added per increment. |
| `debug` | BOOLEAN | false | Log mode/input/output details. |

**Outputs:** `INT` — the value emitted this run.

Random modes draw from `0`–`4294967295` (seed-sized). In any non-`fixed` mode
the node re-executes on every queue.

### Gated Increment

**An integer that only changes once every N runs — hold a seed or value stable
for a few generations, then advance.**

Each execution advances an internal counter; when it reaches `rate` the value
increments (or randomizes) and the counter resets. The counter is persisted to
disk per node, so it survives ComfyUI restarts, and the node's title shows the
live count, e.g. `Gated Increment (CCN) [3 / 5]`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value` | INT | 0 | Current integer / base for increment. |
| `rate` | INT | 5 | Number of runs between each value change. |
| `mode` | choice | `increment` | `increment` adds `step`; `random` picks a new seed-sized value. |
| `timing` | choice | `before` | Whether the changed value is emitted this run (`before`) or next (`after`). |
| `step` | INT | 1 | Amount added in increment mode. |
| `reset_counter` | BOOLEAN | false | Reset the counter to 0: toggle on, run once, toggle off. |
| `debug` | BOOLEAN | false | Log details. |

**Outputs:** `INT`. Re-executes every queue (necessary for the counter to
advance).

### Random Select

**Randomly forwards one of up to 5 connected inputs (any type) to the
output.**

Only connected inputs are candidates. With `seed = -1` the pick is fresh each
run; a non-negative seed makes it reproducible.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_1`–`input_5` | any | *(optional)* | Candidate values. |
| `seed` | INT | -1 | `-1` = unseeded; `≥ 0` = reproducible selection. |

**Outputs:** `output` (the chosen value), `selected_index` (1-based; 0 if
nothing connected).

### Float Lerp

**Unclamped linear interpolation: `a + (b − a) · t`.**

`t` outside 0–1 extrapolates on purpose.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `a` | FLOAT | 0.0 | Value at `t = 0`. |
| `b` | FLOAT | 1.0 | Value at `t = 1`. |
| `t` | FLOAT | 0.0 | Blend factor, unclamped. |

**Outputs:** `FLOAT`.

!!! info "Related"
    For interpolating along an arbitrary shape rather than a straight line,
    see [Curve Sample](sampling.md#curve-sample).

---

## Variables

The Property nodes give a graph named variables held in server memory, in two
scopes: **workflow** (temporary, per-run working values) and **session**
(persists until the server restarts or is explicitly cleared).

### Property

**A named variable in one node: wire a value into `value` to set it (and pass
it through), leave `value` unwired to get it — teleporting values across a
graph without wires.**

Setter and getter are the same node with the same `name`. Because execution
order between a setter and getter isn't guaranteed by name alone, wire the
setter's `trigger_out` into the getter's `trigger` when ordering matters. A
getter that finds nothing returns `None` by default (useful on first runs), or
raises with a list of available keys when `error_if_missing` is on. If the key
isn't in the chosen scope, the other scope is checked as a fallback.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | STRING | `my_var` | The variable key; setter and getter must match. |
| `session_scope` | BOOLEAN | false | Off = workflow scope; on = session scope. |
| `error_if_missing` | BOOLEAN | false | Raise instead of returning `None` when the key is absent. |
| `value` | any | *(optional)* | Connected = set; disconnected = get. |
| `trigger` | any | *(optional)* | Execution-order dependency, passed through to `trigger_out`. |
| `debug` | BOOLEAN | false | Log store operations. |

**Outputs:** `value` (the stored or retrieved value), `trigger_out`.

<!-- TODO: screenshot — a Property setter and getter pair teleporting a value across the graph -->

### Property Clear

**Clears stored properties — a whole scope, or a single key.**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scope` | choice | — | `workflow`, `session`, or `all`. |
| `key` | STRING | `*` | A specific key, or `*` for every key in the scope. |
| `trigger` | any | *(optional)* | Execution-order dependency. |
| `debug` | BOOLEAN | false | Log what was cleared. |

**Outputs:** `trigger_out`.

### Property List

**Lists all stored properties with short value summaries (tensor shapes, dict
keys, truncated strings), per scope.**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scope` | choice | — | `workflow`, `session`, or `all`. |
| `trigger` | any | *(optional)* | Execution-order dependency. |

**Outputs:** `listing` (the report), `trigger_out`.

---

## Timing & logging

### Timer Start / Timer Stop

**Measure wall-clock time between two points in a graph.**

Timer Start outputs a timestamp; wire it into Timer Stop's `start_time`, and
wire the output of the work you're measuring into Timer Stop's `trigger` so it
evaluates after that work. The elapsed time comes out both human-readable
(`4m32s`) and as raw seconds, and is printed to the console.

**Timer Start** — no inputs. **Outputs:** `start_time` (FLOAT).

**Timer Stop:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_time` | FLOAT (input) | — | Wired from Timer Start. |
| `trigger` | any | *(optional)* | Forces evaluation after the measured work; value ignored. |

**Outputs:** `formatted` (e.g. `1h02m05s`), `seconds`.

### Print

**Prints a custom message to the server console when it executes.**

Connect anything to `trigger` to control when it fires and to pass that value
onward.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | STRING | `Hello` | The message (printed as `[CCN] <text>`). |
| `trigger` | any | *(optional)* | Execution-order dependency, passed through. |

**Outputs:** `trigger` (passed through).

### Inspect Tensor

**Type-aware inspection of any value — tensors get shape/dtype/device and
min/max/mean/std, dicts get per-key summaries — printed to the console and
returned as a string, with the input passed through.**

Insert it inline anywhere without breaking a connection.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `label` | STRING | `Tensor` | Report header. |
| `data` | any | *(optional)* | The value to inspect. |

**Outputs:** `passthrough` (the input, unchanged), `info` (the report).

---

## Files & tokens

### Load JSON File

**Loads a `.json` file from ComfyUI's input directory (chosen from a dropdown)
and returns its raw text plus the filename.**

The dropdown lists JSON files found recursively under the input directory.
Content is validated as JSON — invalid files return an empty string. Edits to
the file on disk re-trigger execution automatically.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `json_file` | choice | — | A JSON file from the input directory. |

**Outputs:** `json_string`, `filename`.

### Load JSON File Path

**The same JSON loader, but for an arbitrary path typed into a string field.**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | STRING | `""` | Absolute or relative path to a JSON file. |

**Outputs:** `json_string`, `filename`.

### Token Counter

**Counts how many tokens a prompt uses, against the connected model's token
budget.**

With a CLIP connected it uses the real tokenizer and reports
`used/capacity (percent)` with warnings when nearing or hitting the limit
(SD-style CLIP has a ~77-token window; T5 encoders like Wan/Flux allow ~512).
Without CLIP it falls back to a rough ~4-characters-per-token estimate.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | STRING | `""` | The prompt. |
| `clip` | CLIP | *(optional)* | Enables an exact count. |

**Outputs:** `token_count`, `info`.

### Conditioning Token Count

**Reads the sequence length — the number of token positions — directly from a
conditioning tensor.**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conditioning` | CONDITIONING | — | The conditioning to measure. |

**Outputs:** `sequence_length`, `info` (includes hidden dim and full shape).
