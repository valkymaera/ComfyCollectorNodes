# Prompt & Text

Nodes for assembling prompts from structured sections, persisting them across
runs, and general string manipulation.

## At a glance

| Node | Summary |
|------|---------|
| [Prompt Builder](#prompt-builder) | Joins five narrative sections (metadata, features, scene, details, feedback) into one prompt. |
| [Prompt Builder B](#prompt-builder-b) | The same builder with technical/style sections (quality, style, mood, motion, general). |
| [Prompt Store](#prompt-store) | A stateful Prompt Builder — sections persist server-side across runs and can be merged or appended. |
| [Prompt Store B](#prompt-store-b) | The stateful counterpart of Prompt Builder B; can share a store name with Prompt Store. |
| [Prompt Store Custom](#prompt-store-custom) | A Prompt Store whose five section headings come from a wired-in headings list instead of being hardcoded. |
| [Prompt Store Headings](#prompt-store-headings) | Five single-line entries emitting the headings list for Prompt Store Custom. |
| [Prompt Store Get](#prompt-store-get) | Fetches one stored category from a named store, with a found flag. |
| [Prompt Store Clear](#prompt-store-clear) | Fully empties a named prompt store. |
| [Prompt Store List](#prompt-store-list) | Lists every active prompt store and a preview of its contents. |
| [Compound Prompt](#compound-prompt) | A 4-way mode dropdown that emits three exclusive booleans to switch prompt-composition branches. |
| [String Concatenate](#string-concatenate) | Joins up to 4 strings with a delimiter, skipping empties. |
| [String Merge Unique](#string-merge-unique) | Merges comma-separated lists with duplicates removed. |
| [String Replacer](#string-replacer) | Applies multiple find→replace rules in one pass and counts the replacements. |
| [String Extractor](#string-extractor) | Splits text into before/middle/after using two bookend substrings. |
| [String List Slicer](#string-list-slicer) | Returns the item at an index from a delimited list (negative indices supported). |
| [String Splitter](#string-splitter) | Splits a string by a delimiter into up to 5 outputs. |

---

## Prompt building

### Prompt Builder

**Assembles a prompt from five labeled multiline sections — metadata,
features, scene, details, feedback — joining the non-empty ones with a
delimiter.**

A stateless authoring helper for structured, narrative-oriented prompts.
Empty sections are skipped cleanly (no dangling delimiters); output order is
fixed: metadata, features, scene, details, feedback.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delimiter` | STRING | `\n\n` | Separator between sections (`\n`/`\t` escapes are honored). |
| `prefix_sections` | BOOLEAN | false | Prefix each section with its name, e.g. `scene: …`. |
| `metadata` | STRING | `""` | Tablesetting & summary. |
| `features` | STRING | `""` | Cast / subjects. |
| `scene` | STRING | `""` | Camera, themes, setting. |
| `details` | STRING | `""` | Finer movement, actions, events. |
| `feedback` | STRING | `""` | Comment / note-style flavor. |

**Outputs:** `prompt`.

### Prompt Builder B

**The same builder with an alternate, technical/style-oriented section set:
quality, style, mood, motion, general.**

Use it alongside [Prompt Builder](#prompt-builder) to author a different facet
of a prompt. Mechanics are identical; only the section names differ.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delimiter` | STRING | `\n\n` | Separator between sections. |
| `prefix_sections` | BOOLEAN | false | Prefix each section with its name. |
| `quality` | STRING | `""` | Technical quality descriptors. |
| `style` | STRING | `""` | Artistic style. |
| `mood` | STRING | `""` | Emotional tone. |
| `motion` | STRING | `""` | Movement descriptors. |
| `general` | STRING | `""` | Catch-all. |

**Outputs:** `prompt`.

---

## Prompt stores

The store nodes keep prompt sections in a named, **in-memory, server-side
store** — no files are written. Stored values persist across runs and across
different workflows, and are wiped when the ComfyUI server restarts. Because
[Prompt Store](#prompt-store) and [Prompt Store B](#prompt-store-b) use
different section keys, they can safely share the same `store_name`, letting
one store hold both narrative and technical facets of a prompt.

!!! note "Caching caveat"
    These stateful nodes have no `IS_CHANGED` override, so ComfyUI may cache
    and skip them when their inputs are unchanged. Change an input (or use the
    `trigger` inputs on Get/Clear/List) to force re-execution and control
    ordering. The exception is [Prompt Store Get](#prompt-store-get), which
    overrides `IS_CHANGED` to always re-execute so it never serves stale data.

### Prompt Store

**A stateful Prompt Builder: the five narrative sections persist in a named
server-side store across runs, and new input can override, merge into, or
append to what's stored.**

Accumulate a prompt across executions — iteratively adding details — without
re-typing everything. Empty inputs preserve the stored value for that section.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_name` | STRING | `default` | Which named store to read/write. Different names are independent. |
| `delimiter` | STRING | `\n\n` | Joins sections in the `prompt` output. |
| `prefix_sections` | BOOLEAN | false | Prefix sections with their names. |
| `clear` | BOOLEAN | false | Reset this node's five section keys before applying new input (Prompt Store B keys in the same store are untouched). |
| `input_mode` | choice | `override` | `override` replaces the stored value; `merge` adds only parts not already present (split on `separator`, deduplicated); `append` concatenates with `separator`. |
| `separator` | STRING | `", "` | Separator used by merge/append. |
| `metadata`…`feedback` | STRING | `""` | New section text (empty = keep stored). |
| `debug` | BOOLEAN | false | Print clear/update messages. |

**Outputs:** `prompt` (all stored sections joined), plus each stored section
individually: `metadata`, `features`, `scene`, `details`, `feedback`.

<!-- TODO: screenshot — Prompt Store accumulating details across two runs with input_mode=merge -->

### Prompt Store B

**The stateful counterpart of Prompt Builder B — same store mechanics, with
the quality/style/mood/motion/general section set.**

Identical behavior to [Prompt Store](#prompt-store); its `clear` only resets
its own five keys, so it coexists with Prompt Store in the same `store_name`.

**Outputs:** `prompt`, plus `quality`, `style`, `mood`, `motion`, `general`.

### Prompt Store Custom

**A Prompt Store with user-defined section headings: wire a
[Prompt Store Headings](#prompt-store-headings) node into `headings` to name
the five sections yourself.**

Store mechanics are identical to [Prompt Store](#prompt-store). Each heading
becomes that section's store key — lowercased and whitespace-trimmed, so
[Prompt Store Get](#prompt-store-get) finds it by name — while
`prefix_sections` uses the heading with its original casing. Empty headings
(or an unwired `headings` input) fall back to generic `section_N` keys.
When a headings node is connected, the editor relabels the five text boxes
and section outputs live; that relabeling is cosmetic — with the headings
node bypassed or muted, execution falls back to the generic keys even if
labels still show custom text.

A few consequences of headings being store keys: duplicate headings collapse
into one shared key (emitted once in `prompt`); changing headings between
runs leaves the old keys' data orphaned in the store (use `clear` or
[Prompt Store Clear](#prompt-store-clear)); and a heading that matches a
Prompt Store / Prompt Store B key in the same `store_name` reads and writes
that same section — useful deliberately, surprising by accident.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_name` | STRING | `default` | Which named store to read/write. |
| `delimiter` | STRING | `\n\n` | Joins sections in the `prompt` output. |
| `prefix_sections` | BOOLEAN | false | Prefix sections with their headings. |
| `clear` | BOOLEAN | false | Reset this node's five effective keys before applying new input. |
| `input_mode` | choice | `override` | `override` / `merge` / `append`, as in Prompt Store. |
| `separator` | STRING | `", "` | Separator used by merge/append. |
| `section_1`–`section_5` | STRING | `""` | New section text (empty = keep stored). |
| `headings` | CCN_PROMPT_HEADINGS | *(optional)* | Wire from Prompt Store Headings to name the sections. |
| `debug` | BOOLEAN | false | Print clear/update messages with the resolved keys. |

**Outputs:** `prompt`, plus each stored section as `section_1`–`section_5`
(relabeled to the headings in the editor when connected).

### Prompt Store Headings

**Five single-line text entries bundled into the headings list consumed by
Prompt Store Custom.**

Each entry names the corresponding section; empty entries leave that section
on its generic `section_N` key. Editing a heading updates the connected store
node's labels immediately.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `heading_1`–`heading_5` | STRING | `""` | Heading for the corresponding section. |

**Outputs:** `headings` (CCN_PROMPT_HEADINGS).

### Prompt Store Get

**Fetches a single stored category from a named store, returning its text and
a `found` flag.**

The read-only counterpart to the store nodes: look up one category (section
key such as `scene` or `mood`) by name and branch on whether data exists. The
lookup never creates a store or key as a side effect, and the category name is
matched case-insensitively with surrounding whitespace ignored. `found` is
false when the store or category doesn't exist *or* when the category holds an
empty string (e.g. after a store node's `clear` toggle); in both cases `value`
is `""`. Unlike the rest of the family, this node always re-executes, so it
always reflects the store's current contents.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_name` | STRING | `default` | The store to read from. |
| `category` | STRING | `""` | The category (section key) to fetch, e.g. `metadata`, `style`. |
| `trigger` | any | *(optional)* | Execution-order dependency (value unused) — wire a store node's output here to fetch after it writes. |

**Outputs:** `value` (the stored text, or `""`), `found` (true only when the
category holds non-empty text).

### Prompt Store Clear

**Fully empties a named store — every key, from both store node types.**

Unlike the per-node `clear` toggle (which resets only that node's own
sections), this wipes the entire named store. The `trigger` input exists only
to create an execution-order dependency; wire any upstream output into it to
control when the clear happens.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_name` | STRING | `default` | The store to clear. |
| `trigger` | any | *(optional)* | Execution-order dependency (value unused). |

**Outputs:** `store_name` (passed through for chaining).

### Prompt Store List

**Lists every active prompt store with a preview of each section's contents.**

Read-only diagnostic; the report is returned as a string and printed to the
console. Wire `trigger` from a store node's output to make it run after
stores change.

**Outputs:** `info` (the report).

---

## Switching

### Compound Prompt

**A 4-way mode dropdown (`Off`, `Temporal`, `Split`, `Neutral`) that emits
three mutually exclusive booleans.**

A UI convenience so one dropdown can drive three prompt-composition branches —
wire each boolean into the enable input of its branch or switch. At most one
output is true; `Off` yields all false. It does not combine conditionings
itself.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | choice | `Off` | Which output is true. |

**Outputs:** `temporal`, `split`, `neutral` (BOOLEAN each).

---

## Strings

### String Concatenate

**Joins up to 4 strings with an optional delimiter, skipping empty ones.**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delimiter` | STRING | `""` | Joiner. Typing `\n` or `\t` produces a real newline/tab. |
| `string_1`–`string_4` | STRING | `""` | Parts to join. |

**Outputs:** `text`.

### String Merge Unique

**Merges up to 3 comma-separated lists into one, removing duplicates while
preserving first-occurrence order.**

Good for deduplicating tag or prompt lists. Matching is exact and
case-sensitive; items are whitespace-trimmed and rejoined with `", "`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `string_1` | STRING | `""` | First list. |
| `string_2`, `string_3` | STRING | `""` | Optional additional lists. |

**Outputs:** `text`.

### String Replacer

**Applies multiple find→replace substitutions in one pass, driven by a compact
rule string, and reports how many replacements were made.**

Rules take the form `find,replace; find2,replace2; …` — pairs separated by
`;`, find/replace split on the first comma (so replacements may contain
commas). Rules apply sequentially, so earlier rules can affect what later ones
see. With `case_sensitive` off, matching is literal but case-insensitive (the
find text is not a regex).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | STRING | `""` | Source text. |
| `replacements` | STRING | `""` | The rule string. |
| `case_sensitive` | BOOLEAN | true | Literal replace vs. case-insensitive literal replace. |
| `debug` | BOOLEAN | false | Print per-rule counts. |

**Outputs:** `text`, `replacement_count`.

### String Extractor

**Splits a string into three parts using two "bookend" substrings: the text up
to and including the start bookend, the text between them, and the text from
the end bookend onward.**

With `wide_bookends` off (the default) the middle is maximally expanded (first
start occurrence to last end occurrence); on, it is minimally squeezed (last
start to first end). If only one bookend is found, the split degrades
gracefully; if neither is found, the whole input comes out as `middle`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_string` | STRING | `""` | Source text. |
| `start_bookend` | STRING | `""` | The opening marker. |
| `end_bookend` | STRING | `""` | The closing marker. |
| `wide_bookends` | BOOLEAN | false | Expand vs. squeeze the middle (see above). |
| `case_sensitive` | BOOLEAN | true | Whether bookend matching honors case. |

**Outputs:** `before`, `middle` (whitespace-trimmed), `after`.

### String List Slicer

**Splits a delimited string into items and returns the one at a given index,
plus the item count.**

Supports Python-style negative indices (`-1` = last). Out-of-range indices
return `default_value` instead of erroring.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_string` | STRING | `""` | The delimited list. |
| `index` | INT | 0 | Which item; negatives count from the end. |
| `delimiter` | STRING | `,` | Literal split delimiter. |
| `strip_whitespace` | BOOLEAN | true | Trim items and drop empty ones (affects the count). |
| `default_value` | STRING | `""` | Returned when the index is out of range. |

**Outputs:** `item`, `list_length`.

### String Splitter

**Splits a string by a delimiter into up to 5 separate outputs.**

For example, fanning a CSV row into fields. Only the first four delimiters
split; the fifth output holds the entire remainder. Each part is
whitespace-trimmed; unused outputs are empty strings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | STRING | `""` | Source text. |
| `delimiter` | STRING | `,` | Literal split delimiter. |

**Outputs:** `out_1`–`out_5`.
