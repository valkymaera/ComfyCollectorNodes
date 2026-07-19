"""
Prompt Store - Persistent prompt sections with session memory
"""

# Module-level storage persists during ComfyUI session
_prompt_stores = {}

INPUT_MODES = ["override", "merge", "append"]


def _get_store(store_name):
    """Get or create a store, handling dynamic keys."""
    global _prompt_stores
    if store_name not in _prompt_stores:
        _prompt_stores[store_name] = {}
    return _prompt_stores[store_name]


def _merge_unique(existing, new, separator=","):
    """Merge two strings, keeping only unique separator-delimited values."""
    if not existing:
        return new
    if not new:
        return existing
    
    existing_parts = [p.strip() for p in existing.split(separator) if p.strip()]
    new_parts = [p.strip() for p in new.split(separator) if p.strip()]
    
    # Add new parts that don't already exist
    seen = set(existing_parts)
    for part in new_parts:
        if part not in seen:
            existing_parts.append(part)
            seen.add(part)
    
    return f"{separator} ".join(existing_parts)


def _apply_input_mode(existing, new, mode, separator):
    """Apply the input mode to combine existing and new values."""
    if not new or not new.strip():
        return existing  # Empty input = keep existing
    
    new = new.strip()
    
    if mode == "override":
        return new
    elif mode == "merge":
        return _merge_unique(existing, new, separator)
    elif mode == "append":
        if existing:
            return f"{existing}{separator}{new}"
        return new
    
    return new  # Fallback to override


_GENERIC_KEYS = ["section_1", "section_2", "section_3", "section_4", "section_5"]


def _resolve_heading_keys(headings):
    """Resolve a headings list into (store_keys, display_names).

    Store keys are lowercased so PromptStoreGet's case-insensitive lookup
    finds them; display names keep the user's casing. Empty or missing
    headings fall back to the generic section_N names.
    """
    keys = []
    names = []
    for i in range(5):
        heading = ""
        if isinstance(headings, (list, tuple)) and i < len(headings):
            heading = str(headings[i]).strip()
        keys.append(heading.lower() if heading else _GENERIC_KEYS[i])
        names.append(heading if heading else _GENERIC_KEYS[i])
    return keys, names


class PromptStore:
    """
    Stores and retrieves prompt sections with session memory.
    
    Input modes:
      - override: Replace stored value with new input
      - merge: Combine unique delimiter-separated values (no duplicates)
      - append: Add new input to end of existing value
    
    Empty inputs = keep previous value (don't change)
    
    Use different store_name values for different prompts.
    Use clear=True to reset all sections for a store.
    
    Sections:
      - Metadata: tablesetting & summary
      - Features: cast/subjects
      - Scene: camera, themes, setting
      - Details: finer movement, actions, events
      - Feedback: comment/note-style flavor
    """
    
    CATEGORY = "ComfyCollectorNodes/Prompt"
    SECTION_KEYS = ["metadata", "features", "scene", "details", "feedback"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "store_name": ("STRING", {"default": "default"}),
                "delimiter": ("STRING", {"default": "\\n\\n"}),
                "prefix_sections": ("BOOLEAN", {"default": False}),
                "clear": ("BOOLEAN", {"default": False}),
                "input_mode": (INPUT_MODES, {"default": "override"}),
                "separator": ("STRING", {"default": ", "}),
            },
            "optional": {
                "metadata": ("STRING", {"default": "", "multiline": True}),
                "features": ("STRING", {"default": "", "multiline": True}),
                "scene": ("STRING", {"default": "", "multiline": True}),
                "details": ("STRING", {"default": "", "multiline": True}),
                "feedback": ("STRING", {"default": "", "multiline": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "metadata", "features", "scene", "details", "feedback")
    FUNCTION = "process_prompt"

    def process_prompt(self, store_name, delimiter, prefix_sections, clear, input_mode, separator,
                       metadata="", features="", scene="", details="", feedback="", debug=False):
        # Handle escaped characters
        delimiter = delimiter.replace("\\n", "\n").replace("\\t", "\t")
        
        store = _get_store(store_name)
        
        # Clear only this node's keys if requested
        if clear:
            for key in self.SECTION_KEYS:
                store[key] = ""
            if debug:
                print(f"[ComfyCollectorNodes] Prompt store '{store_name}' cleared (PromptStore keys)")
        
        # Update sections using input mode
        sections = {
            "metadata": metadata,
            "features": features,
            "scene": scene,
            "details": details,
            "feedback": feedback,
        }
        
        updated = []
        for key, value in sections.items():
            if value and value.strip():
                existing = store.get(key, "")
                store[key] = _apply_input_mode(existing, value, input_mode, separator)
                updated.append(key)
        
        if updated and debug:
            print(f"[ComfyCollectorNodes] Prompt store '{store_name}' updated ({input_mode}): {', '.join(updated)}")
        
        # Build final prompt from stored values
        parts = []
        for key in self.SECTION_KEYS:
            val = store.get(key, "")
            if val:
                if prefix_sections:
                    parts.append(f"{key}: {val}")
                else:
                    parts.append(val)
        
        result = delimiter.join(parts)
        
        return (
            result,
            store.get("metadata", ""),
            store.get("features", ""),
            store.get("scene", ""),
            store.get("details", ""),
            store.get("feedback", ""),
        )


class PromptStoreB:
    """
    Stores and retrieves prompt sections with session memory (alternate labels).
    
    Input modes:
      - override: Replace stored value with new input
      - merge: Combine unique delimiter-separated values (no duplicates)
      - append: Add new input to end of existing value
    
    Empty inputs = keep previous value (don't change)
    
    Can share a store_name with PromptStore - keys don't collide.
    
    Sections:
      - Quality: technical quality descriptors
      - Style: artistic style
      - Mood: emotional tone
      - Motion: movement descriptors
      - General: catch-all
    """
    
    CATEGORY = "ComfyCollectorNodes/Prompt"
    SECTION_KEYS = ["quality", "style", "mood", "motion", "general"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "store_name": ("STRING", {"default": "default"}),
                "delimiter": ("STRING", {"default": "\\n\\n"}),
                "prefix_sections": ("BOOLEAN", {"default": False}),
                "clear": ("BOOLEAN", {"default": False}),
                "input_mode": (INPUT_MODES, {"default": "override"}),
                "separator": ("STRING", {"default": ", "}),
            },
            "optional": {
                "quality": ("STRING", {"default": "", "multiline": True}),
                "style": ("STRING", {"default": "", "multiline": True}),
                "mood": ("STRING", {"default": "", "multiline": True}),
                "motion": ("STRING", {"default": "", "multiline": True}),
                "general": ("STRING", {"default": "", "multiline": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "quality", "style", "mood", "motion", "general")
    FUNCTION = "process_prompt"

    def process_prompt(self, store_name, delimiter, prefix_sections, clear, input_mode, separator,
                       quality="", style="", mood="", motion="", general="", debug=False):
        # Handle escaped characters
        delimiter = delimiter.replace("\\n", "\n").replace("\\t", "\t")
        
        store = _get_store(store_name)
        
        # Clear only this node's keys if requested
        if clear:
            for key in self.SECTION_KEYS:
                store[key] = ""
            if debug:
                print(f"[ComfyCollectorNodes] Prompt store '{store_name}' cleared (PromptStoreB keys)")
        
        # Update sections using input mode
        sections = {
            "quality": quality,
            "style": style,
            "mood": mood,
            "motion": motion,
            "general": general,
        }
        
        updated = []
        for key, value in sections.items():
            if value and value.strip():
                existing = store.get(key, "")
                store[key] = _apply_input_mode(existing, value, input_mode, separator)
                updated.append(key)
        
        if updated and debug:
            print(f"[ComfyCollectorNodes] Prompt store '{store_name}' updated ({input_mode}): {', '.join(updated)}")
        
        # Build final prompt from stored values
        parts = []
        for key in self.SECTION_KEYS:
            val = store.get(key, "")
            if val:
                if prefix_sections:
                    parts.append(f"{key}: {val}")
                else:
                    parts.append(val)
        
        result = delimiter.join(parts)
        
        return (
            result,
            store.get("quality", ""),
            store.get("style", ""),
            store.get("mood", ""),
            store.get("motion", ""),
            store.get("general", ""),
        )


class PromptStoreCustom:
    """
    Stores and retrieves prompt sections with session memory (custom headings).

    Works like PromptStore, but the five section headings come from a
    CCN_PROMPT_HEADINGS input (wire from PromptStoreHeadings). Headings
    name the store keys and section prefixes; empty or unwired headings
    fall back to generic section_N keys.

    Input modes:
      - override: Replace stored value with new input
      - merge: Combine unique delimiter-separated values (no duplicates)
      - append: Add new input to end of existing value

    Empty inputs = keep previous value (don't change)
    """

    CATEGORY = "ComfyCollectorNodes/Prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "store_name": ("STRING", {"default": "default"}),
                "delimiter": ("STRING", {"default": "\\n\\n"}),
                "prefix_sections": ("BOOLEAN", {"default": False}),
                "clear": ("BOOLEAN", {"default": False}),
                "input_mode": (INPUT_MODES, {"default": "override"}),
                "separator": ("STRING", {"default": ", "}),
            },
            "optional": {
                "section_1": ("STRING", {"default": "", "multiline": True}),
                "section_2": ("STRING", {"default": "", "multiline": True}),
                "section_3": ("STRING", {"default": "", "multiline": True}),
                "section_4": ("STRING", {"default": "", "multiline": True}),
                "section_5": ("STRING", {"default": "", "multiline": True}),
                "headings": ("CCN_PROMPT_HEADINGS", {"tooltip": "Wire from Prompt Store Headings to name the five sections."}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "section_1", "section_2", "section_3", "section_4", "section_5")
    FUNCTION = "process_prompt"

    def process_prompt(self, store_name, delimiter, prefix_sections, clear, input_mode, separator,
                       section_1="", section_2="", section_3="", section_4="", section_5="",
                       headings=None, debug=False):
        # Handle escaped characters
        delimiter = delimiter.replace("\\n", "\n").replace("\\t", "\t")

        store = _get_store(store_name)
        keys, names = _resolve_heading_keys(headings)

        # Clear only this node's effective keys if requested
        if clear:
            for key in keys:
                store[key] = ""
            if debug:
                print(f"[ComfyCollectorNodes] Prompt store '{store_name}' cleared (PromptStoreCustom keys: {', '.join(keys)})")

        # Update sections using input mode
        values = [section_1, section_2, section_3, section_4, section_5]

        updated = []
        for key, value in zip(keys, values):
            if value and value.strip():
                existing = store.get(key, "")
                store[key] = _apply_input_mode(existing, value, input_mode, separator)
                updated.append(key)

        if updated and debug:
            print(f"[ComfyCollectorNodes] Prompt store '{store_name}' updated ({input_mode}): {', '.join(updated)}")

        # Build final prompt from stored values; duplicate headings share a
        # store key, so emit each key only once
        parts = []
        seen = set()
        for key, name in zip(keys, names):
            if key in seen:
                continue
            seen.add(key)
            val = store.get(key, "")
            if val:
                if prefix_sections:
                    parts.append(f"{name}: {val}")
                else:
                    parts.append(val)

        result = delimiter.join(parts)

        return (
            result,
            store.get(keys[0], ""),
            store.get(keys[1], ""),
            store.get(keys[2], ""),
            store.get(keys[3], ""),
            store.get(keys[4], ""),
        )


class PromptStoreHeadings:
    """
    Defines the five section headings for PromptStoreCustom.

    Empty entries fall back to the generic section_N keys on the store node.
    """

    CATEGORY = "ComfyCollectorNodes/Prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "heading_1": ("STRING", {"default": ""}),
                "heading_2": ("STRING", {"default": ""}),
                "heading_3": ("STRING", {"default": ""}),
                "heading_4": ("STRING", {"default": ""}),
                "heading_5": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("CCN_PROMPT_HEADINGS",)
    RETURN_NAMES = ("headings",)
    FUNCTION = "build"

    def build(self, heading_1, heading_2, heading_3, heading_4, heading_5):
        return ([heading_1, heading_2, heading_3, heading_4, heading_5],)


class PromptStoreGet:
    """
    Fetches a single stored category from a named prompt store.

    Read-only: never creates a store or key as a side effect.
    found is True only when the category holds non-empty text.
    """

    CATEGORY = "ComfyCollectorNodes/Prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "store_name": ("STRING", {"default": "default"}),
                "category": ("STRING", {"default": ""}),
            },
            "optional": {
                "trigger": ("*",),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("value", "found")
    FUNCTION = "get_value"

    @classmethod
    def IS_CHANGED(cls, store_name, category, trigger=None):
        # Always re-execute: the store mutates without this node's inputs changing
        return float("NaN")

    def get_value(self, store_name, category, trigger=None):
        store = _prompt_stores.get(store_name, {})
        value = store.get(category.strip().lower(), "")
        return (value, bool(value))


class PromptStoreClear:
    """
    Clears a prompt store (all keys). 
    Useful as a utility node to reset without needing to toggle the clear bool.
    """
    
    CATEGORY = "ComfyCollectorNodes/Prompt"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "store_name": ("STRING", {"default": "default"}),
            },
            "optional": {
                "trigger": ("*",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("store_name",)
    FUNCTION = "clear_store"

    def clear_store(self, store_name, trigger=None):
        global _prompt_stores
        
        if store_name in _prompt_stores:
            _prompt_stores[store_name] = {}
            print(f"[ComfyCollectorNodes] Prompt store '{store_name}' fully cleared")
        else:
            print(f"[ComfyCollectorNodes] Prompt store '{store_name}' not found (nothing to clear)")
        
        return (store_name,)


class PromptStoreList:
    """
    Lists all active prompt stores and their contents.
    Useful for debugging.
    """
    
    CATEGORY = "ComfyCollectorNodes/Prompt"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "trigger": ("*",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "list_stores"

    def list_stores(self, trigger=None):
        global _prompt_stores
        
        if not _prompt_stores:
            info = "No prompt stores active."
            print(f"[ComfyCollectorNodes] {info}")
            return (info,)
        
        lines = []
        for name, store in _prompt_stores.items():
            lines.append(f"=== Store: {name} ===")
            if not store:
                lines.append("  (empty)")
            else:
                for key, value in store.items():
                    preview = value[:50] + "..." if len(value) > 50 else value
                    preview = preview.replace("\n", " ")
                    lines.append(f"  {key}: {preview if preview else '(empty)'}")
            lines.append("")
        
        info = "\n".join(lines)
        print(f"[ComfyCollectorNodes] Prompt Stores:\n{info}")
        
        return (info,)
