"""
Property — Named variable store for ComfyUI workflows.
Store and retrieve values by name across the workflow graph.

Usage patterns:
  SETTER: Wire a value in → stores it by name, passes it through.
  GETTER: Leave value disconnected → looks up by name, outputs stored value.
  CHAINING: Wire setter's trigger_out → getter's trigger to guarantee
            execution order (setter runs before getter).

Scope:
  Workflow scope — cleared between workflow executions (queue runs).
  Session scope — persists until ComfyUI restarts.


"""

import logging

logger = logging.getLogger("CCN.Property")

# ---------------------------------------------------------------------------
#  Storage
# ---------------------------------------------------------------------------

_SESSION_STORE = {}
_WORKFLOW_STORE = {}
_LAST_EXECUTION_ID = None


def _get_store(session_scope):
    return _SESSION_STORE if session_scope else _WORKFLOW_STORE


def _auto_clear_workflow_store(prompt_id):
    """Clear workflow store when a new execution begins."""
    global _LAST_EXECUTION_ID
    if prompt_id != _LAST_EXECUTION_ID:
        _WORKFLOW_STORE.clear()
        _LAST_EXECUTION_ID = prompt_id


# Try to register a manual clear endpoint
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.post("/ccn/property/clear")
    async def _clear_endpoint(request):
        """POST /ccn/property/clear?scope=workflow|session|all"""
        scope = request.query.get("scope", "workflow")
        if scope in ("workflow", "all"):
            _WORKFLOW_STORE.clear()
        if scope in ("session", "all"):
            _SESSION_STORE.clear()
        return web.json_response({"cleared": scope})
except Exception:
    pass


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _has_downstream(unique_id, prompt):
    """Check if any node in the prompt references this node's outputs."""
    if not prompt:
        return True  # No prompt data — assume active, not dormant
    uid = str(unique_id)
    for node_id, node_data in prompt.items():
        if node_id == uid:
            continue
        for input_val in node_data.get("inputs", {}).values():
            if isinstance(input_val, list) and len(input_val) >= 1 and str(input_val[0]) == uid:
                return True
    return False


def _describe(value):
    """Short human-readable description of a value."""
    if value is None:
        return "None"

    import torch
    if isinstance(value, torch.Tensor):
        return f"Tensor {list(value.shape)} ({value.dtype})"

    if isinstance(value, dict):
        keys = list(value.keys())[:5]
        suffix = f"... +{len(value) - 5}" if len(value) > 5 else ""
        return f"dict({keys}{suffix})"

    if isinstance(value, (list, tuple)):
        t = type(value).__name__
        if len(value) > 0:
            inner = type(value[0]).__name__
            return f"{t}[{inner}, ...] len={len(value)}"
        return f"{t}[] empty"

    if isinstance(value, str):
        if len(value) > 60:
            return f'"{value[:57]}..."'
        return f'"{value}"'

    if isinstance(value, (int, float, bool)):
        return str(value)

    return f"{type(value).__name__}"


# ---------------------------------------------------------------------------
#  Property (combined set/get)
# ---------------------------------------------------------------------------

class Property:
    """
    Named variable node. Stores and retrieves values by string key.

    SETTER mode (value input connected):
      Stores the value under the given name and passes it through.

    GETTER mode (value input NOT connected):
      Looks up the stored value by name and outputs it.
      Wire the setter's trigger_out → this node's trigger to guarantee
      the setter executes first.

    DORMANT (nothing connected in or out):
      No-ops silently. Safe to park in a workspace.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {
                    "default": "my_var",
                    "tooltip": "Variable name. Setter and getter must use the same name.",
                }),
                "session_scope": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "OFF = cleared each workflow run. "
                               "ON = persists until ComfyUI restarts.",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print store operations to console.",
                }),
            },
            "optional": {
                "value": ("*", {
                    "tooltip": "Connect to STORE the value. "
                               "Leave disconnected to GET the stored value.",
                }),
                "trigger": ("*", {
                    "tooltip": "Execution-order dependency. Wire from a setter's "
                               "trigger_out to ensure it runs before this getter.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("*", "*")
    RETURN_NAMES = ("value", "trigger_out")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "CCN/utils"
    DESCRIPTION = ("Named variable store. Connect value input to SET, "
                   "disconnect to GET. Use trigger for execution ordering.")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if "value" not in kwargs and "trigger" not in kwargs:
            return ""
        if "value" not in kwargs:
            return float("nan")
        return ""

    def execute(self, name, session_scope, debug, **kwargs):
        has_value = "value" in kwargs
        has_trigger = "trigger" in kwargs
        value = kwargs.get("value")
        trigger = kwargs.get("trigger")
        unique_id = kwargs.get("unique_id")
        prompt = kwargs.get("prompt")

        # Dormant — no inputs connected AND no downstream nodes reading outputs
        if not has_value and not has_trigger and not _has_downstream(unique_id, prompt):
            return (None, None)

        store = _get_store(session_scope)
        scope_label = "session" if session_scope else "workflow"

        if has_value:
            # ---- SETTER MODE ----
            store[name] = value
            if debug:
                logger.info(f"Property SET: '{name}' ({scope_label}) = {_describe(value)}")
            return (value, trigger)
        else:
            # ---- GETTER MODE ----
            if name in store:
                stored = store[name]
                if debug:
                    logger.info(f"Property GET: '{name}' ({scope_label}) = {_describe(stored)}")
                return (stored, trigger)
            else:
                # Check the other scope as a fallback
                other_store = _SESSION_STORE if not session_scope else _WORKFLOW_STORE
                if name in other_store:
                    other_label = "session" if not session_scope else "workflow"
                    stored = other_store[name]
                    if debug:
                        logger.warning(
                            f"Property GET: '{name}' not in {scope_label} scope, "
                            f"found in {other_label} scope (fallback)"
                        )
                    return (stored, trigger)

                available = list(store.keys())
                raise ValueError(
                    f"Property '{name}' not found in {scope_label} scope.\n"
                    f"Available keys: {available or '(none)'}\n\n"
                    f"Make sure a Property node with this name has its value "
                    f"input connected and executes before this one.\n"
                    f"Tip: Wire setter's trigger_out → getter's trigger input "
                    f"to guarantee execution order."
                )


# ---------------------------------------------------------------------------
#  PropertyClear
# ---------------------------------------------------------------------------

class PropertyClear:
    """Clears stored properties. Can target a specific key or all keys."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scope": (["workflow", "session", "all"],),
                "key": ("STRING", {
                    "default": "*",
                    "tooltip": "Specific key to clear, or * for all keys in scope.",
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "trigger": ("*", {
                    "tooltip": "Wire any input to force execution ordering.",
                }),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("trigger_out",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "CCN/utils"
    DESCRIPTION = "Clear stored properties by scope and/or key."

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self, scope, key, debug, **kwargs):
        trigger = kwargs.get("trigger")

        if key == "*":
            if scope in ("workflow", "all"):
                count = len(_WORKFLOW_STORE)
                _WORKFLOW_STORE.clear()
                if debug:
                    logger.info(f"PropertyClear: Cleared {count} workflow properties")
            if scope in ("session", "all"):
                count = len(_SESSION_STORE)
                _SESSION_STORE.clear()
                if debug:
                    logger.info(f"PropertyClear: Cleared {count} session properties")
        else:
            if scope in ("workflow", "all"):
                removed = _WORKFLOW_STORE.pop(key, None)
                if debug and removed is not None:
                    logger.info(f"PropertyClear: Removed '{key}' from workflow scope")
            if scope in ("session", "all"):
                removed = _SESSION_STORE.pop(key, None)
                if debug and removed is not None:
                    logger.info(f"PropertyClear: Removed '{key}' from session scope")

        return (trigger,)


# ---------------------------------------------------------------------------
#  PropertyList — debug utility
# ---------------------------------------------------------------------------

class PropertyList:
    """Lists all currently stored property keys and value summaries."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scope": (["workflow", "session", "all"],),
            },
            "optional": {
                "trigger": ("*",),
            },
        }

    RETURN_TYPES = ("STRING", "*")
    RETURN_NAMES = ("listing", "trigger_out")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "CCN/utils"
    DESCRIPTION = "List all stored property names and value types."

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self, scope, **kwargs):
        trigger = kwargs.get("trigger")
        lines = []

        if scope in ("workflow", "all"):
            lines.append("═══ Workflow Scope ═══")
            if _WORKFLOW_STORE:
                for k, v in _WORKFLOW_STORE.items():
                    lines.append(f"  {k}: {_describe(v)}")
            else:
                lines.append("  (empty)")
            lines.append("")

        if scope in ("session", "all"):
            lines.append("═══ Session Scope ═══")
            if _SESSION_STORE:
                for k, v in _SESSION_STORE.items():
                    lines.append(f"  {k}: {_describe(v)}")
            else:
                lines.append("  (empty)")

        result = "\n".join(lines)
        return (result, trigger)


# ---------------------------------------------------------------------------
#  Node mappings
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "CCN_Property": Property,
    "CCN_PropertyClear": PropertyClear,
    "CCN_PropertyList": PropertyList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CCN_Property": "Property (CCN)",
    "CCN_PropertyClear": "Property Clear (CCN)",
    "CCN_PropertyList": "Property List (CCN)",
}
