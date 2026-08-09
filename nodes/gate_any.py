"""
Gate Any — Pass any value through or block downstream execution with a boolean.
"""

from comfy_execution.graph_utils import ExecutionBlocker


class AnyType(str):
    """A type string that never fails an equality check against another type."""

    def __ne__(self, other):
        return False


_ANY = AnyType("*")


class GateAny:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (_ANY, {"lazy": True, "tooltip": "Any value to pass through when the gate is enabled."}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "When off, downstream nodes are silently skipped."}),
                "skip_upstream": ("BOOLEAN", {"default": True, "tooltip": "When the gate is disabled, also skip executing the branch feeding 'value'."}),
            },
        }

    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "gate"
    CATEGORY = "ComfyCollectorNodes/Utils"
    DESCRIPTION = (
        "Pass any value through when enabled; when disabled, downstream nodes are "
        "silently skipped. With skip_upstream on, the branch feeding 'value' is "
        "never executed while the gate is closed."
    )

    def check_lazy_status(self, enabled, skip_upstream, value=None):
        if enabled or not skip_upstream:
            return ["value"]
        return []

    def gate(self, value, enabled, skip_upstream):
        if enabled:
            return (value,)
        return (ExecutionBlocker(None),)
