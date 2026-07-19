"""
ComfyCollectorNodes - A collection of utility nodes for ComfyUI
"""

import os

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# WIP nodes live in wip/, which is .comfyignore'd out of the published
# package; merge them only when the folder is present (local dev).
if os.path.isdir(os.path.join(os.path.dirname(__file__), "wip")):
    from .wip import NODE_CLASS_MAPPINGS as _WIP_NODES, NODE_DISPLAY_NAME_MAPPINGS as _WIP_NAMES
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_WIP_NODES}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_WIP_NAMES}

# Local-only nodes live in local/, which is gitignored and .comfyignore'd;
# merge them only when the folder is present (this machine only).
if os.path.isdir(os.path.join(os.path.dirname(__file__), "local")):
    from .local import NODE_CLASS_MAPPINGS as _LOCAL_NODES, NODE_DISPLAY_NAME_MAPPINGS as _LOCAL_NAMES
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_LOCAL_NODES}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_LOCAL_NAMES}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

__version__ = "1.0.0"
