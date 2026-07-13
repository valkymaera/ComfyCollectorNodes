"""
Work-in-progress nodes. This package is excluded from registry publishing
via .comfyignore; the root __init__ only loads it when the folder exists.
"""

from .better_int import BetterInt
from .gated_increment import GatedIncrement
from .property import Property, PropertyClear, PropertyList
from .timer import TimerStart, TimerStop
from .lora_quantize_fp8 import LoraQuantizeFP8
from .latent_loader_filtered import LatentLoaderFiltered

NODE_CLASS_MAPPINGS = {
    "CCN_BetterInt": BetterInt,
    "CCN_GatedIncrement": GatedIncrement,
    "CCN_Property": Property,
    "CCN_PropertyClear": PropertyClear,
    "CCN_PropertyList": PropertyList,
    "CCN_TimerStart": TimerStart,
    "CCN_TimerStop": TimerStop,
    "CCN_LoraQuantizeFP8": LoraQuantizeFP8,
    # Key must match js/wip/latent_loader_filtered.js
    "CCN_LatentLoaderFiltered": LatentLoaderFiltered,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CCN_BetterInt": "Better Int (CCN)",
    "CCN_GatedIncrement": "Gated Increment (CCN)",
    "CCN_Property": "Property (CCN)",
    "CCN_PropertyClear": "Property Clear (CCN)",
    "CCN_PropertyList": "Property List (CCN)",
    "CCN_TimerStart": "Timer Start (CCN)",
    "CCN_TimerStop": "Timer Stop (CCN)",
    "CCN_LoraQuantizeFP8": "LoRA Quantize FP8 (CCN)",
    "CCN_LatentLoaderFiltered": "Latent Loader Filtered (CCN)",
}
