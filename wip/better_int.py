"""
BetterInt — Integer node with before/after increment and randomize.

Unlike ComfyUI's built-in randomize (which only fires after execution),
this node supports incrementing or randomizing BEFORE the current run,
so the output reflects the changed value immediately.

The widget value updates visually via the JS extension (js/ccn_int_nodes.js),
allowing you to pause a batch, see the current value, and resume.

Modes:
  fixed              — Output matches widget. No changes between runs.
  increment_before   — Increment, then output the new value this run.
  increment_after    — Output current value, then increment for next run.
  random_before      — Randomize, then output the new value this run.
  random_after       — Output current value, then randomize for next run.


"""

import random
import logging

logger = logging.getLogger("CCN.BetterInt")

_INT_MAX = 2**53 - 1
_INT_MIN = -(2**53 - 1)
_RAND_MAX = 0xFFFFFFFF


class BetterInt:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": _INT_MIN, "max": _INT_MAX}),
                "mode": (["fixed", "increment_before", "increment_after",
                          "random_before", "random_after"],),
                "step": ("INT", {"default": 1, "min": 1, "max": 1000000}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("INT",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "CCN/utils"
    DESCRIPTION = ("Integer with before/after increment and randomize modes. "
                   "Widget updates between runs for full control.")

    @classmethod
    def IS_CHANGED(cls, mode, **kwargs):
        if mode != "fixed":
            return float("nan")
        return ""

    def execute(self, value, mode, step, debug):
        if mode == "fixed":
            output = value
            next_widget = value
        elif mode == "increment_before":
            output = value + step
            next_widget = output
        elif mode == "increment_after":
            output = value
            next_widget = value + step
        elif mode == "random_before":
            output = random.randint(0, _RAND_MAX)
            next_widget = output
        elif mode == "random_after":
            output = value
            next_widget = random.randint(0, _RAND_MAX)
        else:
            output = value
            next_widget = value

        if debug:
            logger.info(
                f"BetterInt: mode={mode} input={value} "
                f"output={output} next_widget={next_widget}"
            )

        return {"result": (output,), "ui": {"output_value": [next_widget]}}
