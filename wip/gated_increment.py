"""
GatedIncrement — Integer that changes after every N runs.

Tracks an internal counter that increments each execution. When the
counter reaches the rate threshold, the value is incremented or
randomized (depending on mode), and the counter resets.

The counter persists across ComfyUI restarts via a JSON state file
stored alongside the node code (.ccn_gated_state.json).

Timing:
  before — The changed value is THIS run's output.
  after  — THIS run outputs the old value; the widget updates for next run.

"""

import random
import json
import os
import logging

logger = logging.getLogger("CCN.GatedIncrement")

_INT_MAX = 2**53 - 1
_INT_MIN = -(2**53 - 1)
_RAND_MAX = 0xFFFFFFFF
_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ccn_gated_state.json")


# ---------------------------------------------------------------------------
#  Persistent state helpers
# ---------------------------------------------------------------------------

def _load_all_state():
    try:
        with open(_STATE_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_all_state(state):
    try:
        with open(_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except OSError:
        pass


def _get_node_state(unique_id):
    state = _load_all_state()
    return state.get(str(unique_id), {"counter": 0})


def _set_node_state(unique_id, node_state):
    state = _load_all_state()
    state[str(unique_id)] = node_state
    _save_all_state(state)


# ---------------------------------------------------------------------------
#  Node
# ---------------------------------------------------------------------------

class GatedIncrement:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": _INT_MIN, "max": _INT_MAX}),
                "rate": ("INT", {
                    "default": 5, "min": 1, "max": 1000000,
                    "tooltip": "Number of runs between each value change.",
                }),
                "mode": (["increment", "random"],),
                "timing": (["before", "after"],),
                "step": ("INT", {
                    "default": 1, "min": 1, "max": 1000000,
                    "tooltip": "Amount to add in increment mode.",
                }),
                "reset_counter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reset the run counter to 0. Toggle on, run once, toggle off.",
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("INT",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "CCN/utils"
    DESCRIPTION = ("Integer that increments or randomizes every N runs. "
                   "Counter persists across ComfyUI restarts.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self, value, rate, mode, timing, step, reset_counter, debug, **kwargs):
        unique_id = kwargs.get("unique_id", "unknown")

        # Handle reset
        if reset_counter:
            _set_node_state(unique_id, {"counter": 0})
            if debug:
                logger.info(f"GatedIncrement [{unique_id}]: Counter reset to 0")
            return {"result": (value,), "ui": {"output_value": [value], "counter_display": [f"0 / {rate}"]}}

        # Load and advance counter
        node_state = _get_node_state(unique_id)
        counter = node_state.get("counter", 0) + 1
        should_change = counter >= rate

        if should_change:
            counter = 0

        # Persist
        _set_node_state(unique_id, {"counter": counter})

        # Compute new value if triggered
        if should_change:
            if mode == "increment":
                new_val = value + step
            else:
                new_val = random.randint(0, _RAND_MAX)
        else:
            new_val = None

        # Apply timing
        if timing == "before":
            output = new_val if should_change else value
            next_widget = output
        else:  # after
            output = value
            next_widget = new_val if should_change else value

        if debug:
            logger.info(
                f"GatedIncrement [{unique_id}]: mode={mode} timing={timing} "
                f"counter={counter}/{rate} triggered={should_change} "
                f"input={value} output={output} next_widget={next_widget}"
            )

        return {
            "result": (output,),
            "ui": {
                "output_value": [next_widget],
                "counter_display": [f"{counter} / {rate}"],
            },
        }
