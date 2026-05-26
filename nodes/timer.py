"""
Timer pair for measuring workflow execution time.

TimerStart records a timestamp. TimerStop computes elapsed time and outputs
both a formatted string (e.g. "4m32s") and raw seconds. Connect any
downstream dependency to TimerStop's trigger input to ensure it evaluates
after the work you want to measure.
"""

import time


class TimerStart:
    """Record a timestamp at execution time."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("start_time",)
    FUNCTION = "execute"
    CATEGORY = "CCN/utils"

    def execute(self):
        return (time.time(),)


class TimerStop:
    """Compute elapsed time from a TimerStart timestamp.
    
    Connect a downstream node's output to the 'trigger' input to
    ensure this node evaluates after the work you want to measure.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_time": ("FLOAT", {
                    "forceInput": True,
                    "tooltip": "Timestamp from a TimerStart node.",
                }),
            },
            "optional": {
                "trigger": ("*", {
                    "tooltip": (
                        "Connect any output here to force this node to "
                        "evaluate after that node. The value is ignored."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT",)
    RETURN_NAMES = ("formatted", "seconds",)
    FUNCTION = "execute"
    CATEGORY = "CCN/utils"

    @classmethod
    def VALIDATE_INPUTS(cls, start_time=None, trigger=None):
        return True

    def execute(self, start_time, trigger=None):
        elapsed = time.time() - start_time
        formatted = self._format_time(elapsed)
        print(f"[CCN] Timer: {formatted} ({elapsed:.2f}s)")
        return (formatted, elapsed)

    @staticmethod
    def _format_time(seconds):
        """Format seconds into a human-readable and filename-safe string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        
        minutes = int(seconds // 60)
        secs = int(seconds % 60)

        if minutes < 60:
            return f"{minutes}m{secs:02d}s"
        
        hours = int(minutes // 60)
        mins = minutes % 60
        return f"{hours}h{mins:02d}m{secs:02d}s"
