"""
Python Exec - Execute arbitrary Python code for debugging
"""

import torch


class PythonExec:
    """
    Execute arbitrary Python code for debugging and experimentation.
    
    Available variables in your code:
      - input_1, input_2, input_3: Your connected inputs (or None)
      - torch: PyTorch module
      - result: Set this to return a value
    
    Example code:
      print(f"Input type: {type(input_1)}")
      print(f"Input shape: {input_1.shape}")
      result = input_1 * 2
    
    WARNING: This executes arbitrary code. Use responsibly.
    """
    
    CATEGORY = "ComfyCollectorNodes/Utils"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "code": ("STRING", {"default": "print('Hello!')\nresult = input_1", "multiline": True}),
            },
            "optional": {
                "input_1": ("*",),
                "input_2": ("*",),
                "input_3": ("*",),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("result", "output")
    FUNCTION = "execute"

    def execute(self, code, input_1=None, input_2=None, input_3=None):
        # Build execution context
        local_vars = {
            "input_1": input_1,
            "input_2": input_2,
            "input_3": input_3,
            "torch": torch,
            "result": None,
        }
        
        # Capture print output
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            exec(code, {"__builtins__": __builtins__}, local_vars)
            output = captured_output.getvalue()
        except Exception as e:
            output = f"ERROR: {type(e).__name__}: {e}"
            print(f"[ComfyCollectorNodes] PythonExec error: {e}")
        finally:
            sys.stdout = old_stdout
        
        # Also print to console
        if output:
            print(f"[CCN PythonExec Output]\n{output}")
        
        return (local_vars.get("result"), output)


class InspectTensor:
    """
    Quick inspection of any tensor-like input.
    Prints detailed info to console and returns a summary string.
    """
    
    CATEGORY = "ComfyCollectorNodes/Utils"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "Tensor"}),
            },
            "optional": {
                "data": ("*",),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("passthrough", "info")
    FUNCTION = "inspect"

    def inspect(self, label, data=None):
        lines = [f"=== {label} ==="]
        
        if data is None:
            lines.append("Value: None")
        else:
            lines.append(f"Type: {type(data).__name__}")
            
            # Handle different types
            if hasattr(data, 'shape'):
                lines.append(f"Shape: {list(data.shape)}")
            if hasattr(data, 'dtype'):
                lines.append(f"Dtype: {data.dtype}")
            if hasattr(data, 'device'):
                lines.append(f"Device: {data.device}")
            
            # Tensor stats
            if isinstance(data, torch.Tensor):
                lines.append(f"Min: {data.min().item():.6f}")
                lines.append(f"Max: {data.max().item():.6f}")
                lines.append(f"Mean: {data.mean().item():.6f}")
                lines.append(f"Std: {data.std().item():.6f}")
            
            # Dict inspection
            elif isinstance(data, dict):
                lines.append(f"Keys: {list(data.keys())}")
                for k, v in data.items():
                    v_info = f"type={type(v).__name__}"
                    if hasattr(v, 'shape'):
                        v_info += f", shape={list(v.shape)}"
                    lines.append(f"  {k}: {v_info}")
            
            # List/tuple inspection
            elif isinstance(data, (list, tuple)):
                lines.append(f"Length: {len(data)}")
                if len(data) > 0:
                    lines.append(f"First item type: {type(data[0]).__name__}")
                    if len(data) <= 5:
                        for i, item in enumerate(data):
                            item_info = f"type={type(item).__name__}"
                            if hasattr(item, 'shape'):
                                item_info += f", shape={list(item.shape)}"
                            lines.append(f"  [{i}]: {item_info}")
            
            # String
            elif isinstance(data, str):
                lines.append(f"Length: {len(data)}")
                lines.append(f"Preview: {data[:100]}{'...' if len(data) > 100 else ''}")
            
            # Other - try to repr
            else:
                try:
                    lines.append(f"Repr: {repr(data)[:200]}")
                except:
                    lines.append("(Could not repr)")
        
        info = "\n".join(lines)
        print(f"\n[CCN Inspect]\n{info}\n")
        
        return (data, info)
