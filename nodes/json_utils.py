"""
JSON file utilities
"""

import os
import json
import folder_paths


class LoadJSONFile:
    """
    Load a JSON file and output as string.
    
    Useful for storing template workflows or configuration
    that you want to embed in outputs instead of the actual workflow.
    """
    
    CATEGORY = "ComfyCollectorNodes/Utils"
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        
        # Scan for .json files in input directory
        for root, dirs, filenames in os.walk(input_dir):
            for f in filenames:
                if f.endswith('.json'):
                    rel_path = os.path.relpath(os.path.join(root, f), input_dir)
                    files.append(rel_path)
        
        files = sorted(files) if files else ["No JSON files found"]
        
        return {
            "required": {
                "json_file": (files, {"default": files[0] if files else ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("json_string", "filename")
    FUNCTION = "load_json"

    def load_json(self, json_file):
        if json_file == "No JSON files found":
            print("[ComfyCollectorNodes] LoadJSONFile: No file selected")
            return ("", "")
        
        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, json_file)
        
        if not os.path.exists(file_path):
            print(f"[ComfyCollectorNodes] LoadJSONFile: File not found: {file_path}")
            return ("", "")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Validate it's actually valid JSON
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[ComfyCollectorNodes] LoadJSONFile: Invalid JSON in {json_file}: {e}")
            return ("", json_file)
        
        print(f"[ComfyCollectorNodes] LoadJSONFile: Loaded {json_file}")
        return (content, json_file)
    
    @classmethod
    def IS_CHANGED(cls, json_file):
        if json_file == "No JSON files found":
            return ""
        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, json_file)
        if os.path.exists(file_path):
            return os.path.getmtime(file_path)
        return ""


class LoadJSONFilePath:
    """
    Load a JSON file from an arbitrary path.
    """
    
    CATEGORY = "ComfyCollectorNodes/Utils"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("json_string", "filename")
    FUNCTION = "load_json"

    def load_json(self, file_path):
        if not file_path or not file_path.strip():
            print("[ComfyCollectorNodes] LoadJSONFilePath: No path provided")
            return ("", "")
        
        file_path = file_path.strip()
        
        if not os.path.exists(file_path):
            print(f"[ComfyCollectorNodes] LoadJSONFilePath: File not found: {file_path}")
            return ("", "")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[ComfyCollectorNodes] LoadJSONFilePath: Invalid JSON: {e}")
            return ("", os.path.basename(file_path))
        
        print(f"[ComfyCollectorNodes] LoadJSONFilePath: Loaded {file_path}")
        return (content, os.path.basename(file_path))
    
    @classmethod
    def IS_CHANGED(cls, file_path):
        if file_path and os.path.exists(file_path.strip()):
            return os.path.getmtime(file_path.strip())
        return ""
