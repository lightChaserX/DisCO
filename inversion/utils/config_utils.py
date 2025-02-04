import os
import sys
import importlib

def load_config(config_path):
    """Dynamically loads a Python config file as a module."""
    config_name = os.path.splitext(os.path.basename(config_path))[0]  # Extract filename without .py
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import config from {config_path}")
    
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[config_name] = config_module
    spec.loader.exec_module(config_module)
    
    return config_module  # Return the loaded module

def get_embedding_dir(paths_config, file_name):
    return f'{paths_config.embedding_base_dir}/{file_name}'

def get_checkpoints_dir(paths_config, file_name):
    return f'{paths_config.checkpoints_dir}/{file_name}.pt'