from typing import Dict, Type, Callable
import torch.nn as nn

# A simple registry for models
ModelRegistry: Dict[str, Type[nn.Module] | Callable] = {}

def register_model(name: str):
    """A decorator to register a new model in the registry."""
    def decorator(cls: Type[nn.Module]):
        ModelRegistry[name.lower()] = cls
        return cls
    return decorator

def get_model(name: str) -> Type[nn.Module] | Callable:
    """Gets a model from the registry."""
    name = name.lower()
    if name not in ModelRegistry:
        raise ValueError(f"Model '{name}' not found in registry. Available models: {list(ModelRegistry.keys())}")
    return ModelRegistry[name]
