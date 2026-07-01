from types import MappingProxyType
from typing import Any


def freeze(obj: Any) -> Any:
    """
    Recursively freezes a given object, making it immutable.
    """
    if isinstance(obj, dict):
        return MappingProxyType({key: freeze(value) for key, value in obj.items()})
    elif isinstance(obj, list):
        return tuple(freeze(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(freeze(item) for item in obj)
    else:
        return obj
