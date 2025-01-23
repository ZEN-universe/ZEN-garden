"""
Function that loads all classes and subclasses of carrier directory.
"""
from pathlib import Path

# register the subclasses
modules = Path(__file__).parent.glob("*.py")
__all__ = [f.stem for f in modules if f.is_file()]
