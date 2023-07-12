"""==========
:Title:        ZEN-GARDEN
:Created:      March-2022
:Authors:      Alissa Ganter (aganter@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

  function that loads all classes and subclasses of technology directory.
"""
from pathlib import Path

# register the subclasses
modules = Path(__file__).parent.glob("*.py")
__all__ = [f.stem for f in modules if f.is_file()]
