from . import wrapper
from .model.element import Element
from .optimization_setup import OptimizationSetup
from .postprocess.comparisons import (
    compare_configs,
    compare_dicts,
    compare_model_values,
)
from .postprocess.results.results import Results
from .runner import run
from .utils import download_example_dataset, get_inheritors

__all__ = [
    "run",
    "Results",
    "download_example_dataset",
    "compare_model_value",
    "compare_configs",
    "compare_model_values",
    "compare_dicts",
    "wrapper",
]


# set the element classes of the EnergySystem class
inheritors = get_inheritors(Element)
OptimizationSetup.dict_element_classes.update(
    {klass.__name__: klass for klass in inheritors}
)
