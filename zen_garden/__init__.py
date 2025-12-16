from . import model
from . import postprocess
from . import preprocess
from .utils import get_inheritors
from .model.element import Element
from .optimization_setup import OptimizationSetup
from .runner import run
from .postprocess.results.results import Results
from .utils import download_example_dataset
from .postprocess.comparisons import compare_model_values, compare_configs, compare_dicts

__all__ = ["run", "Results", "download_example_dataset", "compare_model_value", "compare_configs", "compare_dicts"]


# set the element classes of the EnergySystem class
inheritors = get_inheritors(Element)
OptimizationSetup.dict_element_classes.update({klass.__name__: klass for klass in inheritors})
