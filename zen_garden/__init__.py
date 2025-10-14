from . import model
from . import postprocess
from . import preprocess
from .utils import get_inheritors

from .model.element import Element
from .optimization_setup import OptimizationSetup

# set the element classes of the EnergySystem class
inheritors = get_inheritors(Element)
OptimizationSetup.dict_element_classes.update({klass.__name__: klass for klass in inheritors})
