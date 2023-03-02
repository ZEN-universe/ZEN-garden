from . import model
from . import postprocess
from . import preprocess
from .utils import get_inheritors

from .model.objects.element import Element
from .model.objects.energy_system import EnergySystem

# set the element classes of the EnergySystem class
inheritors = get_inheritors(Element)
EnergySystem.dict_element_classes.update({klass.__name__: klass for klass in inheritors})
