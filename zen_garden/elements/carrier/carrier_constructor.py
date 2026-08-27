"""Constructor for the Carrier elements."""

from zen_garden.elements.carrier import Carrier
from zen_garden.elements.model_constructor import ModelConstructor


class CarrierConstructor(ModelConstructor):
    element_class = Carrier
