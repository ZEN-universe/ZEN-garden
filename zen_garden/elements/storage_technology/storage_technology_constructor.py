"""Constructor for the StorageTechnology elements."""

from zen_garden.elements.model_constructor import ModelConstructor
from zen_garden.elements.storage_technology import StorageTechnology


class StorageTechnologyConstructor(ModelConstructor):
    element_class = StorageTechnology
