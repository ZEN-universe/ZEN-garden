"""Constructor for the RetrofittingTechnology elements."""

from zen_garden.elements.model_constructor import ModelConstructor
from zen_garden.elements.retrofitting_technology import (
    RetrofittingTechnology,
)


class RetrofittingTechnologyConstructor(ModelConstructor):
    element_class = RetrofittingTechnology
    # Optional, self-contained type: only build when retrofitting technologies
    # are configured.
    always_construct = False
