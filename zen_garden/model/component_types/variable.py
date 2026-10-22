from abc import ABC
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from zen_garden.model.component_types.mask import GenericMask
    from zen_garden.model.constructor import ModelConstructor


class GenericVariable(ABC):
    """Abstract base class for variables in ZEN-garden."""

    name: ClassVar[str]
    indices: ClassVar[list[str]]
    doc: ClassVar[str]
    unit_category: ClassVar[dict[str, int]]
    integer: ClassVar[bool] = False
    binary: ClassVar[bool] = False
    masks: ClassVar[list[type["GenericMask"]]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        required = ("name", "indices", "doc", "unit_category")

        for attr in required:
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__name__} must define {attr!r}")
        inherited_masks = []
        for base in cls.__bases__:
            inherited_masks.extend(getattr(base, "masks", ()))
        own_masks = cls.__dict__.get("masks", ())
        cls.masks = list(dict.fromkeys([*inherited_masks, *own_masks]))

    @classmethod
    def build(cls, model_constructor: "ModelConstructor") -> None:
        """Construct and register this variable on the optimization model."""
        if not cls.should_construct(model_constructor):
            return
        index_sets = cls.get_index_sets(model_constructor)
        declared_mask = model_constructor.optimization_model.masks.combine_for_indices(
            cls.masks, model_constructor, index_sets
        )
        custom_mask = cls.get_mask(model_constructor, index_sets)
        if declared_mask is None:
            mask = custom_mask
        elif custom_mask is None:
            mask = declared_mask
        else:
            custom_mask = custom_mask.reindex_like(declared_mask, fill_value=False)
            mask = declared_mask & custom_mask
        if mask is not None:
            model_constructor.optimization_model.masks.log_reduction(cls.name, mask)
        model_constructor.optimization_model.add_variable(
            name=cls.name,
            index_sets=index_sets,
            integer=cls.integer,
            binary=cls.binary,
            bounds=cls.get_bounds(model_constructor, index_sets),
            mask=mask,
            doc=cls.doc,
            unit_category=cls.unit_category,
        )

    @classmethod
    def get_index_sets(cls, model_constructor: "ModelConstructor"):
        """Return the model indices used to construct this variable."""
        return model_constructor.create_custom_set(cls.indices)

    @classmethod
    def get_bounds(cls, model_constructor: "ModelConstructor", index_sets):
        """Return the variable bounds, if any."""
        return None

    @classmethod
    def get_mask(cls, model_constructor: "ModelConstructor", index_sets):
        """Return an optional mask restricting the constructed variable."""
        return None

    @classmethod
    def should_construct(cls, model_constructor: "ModelConstructor") -> bool:
        """Return whether this variable is required by the current configuration."""
        return True
