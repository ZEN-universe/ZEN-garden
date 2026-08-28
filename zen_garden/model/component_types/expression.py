"""Generic linear-expression class for ZenModel."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor


class GenericExpression(ABC):
    """Base class for reusable linear expressions.

    An expression is a named piece of a constraint or objective -- a linopy
    ``LinearExpression`` (or ``xarray.DataArray``) built from variables and
    parameters during model construction, after the variables and before the
    constraints. :meth:`build` registers it on the optimization model via
    ``zen_model.add_expression``; it is then reused elsewhere through
    ``zen_model.expressions[<name>]``.
    """

    name: ClassVar[str]
    doc: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name"):
            raise TypeError(f"{cls.__name__} must define 'name'")

    @classmethod
    def build(cls, model_constructor: "ModelConstructor") -> None:
        """Compute the expression and register it on the optimization model."""
        if not cls.should_construct(model_constructor):
            return
        model_constructor.zen_model.add_expression(
            cls.name, cls.get_expression(model_constructor)
        )

    @classmethod
    @abstractmethod
    def get_expression(cls, model_constructor: "ModelConstructor") -> Any:
        """Return the linear expression built from variables and parameters."""

    @classmethod
    def should_construct(cls, model_constructor: "ModelConstructor") -> bool:
        """Return whether this expression is needed by the current configuration."""
        return True
