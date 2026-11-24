"""Operational activity of retrofit technologies."""

import pandas as pd

from zen_garden.elements.conversion_technology.expressions.conversion_reference_flow import (  # noqa: E501
    ConversionReferenceFlow,
)
from zen_garden.elements.retrofitting_technology.sets.relation_retrofitting_technology_base_technology import (  # noqa: E501
    RelationRetrofittingTechnologyBaseTechnology,
)
from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.elements.technology.sets.set_retrofitting_technologies import (
    SetRetrofittingTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.linear_expression import map_and_expand


class RetrofitActivity(GenericExpression):
    """Map reference flow to the configured retrofit technologies."""

    name = "retrofit_activity"
    unit_category = {"energy_quantity": 1, "time": -1}
    indices = (
        "set_conversion_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )
    dependencies = (ConversionReferenceFlow,)

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct map reference flow to the configured retrofit technologies.

        Summary:
            Map reference flow to the configured retrofit technologies.

        Formulation:

        .. math::

            \\{F^{ref}_{h,n,y,r}:h\\in\\mathcal H^{retrofit}(b)\\}

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Expression over its declared model indices.
        """
        optimization_model = model_constructor.optimization_model
        base_technologies = {}
        for technology in optimization_model.sets[SetRetrofittingTechnologies]:
            related = optimization_model.sets.project_relation(
                RelationRetrofittingTechnologyBaseTechnology,
                SetConversionTechnologies,
                set_retrofitting_technologies=technology,
            )
            if len(related) != 1:
                raise ValueError(
                    f"Retrofit technology {technology!r} must have exactly one "
                    f"base technology, found {len(related)}"
                )
            base_technologies[technology] = related[0]
        mapping = pd.Series(base_technologies, name="set_conversion_technologies")
        mapping.index.name = "set_conversion_technologies"
        return map_and_expand(
            optimization_model.expressions[ConversionReferenceFlow], mapping
        )
