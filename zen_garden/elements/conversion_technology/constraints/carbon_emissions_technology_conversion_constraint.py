from zen_garden.elements.conversion_technology.expressions.conversion_reference_flow import (  # noqa: E501
    ConversionReferenceFlow,
)
from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.elements.technology.parameters.carbon_intensity_technology import (
    CarbonIntensityTechnology,
)
from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.elements.technology.variables.carbon_emissions_technology import (
    CarbonEmissionsTechnology,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class CarbonEmissionsTechnologyConversionConstraint(GenericConstraint):
    """Calculate operating emissions from conversion reference flow."""

    name = "constraint_carbon_emissions_technology_conversion"
    indices = (
        "set_conversion_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )
    masks = (PowerOperationMask,)

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(
            model_constructor.optimization_model.sets[SetConversionTechnologies]
        )

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct calculate operating emissions from conversion reference flow.

        Summary:
            Calculate operating emissions from conversion reference flow.

        Formulation:

        .. math::

            E^{technology}=e^{technology}F^{ref}

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        model = model_constructor.optimization_model
        technologies = model.sets[SetConversionTechnologies]
        nodes = model.sets[SetNodes]
        emissions = (
            model.variables[CarbonEmissionsTechnology]
            .sel(set_technologies=technologies, set_location=nodes)
            .rename(
                set_technologies="set_conversion_technologies",
                set_location="set_nodes",
            )
        )
        factor = model.parameters[CarbonIntensityTechnology].rename(
            set_technologies="set_conversion_technologies",
            set_location="set_nodes",
        )
        return emissions == factor * model.expressions[ConversionReferenceFlow]
