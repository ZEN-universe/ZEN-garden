import numpy as np
import xarray as xr

from zen_garden.elements.conversion_technology.expressions.conversion_reference_flow import (  # noqa: E501
    ConversionReferenceFlow,
)
from zen_garden.elements.conversion_technology.parameters.min_full_load_hours_fraction import (  # noqa: E501
    MinFullLoadHoursFraction,
)
from zen_garden.elements.energy_system.parameters.modeled_year_hours import (
    ModeledYearHours,
)
from zen_garden.elements.energy_system.parameters.weight_representative_hours import (
    WeightRepresentativeHours,
)
from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.technology.masks import PowerCapacityYearMask
from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.elements.technology.variables.capacity import Capacity
from zen_garden.model.component_types.constraint import GenericConstraint


class MinimumFullLoadHoursConstraint(GenericConstraint):
    """Minimum Full Load Hours."""

    name = "constraint_minimum_full_load_hours"
    indices = (
        "set_conversion_technologies",
        "set_nodes",
        "set_years",
    )
    masks = (PowerCapacityYearMask,)

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(
            model_constructor.optimization_model.sets[SetConversionTechnologies]
        )

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Sets minimum full load hours for each unit.

        This constraint requires that a minimum number of full_load_hours be met
        over the course of year. Full load hours are the amount of hours that
        a conversion technology would need to run at full capacity in order
        to produce an output equivalent to its yearly total. The constraint can
        be used to require a conversion technology to always operate at
        baseload capacity. This can be helpful for technologies where ramping
        is not possible or economical for reasons not captured by the model.

        Formulation:

        .. math::
            \\sum_{t\\in\\mathcal{T}_y} \\Delta t_t F^{\\mathrm{ref}}_{h,n,t} \\geq
            T^\\mathrm{unaggregated}_y
            \\underline{\\pi}_{h,n,y} K_{h,n,y}
            \\qquad \\forall h,n,y

        Notation:

        :math:`T^\\mathrm{unaggregated}_y` is the number of unaggregated hours per
        year configured in ``system.json``. The duration weights :math:`\\Delta t_t`
        convert representative operational flows into annual production.

        - :math:`\\underline{\\pi}_{h,n,y}`: minimum number of full load hours,
          expressed as a fraction of the unaggregated time steps per year. Takes
          separate values for each technology :math:`h` at node :math:`n` and
          planning period :math:`y`

        - :math:`K_{h,n,y}`: installed capacity of the technology :math:`h` at
          node :math:`n` in planning period :math:`y`

        - :math:`F^{\\mathrm{ref}}_{h,n,t}`: reference carrier flow of the technology
          :math:`h` at node :math:`n` in time step :math:`t` in planning
          period :math:`y`

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        # get dimensions
        techs = model_constructor.optimization_model.sets[SetConversionTechnologies]
        nodes = model_constructor.optimization_model.sets[SetNodes]
        # define mask
        min_full_load_hours_fraction = model_constructor.optimization_model.parameters[
            MinFullLoadHoursFraction
        ]
        mask = xr.DataArray(
            ~np.isclose(min_full_load_hours_fraction, 0),
            dims=min_full_load_hours_fraction.dims,
            coords=min_full_load_hours_fraction.coords,
        )
        # create constraint
        term_capacity = (
            min_full_load_hours_fraction
            * model_constructor.optimization_model.parameters[ModeledYearHours]
            * model_constructor.optimization_model.variables[Capacity]
            .sel(
                {
                    "set_technologies": techs,
                    "set_location": nodes,
                }
            )
            .rename(
                {
                    "set_technologies": "set_conversion_technologies",
                    "set_location": "set_nodes",
                }
            )
        )
        weights = model_constructor.optimization_model.parameters[
            WeightRepresentativeHours
        ]
        term_annual_production = (
            model_constructor.optimization_model.expressions[ConversionReferenceFlow]
            * weights
        ).sum("set_representative_hours")

        lhs = term_annual_production.where(mask) - term_capacity.where(mask)
        rhs = 0
        constraints = lhs >= rhs

        return constraints
