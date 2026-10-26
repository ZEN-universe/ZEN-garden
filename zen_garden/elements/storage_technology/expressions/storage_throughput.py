"""Combined charging and discharging activity of storage technologies."""

from zen_garden.elements.storage_technology.variables.flow_storage_charge import (
    FlowStorageCharge,
)
from zen_garden.elements.storage_technology.variables.flow_storage_discharge import (
    FlowStorageDischarge,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression


class StorageThroughput(GenericExpression):
    """Return charging plus discharging flow on generic technology dimensions."""

    name = "storage_throughput"
    indices = (
        "set_technologies",
        "set_location",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(model_constructor.optimization_model.sets[SetStorageTechnologies])

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct the model component.

        Summary:
            Return charging plus discharging flow on generic technology dimensions.

        Formulation:

        .. math::

            F^{ch}_{h,n,y,r}+F^{dis}_{h,n,y,r}

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
        return (
            optimization_model.variables[FlowStorageCharge]
            + optimization_model.variables[FlowStorageDischarge]
        ).rename(
            {
                "set_storage_technologies": "set_technologies",
                "set_nodes": "set_location",
            }
        )
