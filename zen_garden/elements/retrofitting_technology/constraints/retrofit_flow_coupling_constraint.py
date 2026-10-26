from zen_garden.elements.retrofitting_technology.expressions.retrofit_activity import (
    RetrofitActivity,
)
from zen_garden.elements.retrofitting_technology.expressions.retrofit_base_activity import (  # noqa: E501
    RetrofitBaseActivity,
)
from zen_garden.elements.retrofitting_technology.parameters.retrofit_flow_coupling_factor import (  # noqa: E501
    RetrofitFlowCouplingFactor,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class RetrofitFlowCouplingConstraint(GenericConstraint):
    """Limit retrofit activity by the activity of its base technology."""

    name = "constraint_retrofit_flow_coupling"
    indices = (
        "set_conversion_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct limit retrofit activity by the activity of its base technology.

        Summary:
            Limit retrofit activity by the activity of its base technology.

        Formulation:

        .. math::

            F^{ref}_{b,n,y,r}\\leq\\gamma_{h,b}F^{ref}_{h,n,y,r}

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
        coupling = model.parameters[RetrofitFlowCouplingFactor].rename(
            {"set_retrofitting_technologies": "set_conversion_technologies"}
        )
        return model.expressions[RetrofitBaseActivity] <= (
            coupling * model.expressions[RetrofitActivity]
        )
