import xarray as xr

from zen_garden.elements.carrier.expressions.nodal_net_supply import NodalNetSupply
from zen_garden.elements.carrier.parameters.demand import Demand
from zen_garden.model.component_types.constraint import GenericConstraint


class NodalEnergyBalanceConstraint(GenericConstraint):
    """Balance net carrier supply and demand at every node and time step."""

    name = "constraint_nodal_energy_balance"
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct balance net carrier supply and demand at every node and time step.

        Summary:
            Balance net carrier supply and demand at every node and time step.

        Formulation:

        .. math::

             \\sum_{h:c \\in \\mathcal C_h^{out}}F^{out}_{h,c,n,y,r}- \\sum_{h:c \\in
            \\mathcal C_h^{in}}F^{in}_{h,c,n,y,r} + \\sum_{h,e
            \\in\\delta^-(n)}(1-\\lambda^{trans}_{h,e})F^{trans}_{h,e,y,r}- \\sum_{h,e
            \\in\\delta^ +(n)}F^{trans}_{h,e,y,r}- \\sum_{h:c_h^{ref}
            =c}F^{ch}_{h,n,y,r} + \\sum_{h:c_h^{ref} =c}F^{dis}_{h,n,y,r}
            +F^{imp}_{c,n,y,r}-F^{exp}_{c,n,y,r} +F^{shed}_{c,n,y,r} =d_{c,n,y,r}

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
        net_supply = model.expressions[NodalNetSupply]
        demand = model.parameters[Demand]
        aligned_coordinates = xr.align(net_supply.coords, demand, join="inner")[0]
        return net_supply.sel(aligned_coordinates) == demand.sel(aligned_coordinates)
