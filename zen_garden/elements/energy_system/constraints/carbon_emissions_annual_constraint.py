from zen_garden.model.component_types.constraint import GenericConstraint


class CarbonEmissionsAnnualConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Add up all carbon emissions from technologies and carriers.

        Formulation:

        .. math::
            M_y = M_y^\\mathrm{tech} + M_y^\\mathrm{carrier}

        Notation:

        :math:`M_y^\\mathrm{tech}`: carbon emissions from technologies in
        year :math:`y`
        :math:`M_y^\\mathrm{carrier}`: carbon emissions from carriers in year
        :math:`y`
        """
        lhs = (
            model_constructor.optimization_model.variables["carbon_emissions_annual"]
            - model_constructor.optimization_model.variables[
                "carbon_emissions_technology_total"
            ]
            - model_constructor.optimization_model.variables[
                "carbon_emissions_carrier_total"
            ]
        )
        rhs = 0
        constraints = lhs == rhs

        model_constructor.optimization_model.add_constraint(
            "constraint_carbon_emissions_annual", constraints
        )
