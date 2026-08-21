from zen_garden.constraints.generic_constraint import GenericConstraint


class CarbonEmissionsAnnualConstraint(GenericConstraint):
    def build(self):
        r"""Summary:
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
            self.zen_model.variables["carbon_emissions_annual"]
            - self.zen_model.variables["carbon_emissions_technology_total"]
            - self.zen_model.variables["carbon_emissions_carrier_total"]
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint("constraint_carbon_emissions_annual", constraints)
