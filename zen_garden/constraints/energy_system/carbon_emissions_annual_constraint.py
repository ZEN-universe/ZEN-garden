from zen_garden.constraints.generic_constraint import GenericConstraint


class CarbonEmissionsAnnualConstraint(GenericConstraint):
    def build(self):
        """Add up all carbon emissions from technologies and carriers.

        .. math::
            E_y = E_{y,\\mathcal{H}} + E_{y,\\mathcal{C}}

        :math:`E_{y,\\mathcal{H}}`: carbon emissions from technologies in
        year :math:`y` \n
        :math:`E_{y,\\mathcal{C}}`: carbon emissions from carriers in year :math

        """
        lhs = (
            self.zen_model.variables["carbon_emissions_annual"]
            - self.zen_model.variables["carbon_emissions_technology_total"]
            - self.zen_model.variables["carbon_emissions_carrier_total"]
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint("constraint_carbon_emissions_annual", constraints)
