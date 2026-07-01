"""Class defining a generic energy carrier.

The class takes as inputs the abstract optimization model. The class adds parameters,
variables and constraints of a generic carrier and returns the abstract optimization
model.
"""

import logging
from typing import TYPE_CHECKING, override

import linopy as lp
import numpy as np
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.model.components.zen_index import ZenIndex
from zen_garden.model.config import Config
from zen_garden.model.context import Context
from zen_garden.model.element import Element, ElementConstructor
from zen_garden.model.generic_rule import GenericRule
from zen_garden.model.zen_model import ZenModel

if TYPE_CHECKING:
    from zen_garden.model.energy_system import EnergySystem
    from zen_garden.preprocess.unit_handling import UnitHandling
    from zen_garden.services.element_registry import ElementRegistry

logger = logging.getLogger(__name__)


class Carrier(Element):
    """Class defining a generic energy carrier."""

    # set label
    label = "set_carriers"
    # empty list of elements
    list_of_elements = []

    def __init__(
        self,
        carrier_name: str,
        config: Config,
        context: Context,
        energy_system: "EnergySystem",
        element_registry: "ElementRegistry",
        unit_handling: "UnitHandling",
    ):
        """Initialization of a generic carrier object.

        :param carrier: carrier that is added to the model
        :param optimization_setup: The OptimizationSetup the element is part of
        """
        super().__init__(
            carrier_name,
            config,
            context,
            energy_system,
            element_registry,
            unit_handling,
        )

    def store_input_data(self):
        """Retrieves and stores input data for element as attributes. Each Child class
        overwrites method to store different attributes.
        """
        # store scenario dict
        super().store_scenario_dict()
        # set attributes of carrier
        # raw import
        self.raw_time_series = dict()
        self.raw_time_series["demand"] = self.data_input.extract_input_data(
            "demand",
            index_sets=["set_nodes", "set_time_steps"],
            time_steps="set_base_time_steps_yearly",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        self.raw_time_series["availability_import"] = (
            self.data_input.extract_input_data(
                "availability_import",
                index_sets=["set_nodes", "set_time_steps"],
                time_steps="set_base_time_steps_yearly",
                unit_category={"energy_quantity": 1, "time": -1},
            )
        )
        self.raw_time_series["availability_export"] = (
            self.data_input.extract_input_data(
                "availability_export",
                index_sets=["set_nodes", "set_time_steps"],
                time_steps="set_base_time_steps_yearly",
                unit_category={"energy_quantity": 1, "time": -1},
            )
        )
        self.raw_time_series["price_export"] = self.data_input.extract_input_data(
            "price_export",
            index_sets=["set_nodes", "set_time_steps"],
            time_steps="set_base_time_steps_yearly",
            unit_category={"money": 1, "energy_quantity": -1},
        )
        self.raw_time_series["price_import"] = self.data_input.extract_input_data(
            "price_import",
            index_sets=["set_nodes", "set_time_steps"],
            time_steps="set_base_time_steps_yearly",
            unit_category={"money": 1, "energy_quantity": -1},
        )
        # non-time series input data
        self.availability_import_yearly = self.data_input.extract_input_data(
            "availability_import_yearly",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"energy_quantity": 1},
        )
        self.availability_export_yearly = self.data_input.extract_input_data(
            "availability_export_yearly",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"energy_quantity": 1},
        )
        self.carbon_intensity_carrier_import = self.data_input.extract_input_data(
            "carbon_intensity_carrier_import",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"emissions": 1, "energy_quantity": -1},
        )
        self.carbon_intensity_carrier_export = self.data_input.extract_input_data(
            "carbon_intensity_carrier_export",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"emissions": 1, "energy_quantity": -1},
        )
        self.price_shed_demand = self.data_input.extract_input_data(
            "price_shed_demand",
            index_sets=[],
            unit_category={"money": 1, "energy_quantity": -1},
        )

    def overwrite_time_steps(self, base_time_steps):
        """Overwrites set_time_steps_operation.

        :param base_time_steps: #TODO describe parameter/return
        """
        set_time_steps_operation = self.energy_system.time_steps.encode_time_step(
            base_time_steps=base_time_steps, time_step_type="operation"
        )
        assert isinstance(set_time_steps_operation, np.ndarray)
        self.set_time_steps_operation = set_time_steps_operation.squeeze().tolist()


class CarrierConstructor(ElementConstructor):
    element_class = Carrier

    @override
    def has_elements(self) -> bool:
        """Checks if there are any elements of the class <Carrier>.

        :return: True if there are elements, False otherwise
        """
        return True

    def construct_sets(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the pe.Sets of the class <Carrier>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        logger.info("Constructing sets for Carrier")

    def construct_params(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the pe.Params of the class <Carrier>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        logger.info("Constructing parameters for Carrier")

        # demand of carrier
        self.add_parameter(
            zen_model,
            energy_system,
            name="demand",
            index_names=["set_carriers", "set_nodes", "set_time_steps_operation"],
            doc="Parameter which specifies the carrier demand",
        )
        # availability of carrier
        self.add_parameter(
            zen_model,
            energy_system,
            name="availability_import",
            index_names=["set_carriers", "set_nodes", "set_time_steps_operation"],
            doc="Parameter which specifies the maximum energy that can be imported "
            "from outside the system boundaries",
        )
        # availability of carrier
        self.add_parameter(
            zen_model,
            energy_system,
            name="availability_export",
            index_names=["set_carriers", "set_nodes", "set_time_steps_operation"],
            doc="Parameter which specifies the maximum energy that can be exported "
            "to outside the system boundaries",
        )
        # availability of carrier
        self.add_parameter(
            zen_model,
            energy_system,
            name="availability_import_yearly",
            index_names=["set_carriers", "set_nodes", "set_time_steps_yearly"],
            doc="Parameter which specifies the maximum energy that can be imported "
            "from outside the system boundaries for the entire year",
        )
        # availability of carrier
        self.add_parameter(
            zen_model,
            energy_system,
            name="availability_export_yearly",
            index_names=["set_carriers", "set_nodes", "set_time_steps_yearly"],
            doc="Parameter which specifies the maximum energy that can be exported "
            "to outside the system boundaries for the entire year",
        )
        # import price
        self.add_parameter(
            zen_model,
            energy_system,
            name="price_import",
            index_names=["set_carriers", "set_nodes", "set_time_steps_operation"],
            doc="Parameter which specifies the import carrier price",
        )
        # export price
        self.add_parameter(
            zen_model,
            energy_system,
            name="price_export",
            index_names=["set_carriers", "set_nodes", "set_time_steps_operation"],
            doc="Parameter which specifies the export carrier price",
        )
        # demand shedding price
        self.add_parameter(
            zen_model,
            energy_system,
            name="price_shed_demand",
            index_names=["set_carriers"],
            doc="Parameter which specifies the price to shed demand",
        )
        # carbon intensity carrier import
        self.add_parameter(
            zen_model,
            energy_system,
            name="carbon_intensity_carrier_import",
            index_names=["set_carriers", "set_nodes", "set_time_steps_yearly"],
            doc="Parameter which specifies the carbon intensity of carrier import",
        )
        # carbon intensity carrier export
        self.add_parameter(
            zen_model,
            energy_system,
            name="carbon_intensity_carrier_export",
            index_names=["set_carriers", "set_nodes", "set_time_steps_yearly"],
            doc="Parameter which specifies the carbon intensity of carrier export",
        )

    def construct_vars(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the pe.Vars of the class <Carrier>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        logger.info("Constructing variables for Carrier")

        variables = zen_model.variables
        sets = zen_model.sets

        # flow of imported carrier
        variables.add_variable(
            name="flow_import",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
                zen_model,
                energy_system,
            ),
            bounds=(0.0, np.inf),
            doc="node- and time-dependent carrier import from the grid",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # flow of exported carrier
        variables.add_variable(
            name="flow_export",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
                zen_model,
                energy_system,
            ),
            bounds=(0.0, np.inf),
            doc="node- and time-dependent carrier export from the grid",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # carrier import/export cost
        variables.add_variable(
            name="cost_carrier",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
                zen_model,
                energy_system,
            ),
            doc="node- and time-dependent carrier cost due to import and export",
            unit_category={"money": 1, "time": -1},
        )
        # total carrier import/export cost
        variables.add_variable(
            name="cost_carrier_total",
            index_sets=sets["set_time_steps_yearly"],
            doc="total carrier cost due to import and export",
            unit_category={"money": 1},
        )
        # carbon emissions
        variables.add_variable(
            name="carbon_emissions_carrier",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
                zen_model,
                energy_system,
            ),
            doc="carbon emissions of importing and exporting carrier",
            unit_category={"emissions": 1, "time": -1},
        )
        # carbon emissions carrier
        variables.add_variable(
            name="carbon_emissions_carrier_total",
            index_sets=sets["set_time_steps_yearly"],
            doc="total carbon emissions of importing and exporting carrier",
            unit_category={"emissions": 1},
        )
        # shed demand
        variables.add_variable(
            name="shed_demand",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
                zen_model,
                energy_system,
            ),
            bounds=(0.0, np.inf),
            doc="shed demand of carrier",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # cost of shed demand
        variables.add_variable(
            name="cost_shed_demand",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
                zen_model,
                energy_system,
            ),
            bounds=(0.0, np.inf),
            doc="shed demand of carrier",
            unit_category={"money": 1, "time": -1},
        )

    def construct_constraints(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the Constraints of the class <Carrier>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        logger.info("Constructing constraints for Carrier")

        rules = CarrierRules(self.config, self.context)

        # limit import/export flow by availability
        rules.constraint_availability_import_export(zen_model, energy_system)

        # limit import/export flow by availability for each year
        rules.constraint_availability_import_export_yearly(zen_model, energy_system)

        # cost for carrier
        rules.constraint_cost_carrier(zen_model, energy_system)

        # cost and limit for shed demand
        rules.constraint_cost_limit_shed_demand(zen_model, energy_system)

        # total cost for carriers
        rules.constraint_cost_carrier_total(zen_model, energy_system)

        # carbon emissions
        rules.constraint_carbon_emissions_carrier(zen_model, energy_system)

        # carbon emissions carrier
        rules.constraint_carbon_emissions_carrier_total(zen_model, energy_system)

        # energy balance
        index_values, index_names = self.create_custom_set(
            ["set_carriers", "set_nodes", "set_time_steps_operation"],
            zen_model,
            energy_system,
        )
        rules.constraint_nodal_energy_balance(
            zen_model,
            energy_system,
            ZenIndex(index_values, index_names),
            index_names[:1],
        )


class CarrierRules(GenericRule):
    """Rules for the Carrier class."""

    def __init__(self, config: Config, context: Context):
        """Inits the rules for a given EnergySystem.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        super().__init__(config, context)

    # Rule-based constraints
    # ----------------------

    def constraint_cost_carrier_total(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Total cost of importing and exporting carrier.

        .. math::
            C_y^{\\mathcal{C}} = \\sum_{c\\in\\mathcal{C}}\\sum_{n\\in\\mathcal{N}}
            \\sum_{t\\in\\mathcal{T}} \\tau_t (O_{c,n,t} + O_{c,n,t}^{\\mathrm{shed}\\
            \\mathrm{demand}})

        :math:`O_{c,n,t}`: cost of importing and exporting carrier :math:`c`
        at node :math:`n` and time step :math:`t`\n
        :math:`O_{c,n,t}^{\\mathrm{shed\\ demand}}`: cost of shedding demand
        of carrier :math:`c` at node :math:`n` and time step :math:`t`\n
        :math:`\\tau_t`: duration of time step :math:`t`


        """
        times = self.get_year_time_step_duration_array(zen_model, energy_system)
        term_summed_cost_carrier = (
            (
                zen_model.lp_model.variables["cost_carrier"].broadcast_like(times)
                + zen_model.lp_model.variables["cost_shed_demand"].broadcast_like(times)
            )
            * times
        ).sum(["set_carriers", "set_nodes", "set_time_steps_operation"])
        lhs = (
            zen_model.lp_model.variables["cost_carrier_total"]
            - term_summed_cost_carrier
        )
        rhs = 0
        constraints = lhs == rhs

        zen_model.constraints.add_constraint(
            "constraint_cost_carrier_total", constraints
        )

    def constraint_carbon_emissions_carrier_total(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Total carbon emissions of importing and exporting carrier.

        .. math::
            E_y^{\\mathcal{C}} = \\sum_{c\\in\\mathcal{C}}\\sum_{n\\in\\mathcal{N}}
            \\sum_{t\\in\\mathcal{T}} \\tau_t \\theta_{c,n,t}^{\\mathrm{carrier}}

        :math:`\\theta_{c,n,t}^{\\mathrm{carrier}}`: carbon emissions of importing and
        exporting carrier :math:`c` at node :math:`n` and time step :math:`t`\n
        :math:`\\tau_t`: duration of time step :math:`t`

        """
        term_summed_carbon_emissions_carrier = (
            zen_model.lp_model.variables["carbon_emissions_carrier"]
            * self.get_year_time_step_duration_array(zen_model, energy_system)
        ).sum(["set_carriers", "set_nodes", "set_time_steps_operation"])
        lhs = (
            zen_model.lp_model.variables["carbon_emissions_carrier_total"]
            - term_summed_carbon_emissions_carrier
        )
        rhs = 0
        constraints = lhs == rhs

        zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_carrier_total", constraints
        )

    def constraint_availability_import_export(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """node- and time-dependent carrier availability to import/export from outside
        the system boundaries.

        .. math::
            \\underline{U}_{c,n,t} \\leq \\underline{a}_{c,n,t}

        .. math::
            \\overline{U}_{c,n,t} \\leq \\overline{a}_{c,n,t}

        :math:`\\underline{U}_{c,n,t}`: flow of carrier :math:`c` imported
        at node :math:`n` and time step :math:`t`\n
        :math:`\\overline{U}_{c,n,t}`: flow of carrier :math:`c` exported
        at node :math:`n` and time step :math:`t`\n
        :math:`\\underline{a}_{c,n,t}`: availability of carrier :math:`c` to import
        at node :math:`n` and time step :math:`t`\n
        :math:`\\overline{a}_{c,n,t}`: availability of carrier :math:`c` to export
        at node :math:`n` and time step :math:`t`

        """
        lhs_imp = zen_model.lp_model.variables["flow_import"]
        rhs_imp = zen_model.parameters.availability_import
        constraints_imp = lhs_imp <= rhs_imp

        lhs_exp = zen_model.lp_model.variables["flow_export"]
        rhs_exp = zen_model.parameters.availability_export
        constraints_exp = lhs_exp <= rhs_exp

        zen_model.constraints.add_constraint(
            "constraint_availability_import", constraints_imp
        )
        zen_model.constraints.add_constraint(
            "constraint_availability_export", constraints_exp
        )

    def constraint_availability_import_export_yearly(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """node- and year-dependent carrier availability to import/export from outside
        the system boundaries.

        .. math::
            \\underline{a}_{c,n,y}^\\mathrm{Y} \\geq \\sum_{t\\in\\mathcal{T}}\\tau_t
            \\underline{U}_{c,n,t}

        .. math::
            \\overline{a}_{c,n,y}^\\mathrm{Y} \\geq \\sum_{t\\in\\mathcal{T}}\\tau_t
            \\overline{U}_{c,n,t}

        :math:`\\underline{a}_{c,n,y}^\\mathrm{Y}`: yearly availability of
        carrier :math:`c` to import at node :math:`n`\n
        :math:`\\overline{a}_{c,n,y}^\\mathrm{Y}`: yearly availability of
        carrier :math:`c` to export at node :math:`n`\n
        :math:`\\tau_t`: is the duration of time step :math:`t`\n
        :math:`\\underline{U}_{c,n,t}`: flow of carrier :math:`c` imported at
        node :math:`n` at time step :math:`t`\n
        :math:`\\overline{U}_{c,n,t}`: flow of carrier :math:`c` exported at
        node :math:`n` at time step :math:`t`


        """
        # The constraint is only constrained if the availability is finite
        mask_imp = zen_model.parameters.availability_import_yearly != np.inf
        mask_exp = zen_model.parameters.availability_export_yearly != np.inf

        # import
        lhs_imp = (
            (
                zen_model.lp_model.variables["flow_import"]
                * self.get_year_time_step_duration_array(zen_model, energy_system)
            )
            .sum("set_time_steps_operation")
            .where(mask_imp)
        )
        rhs_imp = zen_model.parameters.availability_import_yearly.where(mask_imp)
        constraints_imp = lhs_imp <= rhs_imp

        # export
        lhs_exp = (
            (
                zen_model.lp_model.variables["flow_export"]
                * self.get_year_time_step_duration_array(zen_model, energy_system)
            )
            .sum("set_time_steps_operation")
            .where(mask_exp)
        )
        rhs_exp = zen_model.parameters.availability_export_yearly.where(mask_exp)
        constraints_exp = lhs_exp <= rhs_exp

        zen_model.constraints.add_constraint(
            "constraint_availability_import_yearly", constraints_imp
        )
        zen_model.constraints.add_constraint(
            "constraint_availability_export_yearly", constraints_exp
        )

    def constraint_cost_carrier(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Cost of importing and exporting carrier.

        .. math::
           O_{c,n,t} = \\underline{u}_{c,n,t} \\underline{U}_{c,n,t} -
           \\overline{v}_{c,n,t} \\overline{U}_{c,n,t}

        :math:`\\underline{u}_{c,n,t}`: import price of carrier :math:`c` at
        node :math:`n` and time step :math:`t`\n
        :math:`\\overline{v}_{c,n,t}`: export price of carrier :math:`c` at
        node :math:`n` and time step :math:`t`\n
        :math:`\\underline{U}_{c,n,t}`: flow of carrier :math:`c` imported at
        node :math:`n` and time step :math:`t`\n
        :math:`\\overline{U}_{c,n,t}`: flow of carrier :math:`c` exported at
        node :math:`n` and time step :math:`t`

        """
        ### formulate constraint
        lhs = (
            zen_model.lp_model.variables["cost_carrier"]
            - zen_model.parameters.price_import
            * zen_model.lp_model.variables["flow_import"]
            + zen_model.parameters.price_export
            * zen_model.lp_model.variables["flow_export"]
        )
        rhs = 0
        constraints = lhs == rhs

        zen_model.constraints.add_constraint("constraint_cost_carrier", constraints)

    def constraint_cost_limit_shed_demand(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Cost and limit of shedding demand of carrier.

        .. math::
           O_{c,n,t}^{\\mathrm{shed\\ demand}} = D_{c,n,t} \\nu_c \n
           D_{c,n,t} \\leq d_{c,n,t}

        :math:`O_{c,n,t}^{\\mathrm{shed\\ demand}}`: total cost of shedding
        demand of carrier :math:`c` at node :math:`n` and time step :math:`t`\n
        :math:`\\nu_c`: price to shed demand of carrier :math:`c`\n
        :math:`D_{c,n,t}`: shed demand of carrier :math:`c` at node :math:`n` and
        time step :math:`t`\n
        :math:`d_{c,n,t}`: demand of carrier :math:`c` at node :math:`n` and
        time step :math:`t`


        """
        ### mask for finite price, otherwise the shed demand is zero
        mask = zen_model.parameters.price_shed_demand != np.inf

        # cost of shedding demand
        lhs_cost = (
            zen_model.lp_model.variables["cost_shed_demand"]
            - zen_model.parameters.price_shed_demand
            * zen_model.lp_model.variables["shed_demand"]
        ).where(mask)
        rhs_cost = 0
        constraints_cost = lhs_cost == rhs_cost

        # limit of shedding demand:
        #   either the demand (price != inf) or zero (price == inf)
        lhs_shed_demand = zen_model.lp_model.variables["shed_demand"]
        rhs_shed_demand = zen_model.parameters.demand.where(mask, 0.0)
        constraints_shed_demand = lhs_shed_demand <= rhs_shed_demand

        zen_model.constraints.add_constraint(
            "constraint_cost_shed_demand", constraints_cost
        )
        zen_model.constraints.add_constraint(
            "constraint_limit_shed_demand", constraints_shed_demand
        )

    def constraint_carbon_emissions_carrier(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Carbon emissions of importing and exporting carrier.

        .. math::
           \\theta_{c,n,t}^{\\mathrm{carrier}} = \\underline{\\epsilon_c}
           \\underline{U}_{c,n,t} - \\overline{\\epsilon_c} \\overline{U}_{c,n,t}

        :math:`\\theta_{c,n,t}^{\\mathrm{carrier}}`: carbon emissions of importing and
        exporting carrier :math:`c` at node :math:`n` and time step :math:`t`\n
        :math:`\\underline{\\epsilon_c}`: carbon intensity of carrier import :math:`c`\n
        :math:`\\overline{\\epsilon_c}`: carbon intensity of carrier export :math:`c`\n
        :math:`\\underline{U}_{c,n,t}`: flow of carrier :math:`c` imported at
        node :math:`n` and time step :math:`t`\n
        :math:`\\overline{U}_{c,n,t}`: flow of carrier :math:`c` exported at
        node :math:`n` and time step :math:`t`

        """
        # create times xarray with 1 where the operation time step is in the year
        times = self.get_year_time_step_array(zen_model, energy_system)
        # convert the carbon intensity carrier from yearly to operation time steps
        # TODO map and expand
        carbon_intensity_carrier_import = (
            zen_model.parameters.carbon_intensity_carrier_import.broadcast_like(times)
            * times
        ).sum("set_time_steps_yearly")
        carbon_intensity_carrier_export = (
            zen_model.parameters.carbon_intensity_carrier_export.broadcast_like(times)
            * times
        ).sum("set_time_steps_yearly")
        lhs = zen_model.lp_model.variables["carbon_emissions_carrier"] - (
            zen_model.lp_model.variables["flow_import"]
            * carbon_intensity_carrier_import
            - zen_model.lp_model.variables["flow_export"]
            * carbon_intensity_carrier_export
        )

        rhs = 0

        constraints = lhs == rhs

        zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_carrier", constraints
        )

    def constraint_nodal_energy_balance(
        self,
        zen_model: ZenModel,
        energy_system: "EnergySystem",
        index: ZenIndex,
        first_index_name,
    ):
        """Nodal energy balance for each time step.

        .. math::
            0 = -(d_{c,n,t}-D_{c,n,t})
            + \\sum_{i\\in\\mathcal{I}}(\\overline{G}_{c,i,n,t}
            - \\underline{G}_{c,i,n,t})
            + \\sum_{j\\in\\mathcal{J}}(\\sum_{e\\in\\underline{\\mathcal{E}}}(F_{j,e,t}
            - F^\\mathrm{l}_{j,e,t}) - \\sum_{e'\\in\\overline{\\mathcal{E}}}F_{j,e',t})
            + \\sum_{k\\in\\mathcal{K}}(\\overline{H}_{k,n,t} - \\underline{H}_{k,n,t})
            + \\underline{U}_{c,n,t} - \\overline{U}_{c,n,t}

        Sources of carrier :math:`c` at node :math:`n` and time step :math:`t`:\n
        :math:`\\overline{G}_{c,i,n,t}`: output flow of carrier :math:`c` from all
        conversion technologies :math:`i` at node :math:`n` at time step :math:`t`\n
        :math:`F_{j,e,t}`: transported flow of carrier :math:`c` on ingoing
        edges :math:`e` minues the losses :math:`F^\\mathrm{l}_{j,e,t})` of all
        transport technologies :math:`j` at time step :math:`t`\n
        :math:`\\overline{H}_{k,n,t}`: output flow of carrier :math:`c` from all
        storage technologies :math:`k` at node :math:`n` at time step :math:`t`\n
        :math:`\\underline{U}_{c,n,t}`: flow of carrier :math:`c` imported at
        node :math:`n` at time step :math:`t`\n

        Sinks of carrier :math:`c` at node :math:`n` and time step :math:`t`:\n
        :math:`d_{c,n,t}`: demand of carrier :math:`c` at node :math:`n` at
        time step :math:`t` minus the shed demand :math:`D_{c,n,t}`\n
        :math:`\\underline{G}_{c,i,n,t}`: input flow of carrier :math:`c` to all
        conversion technologies :math:`i` at node :math:`n` at time step :math:`t`\n
        :math:`F_{j,e',t}`: transported flow of carrier :math:`c` on outgoing
        edges :math:`e'` at time step :math:`t`\n
        :math:`\\underline{H}_{k,n,t}`: input flow of carrier :math:`c` to all
        storage technologies :math:`k` at node :math:`n` at time step :math:`t`\n
        :math:`\\overline{U}_{c,n,t}`: flow of carrier :math:`c` exported at
        node :math:`n` at time step :math:`t`


        """
        ### masks
        # not necessary

        ### index loop
        # This constraints does not have a central index loop, but multiple in the
        # auxiliary calculations

        ### auxiliary calculations
        # carrier flow transport technologies
        if zen_model.lp_model.variables["flow_transport"].size > 0:
            # recalculate all the edges
            edges_in = {
                node: energy_system.calculate_connected_edges(node, "in")
                for node in zen_model.sets["set_nodes"]
            }
            edges_out = {
                node: energy_system.calculate_connected_edges(node, "out")
                for node in zen_model.sets["set_nodes"]
            }
            max_edges = max(
                [len(edges_in[node]) for node in zen_model.sets["set_nodes"]]
                + [len(edges_out[node]) for node in zen_model.sets["set_nodes"]]
            )

            # create the variables
            flow_transport_in_vars = xr.DataArray(
                -1,
                coords=[
                    zen_model.parameters.demand.coords["set_carriers"],
                    zen_model.parameters.demand.coords["set_nodes"],
                    zen_model.parameters.demand.coords["set_time_steps_operation"],
                    xr.DataArray(
                        np.arange(
                            len(zen_model.sets["set_transport_technologies"])
                            * (2 * max_edges + 1)
                        ),
                        dims=["_term"],
                    ),
                ],
            )
            flow_transport_in_coeffs = xr.full_like(
                flow_transport_in_vars, np.nan, dtype=float
            )
            flow_transport_out_vars = flow_transport_in_vars.copy()
            flow_transport_out_coeffs = xr.full_like(
                flow_transport_in_vars, np.nan, dtype=float
            )
            for carrier, node in index.get_unique([0, 1]):
                techs = [
                    tech
                    for tech in zen_model.sets["set_transport_technologies"]
                    if carrier in zen_model.sets["set_reference_carriers"][tech]
                ]
                edges_in = energy_system.calculate_connected_edges(node, "in")
                edges_out = energy_system.calculate_connected_edges(node, "out")

                # get the variables for the in flow
                in_vars_plus = (
                    zen_model.lp_model.variables["flow_transport"]
                    .labels.loc[techs, edges_in, :]
                    .data
                )
                in_vars_plus = in_vars_plus.reshape((-1, in_vars_plus.shape[-1])).T
                in_coefs_plus = np.ones_like(in_vars_plus)
                in_vars_minus = (
                    zen_model.lp_model.variables["flow_transport_loss"]
                    .labels.loc[techs, edges_in, :]
                    .data
                )
                in_vars_minus = in_vars_minus.reshape((-1, in_vars_minus.shape[-1])).T
                in_coefs_minus = np.ones_like(in_vars_minus)
                in_vars = np.concatenate([in_vars_plus, in_vars_minus], axis=1)
                in_coefs = np.concatenate([in_coefs_plus, -in_coefs_minus], axis=1)
                flow_transport_in_vars.loc[
                    carrier, node, :, : in_vars.shape[-1] - 1
                ] = in_vars
                flow_transport_in_coeffs.loc[
                    carrier, node, :, : in_coefs.shape[-1] - 1
                ] = in_coefs

                # get the variables for the out flow
                out_vars_plus = (
                    zen_model.lp_model.variables["flow_transport"]
                    .labels.loc[techs, edges_out, :]
                    .data
                )
                out_vars_plus = out_vars_plus.reshape((-1, out_vars_plus.shape[-1])).T
                out_coefs_plus = np.ones_like(out_vars_plus)
                flow_transport_out_vars.loc[
                    carrier, node, :, : out_vars_plus.shape[-1] - 1
                ] = out_vars_plus
                flow_transport_out_coeffs.loc[
                    carrier, node, :, : out_coefs_plus.shape[-1] - 1
                ] = out_coefs_plus

            # craete the linear expression
            term_flow_transport_in = lp.LinearExpression(
                xr.Dataset(
                    {"coeffs": flow_transport_in_coeffs, "vars": flow_transport_in_vars}
                ),
                zen_model.lp_model,
            )
            term_flow_transport_out = lp.LinearExpression(
                xr.Dataset(
                    {
                        "coeffs": flow_transport_out_coeffs,
                        "vars": flow_transport_out_vars,
                    }
                ),
                zen_model.lp_model,
            )
        else:
            # if there is no carrier flow we just create empty arrays
            term_flow_transport_in = (
                zen_model.lp_model.variables["flow_import"]
                .where(xr.DataArray(False))
                .to_linexpr()
            )
            term_flow_transport_out = (
                zen_model.lp_model.variables["flow_import"]
                .where(xr.DataArray(False))
                .to_linexpr()
            )

        # carrier input and output conversion technologies
        term_carrier_conversion_in = []
        term_carrier_conversion_out = []
        nodes = list(zen_model.sets["set_nodes"])
        for carrier in index.get_unique([0]):
            techs_in = [
                tech
                for tech in zen_model.sets["set_conversion_technologies"]
                if carrier in zen_model.sets["set_input_carriers"][tech]
            ]
            # we need to catch emtpy lookups
            carrier_in = [carrier] if len(techs_in) > 0 else []
            techs_out = [
                tech
                for tech in zen_model.sets["set_conversion_technologies"]
                if carrier in zen_model.sets["set_output_carriers"][tech]
            ]
            # we need to catch emtpy lookups
            carrier_out = [carrier] if len(techs_out) > 0 else []
            term_carrier_conversion_in.append(
                zen_model.lp_model.variables["flow_conversion_input"]
                .loc[techs_in, carrier_in, nodes]
                .sum(zen_model.lp_model.variables["flow_conversion_input"].dims[:2])
            )
            term_carrier_conversion_out.append(
                zen_model.lp_model.variables["flow_conversion_output"]
                .loc[techs_out, carrier_out, nodes]
                .sum(zen_model.lp_model.variables["flow_conversion_output"].dims[:2])
            )
        # merge and regroup
        term_carrier_conversion_in = lp.merge(
            term_carrier_conversion_in, dim="group", join="outer", cls=LinearExpression
        )
        term_carrier_conversion_in = zen_model.constraints.reorder_group(
            term_carrier_conversion_in,
            None,
            None,
            index.get_unique([0]),
            first_index_name,
            zen_model.lp_model,
        )
        term_carrier_conversion_out = lp.merge(
            term_carrier_conversion_out, dim="group", join="outer", cls=LinearExpression
        )
        term_carrier_conversion_out = zen_model.constraints.reorder_group(
            term_carrier_conversion_out,
            None,
            None,
            index.get_unique([0]),
            first_index_name,
            zen_model.lp_model,
        )

        # carrier flow storage technologies
        if zen_model.lp_model.variables["flow_storage_discharge"].size > 0:
            term_flow_storage_discharge = []
            term_flow_storage_charge = []
            for carrier in index.get_unique([0]):
                storage_techs = [
                    tech
                    for tech in zen_model.sets["set_storage_technologies"]
                    if carrier in zen_model.sets["set_reference_carriers"][tech]
                ]
                term_flow_storage_discharge.append(
                    zen_model.lp_model.variables["flow_storage_discharge"]
                    .loc[storage_techs]
                    .sum("set_storage_technologies")
                )
                term_flow_storage_charge.append(
                    zen_model.lp_model.variables["flow_storage_charge"]
                    .loc[storage_techs]
                    .sum("set_storage_technologies")
                )
            # merge and regroup
            term_flow_storage_discharge = lp.merge(
                term_flow_storage_discharge,
                dim="group",
                join="outer",
                cls=LinearExpression,
            )
            term_flow_storage_discharge = zen_model.constraints.reorder_group(
                term_flow_storage_discharge,
                None,
                None,
                index.get_unique([0]),
                first_index_name,
                zen_model.lp_model,
            )
            term_flow_storage_charge = lp.merge(
                term_flow_storage_charge,
                dim="group",
                join="outer",
                cls=LinearExpression,
            )
            term_flow_storage_charge = zen_model.constraints.reorder_group(
                term_flow_storage_charge,
                None,
                None,
                index.get_unique([0]),
                first_index_name,
                zen_model.lp_model,
            )
        else:
            # if there is no carrier flow we just create empty arrays
            term_flow_storage_discharge = (
                zen_model.lp_model.variables["flow_import"]
                .where(xr.DataArray(False))
                .to_linexpr()
            )
            term_flow_storage_charge = (
                zen_model.lp_model.variables["flow_import"]
                .where(xr.DataArray(False))
                .to_linexpr()
            )

        # carrier import, demand and export
        term_carrier_import = zen_model.lp_model.variables["flow_import"].to_linexpr()
        term_carrier_export = zen_model.lp_model.variables["flow_export"].to_linexpr()
        term_carrier_demand = zen_model.parameters.demand
        # shed demand
        term_carrier_shed_demand = zen_model.lp_model.variables[
            "shed_demand"
        ].to_linexpr()

        ### formulate the constraints
        lhs = lp.merge(
            [
                term_carrier_conversion_out,
                -term_carrier_conversion_in,
                term_flow_transport_in,
                -term_flow_transport_out,
                -term_flow_storage_charge,
                term_flow_storage_discharge,
                term_carrier_import,
                -term_carrier_export,
                term_carrier_shed_demand,
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = term_carrier_demand
        aligned_idx = xr.align(lhs.coords, rhs, join="inner")[0]
        constraints = lhs.sel(aligned_idx) == rhs.sel(aligned_idx)

        ### return
        zen_model.constraints.add_constraint(
            "constraint_nodal_energy_balance", constraints
        )
