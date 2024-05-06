"""
:Title:          ZEN-GARDEN
:Created:        October-2021
:Authors:        Alissa Ganter (aganter@ethz.ch),
                Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Class defining a generic energy carrier.
The class takes as inputs the abstract optimization model. The class adds parameters, variables and
constraints of a generic carrier and returns the abstract optimization model.
"""
import logging

import linopy as lp
import numpy as np
import xarray as xr

from zen_garden.utils import lp_sum
from ..component import ZenIndex
from ..element import Element, GenericRule
import pandas as pd


class Carrier(Element):
    # set label
    label = "set_carriers"
    # empty list of elements
    list_of_elements = []

    def __init__(self, carrier: str, optimization_setup):
        """initialization of a generic carrier object

        :param carrier: carrier that is added to the model
        :param optimization_setup: The OptimizationSetup the element is part of """
        super().__init__(carrier, optimization_setup)

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # store scenario dict
        super().store_scenario_dict()
        # set attributes of carrier
        # raw import
        self.raw_time_series = {}
        self.raw_time_series["demand"] = self.data_input.extract_input_data("demand", index_sets=["set_nodes", "set_time_steps"], time_steps="set_base_time_steps_yearly", unit_category={"energy_quantity": 1, "time": -1})
        self.raw_time_series["availability_import"] = self.data_input.extract_input_data("availability_import", index_sets=["set_nodes", "set_time_steps"], time_steps="set_base_time_steps_yearly", unit_category={"energy_quantity": 1, "time": -1})
        self.raw_time_series["availability_export"] = self.data_input.extract_input_data("availability_export", index_sets=["set_nodes", "set_time_steps"], time_steps="set_base_time_steps_yearly", unit_category={"energy_quantity": 1, "time": -1})
        self.raw_time_series["price_export"] = self.data_input.extract_input_data("price_export", index_sets=["set_nodes", "set_time_steps"], time_steps="set_base_time_steps_yearly", unit_category={"money": 1, "energy_quantity": -1})
        self.raw_time_series["price_import"] = self.data_input.extract_input_data("price_import", index_sets=["set_nodes", "set_time_steps"], time_steps="set_base_time_steps_yearly", unit_category={"money": 1, "energy_quantity": -1})
        # non-time series input data
        self.availability_import_yearly = self.data_input.extract_input_data("availability_import_yearly", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={"energy_quantity": 1})
        self.availability_export_yearly = self.data_input.extract_input_data("availability_export_yearly", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={"energy_quantity": 1})
        self.carbon_intensity_carrier = self.data_input.extract_input_data("carbon_intensity_carrier", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={"emissions": 1, "energy_quantity": -1})
        self.price_shed_demand = self.data_input.extract_input_data("price_shed_demand", index_sets=[], unit_category={"money": 1, "energy_quantity": -1})

    def overwrite_time_steps(self, base_time_steps):
        """ overwrites set_time_steps_operation

        :param base_time_steps: #TODO describe parameter/return
        """
        set_time_steps_operation = self.energy_system.time_steps.encode_time_step(base_time_steps=base_time_steps, time_step_type="operation")
        setattr(self, "set_time_steps_operation", set_time_steps_operation.squeeze().tolist())

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Carrier --- ###
    @classmethod
    def construct_sets(cls, optimization_setup):
        """ constructs the pe.Sets of the class <Carrier>

        :param optimization_setup: The OptimizationSetup the element is part of """
        pass

    @classmethod
    def construct_params(cls, optimization_setup):
        """ constructs the pe.Params of the class <Carrier>

        :param optimization_setup: The OptimizationSetup the element is part of """
        # demand of carrier
        optimization_setup.parameters.add_parameter(name="demand", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"], doc='Parameter which specifies the carrier demand', calling_class=cls)
        # availability of carrier
        optimization_setup.parameters.add_parameter(name="availability_import", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"], doc='Parameter which specifies the maximum energy that can be imported from outside the system boundaries', calling_class=cls)
        # availability of carrier
        optimization_setup.parameters.add_parameter(name="availability_export", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"], doc='Parameter which specifies the maximum energy that can be exported to outside the system boundaries', calling_class=cls)
        # availability of carrier
        optimization_setup.parameters.add_parameter(name="availability_import_yearly", index_names=["set_carriers", "set_nodes", "set_time_steps_yearly"], doc='Parameter which specifies the maximum energy that can be imported from outside the system boundaries for the entire year', calling_class=cls)
        # availability of carrier
        optimization_setup.parameters.add_parameter(name="availability_export_yearly", index_names=["set_carriers", "set_nodes", "set_time_steps_yearly"], doc='Parameter which specifies the maximum energy that can be exported to outside the system boundaries for the entire year', calling_class=cls)
        # import price
        optimization_setup.parameters.add_parameter(name="price_import", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"], doc='Parameter which specifies the import carrier price', calling_class=cls)
        # export price
        optimization_setup.parameters.add_parameter(name="price_export", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"], doc='Parameter which specifies the export carrier price', calling_class=cls)
        # demand shedding price
        optimization_setup.parameters.add_parameter(name="price_shed_demand", index_names=["set_carriers"], doc='Parameter which specifies the price to shed demand', calling_class=cls)
        # carbon intensity
        optimization_setup.parameters.add_parameter(name="carbon_intensity_carrier", index_names=["set_carriers", "set_nodes", "set_time_steps_yearly"], doc='Parameter which specifies the carbon intensity of carrier', calling_class=cls)

    @classmethod
    def construct_vars(cls, optimization_setup):
        """ constructs the pe.Vars of the class <Carrier>

        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        variables = optimization_setup.variables
        sets = optimization_setup.sets

        # flow of imported carrier
        variables.add_variable(model, name="flow_import", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup), bounds=(0,np.inf),
                               doc="node- and time-dependent carrier import from the grid", unit_category={"energy_quantity": 1, "time": -1})
        # flow of exported carrier
        variables.add_variable(model, name="flow_export", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup), bounds=(0,np.inf),
                               doc="node- and time-dependent carrier export from the grid", unit_category={"energy_quantity": 1, "time": -1})
        # flow of exported carrier on average
        variables.add_variable(model, name="flow_export_balancing", index_sets=cls.create_custom_set(["set_carriers", "set_nodes"], optimization_setup), bounds=(0, np.inf),
                               doc="node-dependent carrier export per balancing period to the grid",
                               unit_category={"energy_quantity": 1, "time": -1})
        # carrier import/export cost
        variables.add_variable(model, name="cost_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup),
                               doc="node- and time-dependent carrier cost due to import and export", unit_category={"money": 1, "time": -1})
        # total carrier import/export cost
        variables.add_variable(model, name="cost_carrier_total", index_sets=sets["set_time_steps_yearly"],
                               doc="total carrier cost due to import and export", unit_category={"money": 1})
        # carbon emissions
        variables.add_variable(model, name="carbon_emissions_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup),
                               doc="carbon emissions of importing and exporting carrier", unit_category={"emissions": 1, "time": -1})
        # carbon emissions carrier
        variables.add_variable(model, name="carbon_emissions_carrier_total", index_sets=sets["set_time_steps_yearly"],
                               doc="total carbon emissions of importing and exporting carrier", unit_category={"emissions": 1})
        # shed demand
        variables.add_variable(model, name="shed_demand", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup), bounds=(0,np.inf),
                               doc="shed demand of carrier", unit_category={"energy_quantity": 1, "time": -1})
        # cost of shed demand
        variables.add_variable(model, name="cost_shed_demand", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup), bounds=(0,np.inf),
                               doc="shed demand of carrier", unit_category={"money": 1, "time": -1})

        # add pe.Sets of the child classes
        for subclass in cls.__subclasses__():
            if np.size(optimization_setup.system[subclass.label]):
                subclass.construct_vars(optimization_setup)

    @classmethod
    def construct_constraints(cls, optimization_setup):
        """ constructs the pe.Constraints of the class <Carrier>

        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        constraints = optimization_setup.constraints
        sets = optimization_setup.sets
        rules = CarrierRules(optimization_setup)
        # limit import flow by availability
        constraints.add_constraint_block(model, name="constraint_availability_import",
                                         constraint=rules.constraint_availability_import_block(),
                                         doc='node- and time-dependent carrier availability to import from outside the system boundaries', )
        # limit export flow by availability
        constraints.add_constraint_block(model, name="constraint_availability_export",
                                         constraint=rules.constraint_availability_export_block(),
                                         doc='node- and time-dependent carrier availability to export to outside the system boundaries')
        # limit export flow to constant export flow over time
        constraints.add_constraint_block(model, name="constraint_flow_export_balancing",
                                         constraint=rules.constraint_flow_export_balancing_block(),
                                         doc='node- and time-dependent carrier export to outside the system boundaries has to be constant')
        # compute daily exports
        #constraints.add_constraint_block(model, name="constraint_flow_export_balancing_period",
        #                                 constraint=rules.constraint_flow_export_balancing_period_block(),
        #                                 doc='node- and time-dependent carrier export to outside the system boundaries summed over entire day')
        # limit import flow by availability for each year
        constraints.add_constraint_block(model, name="constraint_availability_import_yearly",
                                         constraint=rules.constraint_availability_import_yearly_block(),
                                         doc='node- and time-dependent carrier availability to import from outside the system boundaries summed over entire year', )
        # limit export flow by availability for each year
        constraints.add_constraint_block(model, name="constraint_availability_export_yearly",
                                         constraint=rules.constraint_availability_export_yearly_block(),
                                         doc='node- and time-dependent carrier availability to export to outside the system boundaries summed over entire year', )
        # cost for carrier
        constraints.add_constraint_block(model, name="constraint_cost_carrier",
                                         constraint=rules.constraint_cost_carrier_block(),
                                         doc="cost of importing and exporting carrier")
        # cost for shed demand
        constraints.add_constraint_block(model, name="constraint_cost_shed_demand",
                                         constraint=rules.constraint_cost_shed_demand_block(),
                                         doc="cost of shedding carrier demand")
        # limit of shed demand
        constraints.add_constraint_block(model, name="constraint_limit_shed_demand",
                                         constraint=rules.constraint_limit_shed_demand_block(),
                                         doc="limit of shedding carrier demand")
        # total cost for carriers
        constraints.add_constraint_rule(model, name="constraint_cost_carrier_total",
                                        index_sets=sets["set_time_steps_yearly"], rule=rules.constraint_cost_carrier_total_rule,
                                        doc="total cost of importing and exporting carriers")
        # carbon emissions
        constraints.add_constraint_block(model, name="constraint_carbon_emissions_carrier",
                                         constraint=rules.constraint_carbon_emissions_carrier_block(),
                                         doc="carbon emissions of importing and exporting carrier")
        # carbon emissions carrier
        constraints.add_constraint_rule(model, name="constraint_carbon_emissions_carrier_total",
                                        index_sets=sets["set_time_steps_yearly"], rule=rules.constraint_carbon_emissions_carrier_total_rule,
            doc="total carbon emissions of importing and exporting carriers")
        # energy balance
        constraints.add_constraint_block(model, name="constraint_nodal_energy_balance",
                                         constraint=rules.constraint_nodal_energy_balance_block(),
                                         doc='node- and time-dependent energy balance for each carrier', )

        # add pe.Sets of the child classes
        for subclass in cls.__subclasses__():
            if len(optimization_setup.system[subclass.label]) > 0:
                subclass.construct_constraints(optimization_setup)


class CarrierRules(GenericRule):
    """
    Rules for the Carrier class
    """

    def __init__(self, optimization_setup):
        """
        Inits the rules for a given EnergySystem

        :param optimization_setup: The OptimizationSetup the element is part of
        """

        super().__init__(optimization_setup)

    # Rule-based constraints
    # ----------------------

    def constraint_cost_carrier_total_rule(self, year):
        """ total cost of importing and exporting carrier

        .. math::
            C_y^{\mathcal{C}} = \sum_{c\in\mathcal{C}}\sum_{n\in\mathcal{N}}\sum_{t\in\mathcal{T}} \\tau_t (C_{c,n,t} + C_{c,n,t}^{\mathrm{shed}\ \mathrm{demand}})

        :param year: #TODO describe parameter/return
        :return: #TODO describe parameter/return
        """

        ### index sets
        # skipped because rule-based constraint

        ### masks
        # skipped because rule-based constraint

        ### index loop
        # skipped because rule-based constraint

        ### auxiliary calculations
        terms = []
        # This vectorizes over times and locations
        for carrier in self.sets["set_carriers"]:
            times = self.energy_system.time_steps.get_time_steps_year2operation(year)
            expr = (self.variables["cost_carrier"].loc[carrier, :, times]
                    + self.variables["cost_shed_demand"].loc[carrier, :, times]) * self.parameters.time_steps_operation_duration.loc[times]
            terms.append(expr.sum())
        term_summed_carrier_shed_demand_costs = lp_sum(terms)

        ### formulate constraint
        lhs = self.variables["cost_carrier_total"].loc[year] - term_summed_carrier_shed_demand_costs
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)

    def constraint_carbon_emissions_carrier_total_rule(self, year):
        """ total carbon emissions of importing and exporting carrier

        .. math::
            E_y^{\mathcal{C}} = \sum_{c\in\mathcal{C}}\sum_{n\in\mathcal{N}}\sum_{t\in\mathcal{T}} \\tau_t E_{c,n,t}

        :param model: #TODO describe parameter/return
        :param year: #TODO describe parameter/return
        """

        ### index sets
        # skipped because rule-based constraint

        ### masks
        # skipped because rule-based constraint

        ### index loop
        # skipped because rule-based constraint

        ### auxiliary calculations
        terms = []
        # This vectorizes over times and locations
        for carrier in self.sets["set_carriers"]:
            times = self.energy_system.time_steps.get_time_steps_year2operation(year)
            expr = self.variables["carbon_emissions_carrier"].loc[carrier, :, times] * self.parameters.time_steps_operation_duration.loc[times]
            terms.append(expr.sum())
        term_summed_carbon_emissions_carrier = lp_sum(terms)

        ### formulate constraint
        lhs = self.variables["carbon_emissions_carrier_total"].loc[year] - term_summed_carbon_emissions_carrier
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)

    # Block-based constraints
    # -----------------------

    def constraint_availability_import_block(self):
        """node- and time-dependent carrier availability to import from outside the system boundaries

        .. math::
            U_{c,n,t} \\leq a_{c,n,t}^\mathrm{import}

        :return: #TODO describe parameter/return
        """

        ### index sets
        # not necessary

        ### masks
        # not necessary

        ### index loop
        # not necessary

        ### auxiliary calculations
        # not necessary

        ### formulate constraint
        lhs = self.variables["flow_import"]
        rhs = self.parameters.availability_import
        constraints = lhs <= rhs

        ### return
        return self.constraints.return_contraints(constraints)

    def constraint_availability_export_block(self):
        """node- and time-dependent carrier availability to export to outside the system boundaries

        .. math::
           V_{c,n,t} \\leq a_{c,n,t}^\mathrm{export}

        :return: #TODO describe parameter/return
        """

        ### index sets
        # not necessary

        ### masks
        # not necessary

        ### index loop
        # not necessary

        ### auxiliary calculations
        # not necessary

        ### formulate constraint
        lhs = self.variables["flow_export"]
        rhs = self.parameters.availability_export
        constraints = lhs <= rhs

        ### return
        return self.constraints.return_contraints(constraints)

    def constraint_flow_export_balancing_block(self):
        """node- and time-dependent carrier export has to be constant over time

        .. math::
           V_{c,n,t} = V^{avg}_{c,n}

        :return: #TODO describe parameter/return
        """
        ## skip constriant formulation if offtake carriers empty
        balancing_carriers = self.system["balancing_carriers"]
        skip_balancing = self.system["balancing_period"] == self.system["unaggregated_time_steps_per_year"]
        if not balancing_carriers or skip_balancing:
            return self.constraints.return_contraints([])

        ### index sets
        # not necessary

        ### masks
        # The constraints is only bounded for the carriers specifed in analysis
        mask = xr.DataArray(0, coords=[self.sets["set_carriers"]], dims=["set_carriers"])
        for c in balancing_carriers:
            if c in self.sets["set_carriers"]:
                mask.loc[c] = 1
            else:
                logging.warning(f"Carrier {c} is not part of the model")

        ### index loop
        # not necessary

        ### formulate constraint
        flow_export = self.variables["flow_export"]
        ts_per_balancing_period = self.system["balancing_period"]
        if ts_per_balancing_period==1:
            lhs = (flow_export - self.variables["flow_export_balancing"]).where(mask)            # special case hourly balancing
        elif ts_per_balancing_period>1:
            ts_set = "set_time_steps_operation"
            ts_operation = np.array(self.sets[ts_set].data)
            group_key = xr.DataArray(ts_operation // ts_per_balancing_period, coords=[ts_operation], dims=ts_set)
            lhs = (flow_export.groupby(group_key).sum() - self.variables["flow_export_balancing"]).where(mask)
        else:
            raise ValueError("Balancing period must be at least 1")
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)

    def constraint_flow_export_balancing_period_block(self):
        """node- and time-dependent carrier export has to be constant over time

        .. math::
           V_{c,n,t} = V^{avg}_{c,n}

        :return: #TODO describe parameter/return
        """
        balancing_carriers = self.system["balancing_carriers"]

        ### index sets
        # not necessary

        if not balancing_carriers:
            return self.constraints.return_contraints([])
        if self.system["conduct_time_series_aggregation"]:
            raise NotImplementedError("Balancing only implemented for unaggregated timeseries")

        ### masks
        # The constraints is only bounded for the carriers specifed in analysis
        mask = xr.DataArray(0, coords=self.variables["flow_export_balancing_period"].coords)
        for c in balancing_carriers:
            if c in self.sets["set_carriers"]:
                mask.loc[c, :, :] = 1
            else:
                logging.warning(f"Carrier {c} is not part of the model")

        ### index loop
        # not necessary

        ### formulate constraint
        flow_export = self.variables["flow_export"]
        ts_per_balancing_period = self.system["balancing_period"]
        ts_set = "set_time_steps_operation"
        ts_operation = np.array(self.sets[ts_set].data)
        group_key = xr.DataArray(ts_operation // ts_per_balancing_period, coords=[ts_operation], dims=ts_set)

        lhs = (flow_export.groupby(group_key).sum() - self.variables["flow_export_balancing_period"])*mask
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=self.sets["set_time_steps_balancing_period"],
                                                  index_names=["set_time_steps_balancing_period"])

    def constraint_availability_import_yearly_block(self):
        """node- and year-dependent carrier availability to import from outside the system boundaries

         .. math::
            a_{c,n,y}^\mathrm{import} \geq \\sum_{t\\in\mathcal{T}}\\tau_t U_{c,n,t}

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_yearly"],
                                                              self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # The constraints is only bounded if the availability is finite
        mask = self.parameters.availability_import_yearly != np.inf

        ### index loop
        # this loop vectorizes over the nodes
        constraints = []
        for carrier, year in index.get_unique(levels=["set_carriers", "set_time_steps_yearly"]):
            ### auxiliary calculations
            operational_time_steps = self.time_steps.get_time_steps_year2operation(year)
            term_summed_import_flow = (self.variables["flow_import"].loc[carrier, :, operational_time_steps]
                                       * self.parameters.time_steps_operation_duration.loc[operational_time_steps]).sum("set_time_steps_operation")

            ### formulate constraint
            lhs = term_summed_import_flow
            rhs = self.parameters.availability_import_yearly.loc[carrier, :, year]
            constraints.append(lhs <= rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  mask=mask,
                                                  index_values=index.get_unique(levels=["set_carriers", "set_time_steps_yearly"]),
                                                  index_names=["set_carriers", "set_time_steps_yearly"])

    def constraint_availability_export_yearly_block(self):
        """node- and year-dependent carrier availability to export to outside the system boundaries

        .. math::
           a_{c,n,y}^\mathrm{export} \geq \\sum_{t\\in\mathcal{T}}\\tau_t V_{c,n,t}

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # The constraints is only bounded if the availability is finite
        mask = self.parameters.availability_export_yearly != np.inf

        ### index loop
        # this loop vectorizes over the nodes
        constraints = []
        for carrier, year in index.get_unique(levels=["set_carriers", "set_time_steps_yearly"]):
            ### auxiliary calculations
            operational_time_steps = self.time_steps.get_time_steps_year2operation(year)
            term_summed_export_flow = (self.variables["flow_export"].loc[carrier, :, operational_time_steps]
                                       * self.parameters.time_steps_operation_duration.loc[operational_time_steps]).sum("set_time_steps_operation")

            ### formulate constraint
            lhs = term_summed_export_flow
            rhs = self.parameters.availability_export_yearly.loc[carrier, :, year]
            constraints.append(lhs <= rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  mask=mask,
                                                  index_values=index.get_unique(levels=["set_carriers", "set_time_steps_yearly"]),
                                                  index_names=["set_carriers", "set_time_steps_yearly"])

    def constraint_cost_carrier_block(self):
        """ cost of importing and exporting carrier

        .. math::
           C_{c,n,t} = u_{c,n,t} U_{c,n,t} - v_{c,n,t} V_{c,n,t}

        :return: #TODO describe parameter/return
        """

        ### index sets
        # not necessary

        ### masks
        # we distinguish the cases where there is availability to import or export and where there is not
        mask = ((self.parameters.availability_import != 0) | (self.parameters.availability_export != 0))
        #mask = xr.align(self.variables.labels, mask)[1]

        ### index loop
        # not necessary

        ### auxiliary calculations
        # not necessary

        ### formulate constraint
        lhs = (self.variables["cost_carrier"]
               # these terms are only necessary if import or export is available
               - self.parameters.price_import.where(mask) * self.variables["flow_import"].where(mask)
               + self.parameters.price_export.where(mask) * self.variables["flow_export"].where(mask))
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)

    def constraint_cost_shed_demand_block(self):
        """ cost of shedding demand of carrier

        .. math::
           C_{c,n,t}^{\mathrm{shed}\ \mathrm{demand}} = D_{c,n,t} \\nu_c

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over the carriers, because the constraint depends on the carrier, vectorization over the nodes and time steps
        constraints = []
        for carrier in index.get_unique(["set_carriers"]):
            ### auxiliary calculations
            # not necessary

            ### formulate constraint
            if self.parameters.price_shed_demand.loc[carrier] != np.inf:
                lhs = (self.variables["cost_shed_demand"].loc[carrier]
                       - self.parameters.price_shed_demand.loc[carrier] * self.variables["shed_demand"].loc[carrier])
                rhs = 0
                constraints.append(lhs == rhs)
            else:
                lhs = self.variables["shed_demand"].loc[carrier]
                rhs = 0
                constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_carriers"]),
                                                  index_names=["set_carriers"])

    def constraint_limit_shed_demand_block(self):
        """ limit demand shedding

        .. math::
           D_{c,n,t} \leq d_{c,n,t}

        :return: #TODO describe parameter/return
        """

        ### index sets
        # not necessary

        ### masks
        # not necessary

        ### index loop
        # not necessary

        ### auxiliary calculations
        # not necessary

        ### formulate constraint
        lhs = self.variables["shed_demand"]
        rhs = self.parameters.demand
        constraints = lhs <= rhs

        ### return
        return self.constraints.return_contraints(constraints)

    def constraint_carbon_emissions_carrier_block(self):
        """ carbon emissions of importing and exporting carrier

        .. math::
           E_{c,n,t} = \\epsilon_c (U_{c,n,t} - V_{c,n,t})

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)
        times = index.get_unique(["set_time_steps_operation"])

        ### masks
        # not necessary

        ### index loop
        # we loop over the carriers and vectorize over the nodes and over the times after converting them from operation to yearly time steps
        constraints = []
        for carrier in index.get_unique(["set_carriers"]):
            ### auxiliary calculations
            yearly_time_steps = [self.time_steps.convert_time_step_operation2year(t) for t in times]

            # get the time-dependent factor
            mask = (self.parameters.availability_import.loc[carrier, :, times] != 0) | (self.parameters.availability_export.loc[carrier, :, times] != 0)
            fac = np.where(mask, self.parameters.carbon_intensity_carrier.loc[carrier, :, yearly_time_steps], 0)
            fac = xr.DataArray(fac, coords=[self.variables.coords["set_nodes"], self.variables.coords["set_time_steps_operation"]])
            term_flow_import_export = fac * (self.variables["flow_import"].loc[carrier, :]) - self.variables["flow_export"].loc[carrier, :]

            ### formulate constraint
            lhs = (self.variables["carbon_emissions_carrier"].loc[carrier, :]
                   - term_flow_import_export)
            rhs = 0
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_carriers"]),
                                                  index_names=["set_carriers"])


    def constraint_nodal_energy_balance_block(self):
        """
        nodal energy balance for each time step

        .. math::
            0 = -(d_{c,n,t}-D_{c,n,t})
            + \\sum_{i\\in\mathcal{I}}(\\overline{G}_{c,i,n,t}-\\underline{G}_{c,i,n,t})
            + \\sum_{j\\in\mathcal{J}}\\sum_{e\\in\\underline{\mathcal{E}}}F_{j,e,t}-F^\mathrm{l}_{j,e,t})-\\sum_{e'\\in\\overline{\mathcal{E}}}F_{j,e',t})
            + \\sum_{k\\in\mathcal{K}}(\\overline{H}_{k,n,t}-\\underline{H}_{k,n,t})
            + U_{c,n,t} - V_{c,n,t}

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Carrier.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # This constraints does not have a central index loop, but multiple in the auxiliary calculations

        ### auxiliary calculations
        # carrier flow transport technologies
        if self.variables["flow_transport"].size > 0:
            # recalculate all the edges
            edges_in = {node: self.energy_system.calculate_connected_edges(node, "in") for node in self.sets["set_nodes"]}
            edges_out = {node: self.energy_system.calculate_connected_edges(node, "out") for node in self.sets["set_nodes"]}
            max_edges = max([len(edges_in[node]) for node in self.sets["set_nodes"]] + [len(edges_out[node]) for node in
                                                                                   self.sets["set_nodes"]])

            # create the variables
            flow_transport_in_vars = xr.DataArray(-1, coords=[self.variables.coords["set_carriers"],
                                                              self.variables.coords["set_nodes"],
                                                              self.variables.coords["set_time_steps_operation"],
                                                              xr.DataArray(np.arange(
                                                                  len(self.sets["set_transport_technologies"]) * (
                                                                          2 * max_edges + 1)), dims=["_term"])])
            flow_transport_in_coeffs = xr.full_like(flow_transport_in_vars, np.nan, dtype=float)
            flow_transport_out_vars = flow_transport_in_vars.copy()
            flow_transport_out_coeffs = xr.full_like(flow_transport_in_vars, np.nan, dtype=float)
            for carrier, node in index.get_unique([0, 1]):
                techs = [tech for tech in self.sets["set_transport_technologies"] if
                         carrier in self.sets["set_reference_carriers"][tech]]
                edges_in = self.energy_system.calculate_connected_edges(node, "in")
                edges_out = self.energy_system.calculate_connected_edges(node, "out")

                # get the variables for the in flow
                in_vars_plus = self.variables["flow_transport"].labels.loc[techs, edges_in, :].data
                in_vars_plus = in_vars_plus.reshape((-1, in_vars_plus.shape[-1])).T
                in_coefs_plus = np.ones_like(in_vars_plus)
                in_vars_minus = self.variables["flow_transport_loss"].labels.loc[techs, edges_out, :].data
                in_vars_minus = in_vars_minus.reshape((-1, in_vars_minus.shape[-1])).T
                in_coefs_minus = np.ones_like(in_vars_minus)
                in_vars = np.concatenate([in_vars_plus, in_vars_minus], axis=1)
                in_coefs = np.concatenate([in_coefs_plus, -in_coefs_minus], axis=1)
                flow_transport_in_vars.loc[carrier, node, :, :in_vars.shape[-1] - 1] = in_vars
                flow_transport_in_coeffs.loc[carrier, node, :, :in_coefs.shape[-1] - 1] = in_coefs

                # get the variables for the out flow
                out_vars_plus = self.variables["flow_transport"].labels.loc[techs, edges_out, :].data
                out_vars_plus = out_vars_plus.reshape((-1, out_vars_plus.shape[-1])).T
                out_coefs_plus = np.ones_like(out_vars_plus)
                flow_transport_out_vars.loc[carrier, node, :, :out_vars_plus.shape[-1] - 1] = out_vars_plus
                flow_transport_out_coeffs.loc[carrier, node, :, :out_coefs_plus.shape[-1] - 1] = out_coefs_plus

            # craete the linear expression
            term_flow_transport_in = lp.LinearExpression(xr.Dataset({"coeffs": flow_transport_in_coeffs,
                                                                "vars": flow_transport_in_vars}), self.model)
            term_flow_transport_out = lp.LinearExpression(xr.Dataset({"coeffs": flow_transport_out_coeffs,
                                                                 "vars": flow_transport_out_vars}), self.model)
        else:
            # if there is no carrier flow we just create empty arrays
            term_flow_transport_in = self.variables["flow_import"].where(False).to_linexpr()
            term_flow_transport_out = self.variables["flow_import"].where(False).to_linexpr()

        # carrier input and output conversion technologies
        term_carrier_conversion_in = []
        term_carrier_conversion_out = []
        nodes = list(self.sets["set_nodes"])
        for carrier in index.get_unique([0]):
            techs_in = [tech for tech in self.sets["set_conversion_technologies"] if
                        carrier in self.sets["set_input_carriers"][tech]]
            # we need to catch emtpy lookups
            carrier_in = [carrier] if len(techs_in) > 0 else []
            techs_out = [tech for tech in self.sets["set_conversion_technologies"] if
                         carrier in self.sets["set_output_carriers"][tech]]
            # we need to catch emtpy lookups
            carrier_out = [carrier] if len(techs_out) > 0 else []
            term_carrier_conversion_in.append(self.variables["flow_conversion_input"].loc[techs_in, carrier_in, nodes].sum(
                self.variables["flow_conversion_input"].dims[:2]))
            term_carrier_conversion_out.append(
                self.variables["flow_conversion_output"].loc[techs_out, carrier_out, nodes].sum(
                    self.variables["flow_conversion_output"].dims[:2]))
        # merge and regroup
        term_carrier_conversion_in = lp.merge(*term_carrier_conversion_in, dim="group")
        term_carrier_conversion_in = self.optimization_setup.constraints.reorder_group(term_carrier_conversion_in, None, None,
                                                                                  index.get_unique([0]),
                                                                                  index_names[:1], self.model)
        term_carrier_conversion_out = lp.merge(*term_carrier_conversion_out, dim="group")
        term_carrier_conversion_out = self.optimization_setup.constraints.reorder_group(term_carrier_conversion_out, None, None,
                                                                                   index.get_unique([0]),
                                                                                   index_names[:1], self.model)

        # carrier flow storage technologies
        if self.variables["flow_storage_discharge"].size > 0:
            term_flow_storage_discharge = []
            term_flow_storage_charge = []
            for carrier in index.get_unique([0]):
                storage_techs = [tech for tech in self.sets["set_storage_technologies"] if
                                 carrier in self.sets["set_reference_carriers"][tech]]
                term_flow_storage_discharge.append(
                    self.variables["flow_storage_discharge"].loc[storage_techs].sum("set_storage_technologies"))
                term_flow_storage_charge.append(
                    self.variables["flow_storage_charge"].loc[storage_techs].sum("set_storage_technologies"))
            # merge and regroup
            term_flow_storage_discharge = lp.merge(*term_flow_storage_discharge, dim="group")
            term_flow_storage_discharge = self.optimization_setup.constraints.reorder_group(term_flow_storage_discharge, None,
                                                                                       None, index.get_unique([0]),
                                                                                       index_names[:1], self.model)
            term_flow_storage_charge = lp.merge(*term_flow_storage_charge, dim="group")
            term_flow_storage_charge = self.optimization_setup.constraints.reorder_group(term_flow_storage_charge, None, None,
                                                                                    index.get_unique([0]),
                                                                                    index_names[:1], self.model)
        else:
            # if there is no carrier flow we just create empty arrays
            term_flow_storage_discharge = self.variables["flow_import"].where(False).to_linexpr()
            term_flow_storage_charge = self.variables["flow_import"].where(False).to_linexpr()

        # carrier import, demand and export
        term_carrier_import = self.variables["flow_import"].to_linexpr()
        term_carrier_export = self.variables["flow_export"].to_linexpr()
        term_carrier_demand = self.parameters.demand
        # shed demand
        term_carrier_shed_demand = self.variables["shed_demand"].to_linexpr()

        ### formulate the constraints
        lhs = lp.merge(term_carrier_conversion_out,
                       -term_carrier_conversion_in,
                       term_flow_transport_in,
                       -term_flow_transport_out,
                       -term_flow_storage_charge,
                       term_flow_storage_discharge,
                       term_carrier_import,
                       -term_carrier_export,
                       term_carrier_shed_demand,
                       compat="broadcast_equals")
        rhs = term_carrier_demand
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)
