"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class defining compressable energy carriers.
                The class takes as inputs the abstract optimization model. The class adds parameters, variables and
                constraints of a generic carrier and returns the abstract optimization model.
==========================================================================================================================================================================="""
import logging

import pyomo.environ as pe
import numpy as np

from .carrier import Carrier


class ConditioningCarrier(Carrier):
    # set label
    label = "set_conditioning_carriers"

    def __init__(self, carrier: str, optimization_setup):
        """initialization of a generic carrier object
        :param carrier: carrier that is added to the model
        :param optimization_setup: The OptimizationSetup the element is part of """

        logging.info(f'Initialize conditioning carrier {carrier}')
        super().__init__(carrier, optimization_setup)
        # store input data
        self.store_input_data()

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().store_input_data()

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Carrier --- ###
    @classmethod
    def construct_vars(cls, optimization_setup):
        """ constructs the pe.Vars of the class <Carrier>
        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        variables = optimization_setup.variables

        # flow of imported carrier
        variables.add_variable(model, name="endogenous_carrier_demand", index_sets=cls.create_custom_set(["set_conditioning_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup), bounds=(0, np.inf),
                               doc="node- and time-dependent model endogenous carrier demand")

    @classmethod
    def construct_constraints(cls, optimization_setup):
        """ constructs the pe.Constraints of the class <Carrier>
        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        rules = ConditioningCarrierRules(optimization_setup)
        # limit import flow by availability
        optimization_setup.constraints.add_constraint(model, name="constraint_carrier_demand_coupling", index_sets=cls.create_custom_set(["set_conditioning_carrier_parents", "set_nodes", "set_time_steps_operation"], optimization_setup),
            rule=rules.constraint_carrier_demand_coupling_rule, doc='coupling model endogenous and exogenous carrier demand', )
        # overwrite energy balance when conditioning carriers are included
        model.constraint_nodal_energy_balance.deactivate()
        optimization_setup.constraints.add_constraint(model, name="constraint_nodal_energy_balance_conditioning", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup),
            rule=rules.constraint_nodal_energy_balance_conditioning_rule, doc='node- and time-dependent energy balance for each carrier', )


class ConditioningCarrierRules:
    """
    Rules for the ConditioningCarrier class
    """

    def __init__(self, optimization_setup):
        """
        Inits the rules for a given EnergySystem
        :param optimization_setup: The OptimizationSetup the element is part of
        """

        self.optimization_setup = optimization_setup
        self.energy_system = optimization_setup.energy_system

    def constraint_carrier_demand_coupling_rule(self, model, parent_carrier, node, time):
        """ sum conditioning Carriers"""

        return (model.endogenous_carrier_demand[parent_carrier, node, time] == sum(
            model.endogenous_carrier_demand[conditioning_carrier, node, time] for conditioning_carrier in model.set_conditioning_carrier_children[parent_carrier]))

    def constraint_nodal_energy_balance_conditioning_rule(self, model, carrier, node, time):
        """
        nodal energy balance for each time step.
        """
        params = self.optimization_setup.parameters

        # carrier input and output conversion technologies
        carrier_conversion_in, carrier_conversion_out = 0, 0
        for tech in model.set_conversion_technologies:
            if carrier in model.set_input_carriers[tech]:
                carrier_conversion_in += model.input_flow[tech, carrier, node, time]
            if carrier in model.set_output_carriers[tech]:
                carrier_conversion_out += model.output_flow[tech, carrier, node, time]
        # carrier flow transport technologies
        carrier_flow_in, carrier_flow_out = 0, 0
        set_edges_in = self.energy_system.calculate_connected_edges(node, "in")
        set_edges_out = self.energy_system.calculate_connected_edges(node, "out")
        for tech in model.set_transport_technologies:
            if carrier in model.set_reference_carriers[tech]:
                carrier_flow_in += sum(model.carrier_flow[tech, edge, time] - model.carrier_loss[tech, edge, time] for edge in set_edges_in)
                carrier_flow_out += sum(model.carrier_flow[tech, edge, time] for edge in set_edges_out)
        # carrier flow storage technologies
        carrier_flow_discharge, carrier_flow_charge = 0, 0
        for tech in model.set_storage_technologies:
            if carrier in model.set_reference_carriers[tech]:
                carrier_flow_discharge += model.carrier_flow_discharge[tech, node, time]
                carrier_flow_charge += model.carrier_flow_charge[tech, node, time]
        # carrier import, demand and export
        carrier_import, carrier_export, carrier_demand = 0, 0, 0
        carrier_import = model.import_carrier_flow[carrier, node, time]
        carrier_export = model.export_carrier_flow[carrier, node, time]
        carrier_demand = params.demand_carrier[carrier, node, time]
        endogenous_carrier_demand = 0

        # check if carrier is conditioning carrier:
        if carrier in model.set_conditioning_carriers:
            # check if carrier is parent_carrier of a conditioning_carrier
            if carrier in model.set_conditioning_carrier_parents:
                endogenous_carrier_demand = - model.endogenous_carrier_demand[carrier, node, time]
            else:
                endogenous_carrier_demand = model.endogenous_carrier_demand[carrier, node, time]

        return (# conversion technologies
                carrier_conversion_out - carrier_conversion_in # transport technologies
                + carrier_flow_in - carrier_flow_out # storage technologies
                + carrier_flow_discharge - carrier_flow_charge # import and export
                + carrier_import - carrier_export # demand
                - endogenous_carrier_demand - carrier_demand == 0)
