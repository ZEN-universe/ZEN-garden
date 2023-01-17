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
from ..energy_system import EnergySystem
from .carrier import Carrier
from ..component import Parameter, Variable, Constraint


class ConditioningCarrier(Carrier):
    # set label
    label = "set_conditioning_carriers"

    def __init__(self, carrier: str, energy_system: EnergySystem):
        """initialization of a generic carrier object
        :param carrier: carrier that is added to the model
        :param energy_system: The energy system the element is part of"""

        logging.info(f'Initialize conditioning carrier {carrier}')
        super().__init__(carrier, energy_system)
        # store input data
        self.store_input_data()

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().store_input_data()

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Carrier --- ###
    @classmethod
    def construct_vars(cls, energy_system: EnergySystem):
        """ constructs the pe.Vars of the class <Carrier>
        :param energy_system: The Energy system to add everything"""
        model = energy_system.pyomo_model

        # flow of imported carrier
        Variable.add_variable(model, name="endogenous_carrier_demand", index_sets=cls.create_custom_set(["set_conditioning_carriers", "set_nodes", "set_time_steps_operation"], energy_system),
            domain=pe.NonNegativeReals, doc='node- and time-dependent model endogenous carrier demand')

    @classmethod
    def construct_constraints(cls, energy_system: EnergySystem):
        """ constructs the pe.Constraints of the class <Carrier>
        :param energy_system: The Energy system to add everything"""
        model = energy_system.pyomo_model
        rules = ConditioningCarrierRules(energy_system)
        # limit import flow by availability
        Constraint.add_constraint(model, name="constraint_carrier_demand_coupling", index_sets=cls.create_custom_set(["set_conditioning_carrier_parents", "set_nodes", "set_time_steps_operation"], energy_system),
            rule=rules.constraint_carrier_demand_coupling_rule, doc='coupling model endogenous and exogenous carrier demand', )
        # overwrite energy balance when conditioning carriers are included
        model.constraint_nodal_energy_balance.deactivate()
        Constraint.add_constraint(model, name="constraint_nodal_energy_balance_conditioning", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], energy_system),
            rule=rules.constraint_nodal_energy_balance_conditioning_rule, doc='node- and time-dependent energy balance for each carrier', )


class ConditioningCarrierRules:
    """
    Rules for the ConditioningCarrier class
    """

    def __init__(self, energy_system: EnergySystem):
        """
        Inits the rules for a given EnergySystem
        :param energy_system: The EnergySystem
        """

        self.energy_system = energy_system

    def constraint_carrier_demand_coupling_rule(self, model, parentCarrier, node, time):
        """ sum conditioning Carriers"""

        return (model.endogenous_carrier_demand[parentCarrier, node, time] == sum(
            model.endogenous_carrier_demand[conditioning_carrier, node, time] for conditioning_carrier in model.set_conditioning_carrier_children[parentCarrier]))

    def constraint_nodal_energy_balance_conditioning_rule(self, model, carrier, node, time):
        """
        nodal energy balance for each time step.
        """
        params = self.energy_system.parameters

        # carrier input and output conversion technologies
        carrier_conversion_in, carrier_conversion_out = 0, 0
        for tech in model.set_conversion_technologies:
            if carrier in model.set_input_carriers[tech]:
                carrier_conversion_in += model.input_flow[tech, carrier, node, time]
            if carrier in model.set_output_carriers[tech]:
                carrier_conversion_out += model.output_flow[tech, carrier, node, time]
        # carrier flow transport technologies
        carrier_flow_in, carrier_flow_out = 0, 0
        set_edges_in = EnergySystem.calculate_connected_edges(node, "in")
        set_edges_out = EnergySystem.calculate_connected_edges(node, "out")
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
            # check if carrier is parentCarrier of a conditioning_carrier
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
