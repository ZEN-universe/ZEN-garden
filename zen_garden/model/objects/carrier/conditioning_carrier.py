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
import pyomo.environ            as pe
from ..energy_system        import EnergySystem
from .carrier               import Carrier
from ..component            import Parameter,Variable,Constraint

class ConditioningCarrier(Carrier):
    # set label
    label = "set_conditioning_carriers"
    # empty list of elements
    list_of_elements = []

    def __init__(self,carrier):
        """initialization of a generic carrier object
        :param carrier: carrier that is added to the model"""

        logging.info(f'Initialize conditioning carrier {carrier}')
        super().__init__(carrier)
        # store input data
        self.store_input_data()
        # add carrier to list
        ConditioningCarrier.add_element(self)

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().store_input_data()

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Carrier --- ###
    @classmethod
    def construct_vars(cls):
        """ constructs the pe.Vars of the class <Carrier> """
        model = EnergySystem.get_pyomo_model()
        
        # flow of imported carrier
        Variable.add_variable(
            model,
            name="endogenous_carrier_demand",
            index_sets= cls.create_custom_set(["set_conditioning_carriers","set_nodes","set_time_steps_operation"]),
            domain = pe.NonNegativeReals,
            doc = 'node- and time-dependent model endogenous carrier demand'
        )

    @classmethod
    def construct_constraints(cls):
        """ constructs the pe.Constraints of the class <Carrier> """
        model = EnergySystem.get_pyomo_model()

        # limit import flow by availability
        Constraint.add_constraint(
            model,
            name="constraint_carrier_demand_coupling",
            index_sets= cls.create_custom_set(["set_conditioning_carrier_parents","set_nodes","set_time_steps_operation"]),
            rule = constraint_carrier_demand_coupling_rule,
            doc = 'coupling model endogenous and exogenous carrier demand',
        )
        # overwrite energy balance when conditioning carriers are included
        model.constraint_nodal_energy_balance.deactivate()
        Constraint.add_constraint(
            model,
            name="constraint_nodal_energy_balance_conditioning",
            index_sets= cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]),
            rule=constraint_nodal_energy_balance_conditioning_rule,
            doc='node- and time-dependent energy balance for each carrier',
        )

def constraint_carrier_demand_coupling_rule(model, parentCarrier, node, time):
    """ sum conditioning Carriers"""

    return(model.endogenous_carrier_demand[parentCarrier,node,time] ==
           sum(model.endogenous_carrier_demand[conditioning_carrier,node,time]
                 for conditioning_carrier in model.set_conditioning_carrier_children[parentCarrier]))

def constraint_nodal_energy_balance_conditioning_rule(model, carrier, node, time):
    """" 
    nodal energy balance for each time step. 
    The constraint is indexed by existing_invested_capacityimeStepsCarrier, which is union of time step sequences of all corresponding technologies and carriers
    timeStepEnergyBalance --> base_time_step --> element_time_step
    """
    params = Parameter.get_component_object()

    # carrier input and output conversion technologies
    carrier_conversion_in, carrier_conversion_out = 0, 0
    for tech in model.set_conversion_technologies:
        if carrier in model.set_input_carriers[tech]:
            carrier_conversion_in     += model.input_flow[tech,carrier,node,time]
        if carrier in model.set_output_carriers[tech]:
            carrier_conversion_out    += model.output_flow[tech,carrier,node,time]
    # carrier flow transport technologies
    carrier_flow_in, carrier_flow_out   = 0, 0
    set_edges_in                      = EnergySystem.calculate_connected_edges(node,"in")
    set_edges_out                     = EnergySystem.calculate_connected_edges(node,"out")
    for tech in model.set_transport_technologies:
        if carrier in model.set_reference_carriers[tech]:
            carrier_flow_in   += sum(model.carrier_flow[tech, edge, time]
                            - model.carrier_loss[tech, edge, time] for edge in set_edges_in)
            carrier_flow_out  += sum(model.carrier_flow[tech, edge, time] for edge in set_edges_out)
    # carrier flow storage technologies
    carrier_flow_discharge, carrier_flow_charge = 0, 0
    for tech in model.set_storage_technologies:
        if carrier in model.set_reference_carriers[tech]:
            carrier_flow_discharge    += model.carrier_flow_discharge[tech,node,time]
            carrier_flow_charge       += model.carrier_flow_charge[tech,node,time]
    # carrier import, demand and export
    carrier_import, carrier_export, carrier_demand = 0, 0, 0
    carrier_import           = model.import_carrier_flow[carrier, node, time]
    carrier_export           = model.export_carrier_flow[carrier, node, time]
    carrier_demand           = params.demand_carrier[carrier, node, time]
    endogenous_carrier_demand = 0

    # check if carrier is conditioning carrier:
    if carrier in model.set_conditioning_carriers:
        # check if carrier is parentCarrier of a conditioning_carrier
        if carrier in model.set_conditioning_carrier_parents:
            endogenous_carrier_demand = - model.endogenous_carrier_demand[carrier, node, time]
        else:
            endogenous_carrier_demand = model.endogenous_carrier_demand[carrier, node, time]

    return (
        # conversion technologies
        carrier_conversion_out - carrier_conversion_in
        # transport technologies
        + carrier_flow_in - carrier_flow_out
        # storage technologies
        + carrier_flow_discharge - carrier_flow_charge
        # import and export 
        + carrier_import - carrier_export
        # demand
        - endogenous_carrier_demand - carrier_demand
        == 0
        )