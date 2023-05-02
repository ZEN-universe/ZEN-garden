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

import linopy as lp
import numpy as np

from .carrier import Carrier
from ..component import ZenIndex
from ..element import Element


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
        variables.add_variable(model, name="endogenous_carrier_demand", index_sets=cls.create_custom_set(
            ["set_conditioning_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup),
                               bounds=(0, np.inf),
                               doc="node- and time-dependent model endogenous carrier demand")

    @classmethod
    def construct_constraints(cls, optimization_setup):
        """ constructs the pe.Constraints of the class <Carrier>
        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        constraints = optimization_setup.constraints
        rules = ConditioningCarrierRules(optimization_setup)
        # limit import flow by availability
        constraints.add_constraint_block(model, name="constraint_carrier_demand_coupling",
                                         constraint=rules.get_constraint_carrier_demand_coupling(),
                                         doc='coupling model endogenous and exogenous carrier demand', )
        # overwrite energy balance when conditioning carriers are included
        constraints.remove_constraint(model, "constraint_nodal_energy_balance")
        constraints.add_constraint_block(model, name="constraint_nodal_energy_balance_conditioning",
                                         constraint=rules.get_constraint_nodal_energy_balance_conditioning(),
                                         doc='node- and time-dependent energy balance for each carrier', )


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

    def get_constraint_carrier_demand_coupling(self):
        """ sum conditioning Carriers"""

        model = self.optimization_setup.model
        sets = self.optimization_setup.sets

        # get all the constraints
        constraints = []
        index_values, index_names = Element.create_custom_set(["set_conditioning_carrier_parents", "set_nodes", "set_time_steps_operation"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)
        for parent_carrier in index.get_unique([0]):
            constraints.append(model.variables["endogenous_carrier_demand"].loc[parent_carrier]
                               - model.variables["endogenous_carrier_demand"].loc[list(sets["set_conditioning_carrier_children"][parent_carrier])].sum("set_conditioning_carriers")
                               == 0)
        return self.optimization_setup.constraints.reorder_list(constraints, index.get_unique([0]), index_names[:1], model)

    def get_constraint_nodal_energy_balance_conditioning(self):
        """
        nodal energy balance for each time step.
        """

        # get parameter object
        params = self.optimization_setup.parameters
        sets = self.optimization_setup.sets
        model = self.optimization_setup.model

        # get the index
        index_values, index_names = Carrier.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        # carrier flow transport technologies
        if model.variables["flow_transport"].size > 0:
            flow_transport_in = []
            flow_transport_out = []
            # precalculate the edges
            edges_in = {node: self.energy_system.calculate_connected_edges(node, "in") for node in index.get_unique([1])}
            edges_out = {node: self.energy_system.calculate_connected_edges(node, "out") for node in index.get_unique([1])}
            # loop over all nodes and carriers
            for carrier, node in index.get_unique([0, 1]):
                techs = []
                for tech in sets["set_transport_technologies"]:
                    if carrier in sets["set_reference_carriers"][tech]:
                        techs.append(tech)
                flow_transport_in.append((model.variables["flow_transport"].loc[techs, edges_in[node]] - model.variables["flow_transport_loss"].loc[techs, edges_in[node]]).sum(["set_transport_technologies", "set_edges"]))
                flow_transport_out.append(model.variables["flow_transport"].loc[techs, edges_out[node]].sum(["set_transport_technologies", "set_edges"]))
            # merge and regroup
            flow_transport_in = lp.merge(*flow_transport_in, dim="group")
            flow_transport_in = self.optimization_setup.constraints.reorder_group(flow_transport_in, None, None, index.get_unique([0, 1]), index_names[:-1], model)
            flow_transport_out = lp.merge(*flow_transport_out, dim="group")
            flow_transport_out = self.optimization_setup.constraints.reorder_group(flow_transport_out, None, None, index.get_unique([0, 1]), index_names[:-1], model)
        else:
            # if there is no carrier flow we just create empty arrays
            flow_transport_in = model.variables["flow_import"].where(False).to_linexpr()
            flow_transport_out = model.variables["flow_import"].where(False).to_linexpr()

        # carrier flow transport technologies
        carrier_conversion_in = []
        carrier_conversion_out = []
        nodes = list(sets["set_nodes"])
        for carrier in index.get_unique([0]):
            techs_in = [tech for tech in sets["set_conversion_technologies"] if carrier in sets["set_input_carriers"][tech]]
            # we need to catch emtpy lookups
            if len(techs_in) == 0:
                carrier_in = []
            else:
                carrier_in = [carrier]
            techs_out = [tech for tech in sets["set_conversion_technologies"] if carrier in sets["set_output_carriers"][tech]]
            # we need to catch emtpy lookups
            if len(techs_out) == 0:
                carrier_out = []
            else:
                carrier_out = [carrier]
            carrier_conversion_in.append(model.variables["flow_conversion_input"].loc[techs_in, carrier_in, nodes].sum(model.variables["flow_conversion_input"].dims[:2]))
            carrier_conversion_out.append(model.variables["flow_conversion_output"].loc[techs_out, carrier_out, nodes].sum(model.variables["flow_conversion_output"].dims[:2]))
        # merge and regroup
        carrier_conversion_in = lp.merge(*carrier_conversion_in, dim="group")
        carrier_conversion_in = self.optimization_setup.constraints.reorder_group(carrier_conversion_in, None, None, index.get_unique([0]), index_names[:1], model)
        carrier_conversion_out = lp.merge(*carrier_conversion_out, dim="group")
        carrier_conversion_out = self.optimization_setup.constraints.reorder_group(carrier_conversion_out, None, None, index.get_unique([0]), index_names[:1], model)

        # carrier flow storage technologies
        if model.variables["flow_storage_discharge"].size > 0:
            flow_storage_discharge = []
            flow_storage_charge = []
            for carrier in index.get_unique([0]):
                storage_techs = [tech for tech in sets["set_storage_technologies"] if carrier in sets["set_reference_carriers"][tech]]
                flow_storage_discharge.append(model.variables["flow_storage_discharge"].loc[storage_techs].sum("set_storage_technologies"))
                flow_storage_charge.append(model.variables["flow_storage_charge"].loc[storage_techs].sum("set_storage_technologies"))
            # merge and regroup
            flow_storage_discharge = lp.merge(*flow_storage_discharge, dim="group")
            flow_storage_discharge = self.optimization_setup.constraints.reorder_group(flow_storage_discharge, None, None, index.get_unique([0]), index_names[:1], model)
            flow_storage_charge = lp.merge(*flow_storage_charge, dim="group")
            flow_storage_charge = self.optimization_setup.constraints.reorder_group(flow_storage_charge, None, None, index.get_unique([0]), index_names[:1], model)
        else:
            # if there is no carrier flow we just create empty arrays
            flow_storage_discharge = model.variables["flow_import"].where(False).to_linexpr()
            flow_storage_charge = model.variables["flow_import"].where(False).to_linexpr()

        # carrier import, demand and export
        carrier_import = model.variables["flow_import"].to_linexpr()
        carrier_export = model.variables["flow_export"].to_linexpr()
        carrier_demand = params.demand

        # check if carrier is conditioning carrier:
        endogenous_carrier_demand = []
        for carrier in index.get_unique([0]):
            # check if carrier is conditioning carrier
            if carrier in sets["set_conditioning_carriers"]:
                # check if carrier is parent_carrier of a conditioning_carrier
                if carrier in sets["set_conditioning_carrier_parents"]:
                    endogenous_carrier_demand.append(-1.0*model.variables["endogenous_carrier_demand"].loc[carrier])
                else:
                    endogenous_carrier_demand.append(1.0 * model.variables["endogenous_carrier_demand"].loc[carrier])
            else:
                # something empty with the right shape
                endogenous_carrier_demand.append(lp.LinearExpression(data=None, model=model))
        # merge and regroup
        endogenous_carrier_demand = lp.merge(*endogenous_carrier_demand, dim="group")
        endogenous_carrier_demand = self.optimization_setup.constraints.reorder_group(endogenous_carrier_demand, None, None, index.get_unique([0]), index_names[:1], model)
        # shed demand
        carrier_shed_demand = model.variables["shed_demand"].to_linexpr()

        # Add everything
        lhs = lp.merge(carrier_conversion_out,
                       -carrier_conversion_in,
                       flow_transport_in,
                       -flow_transport_out,
                       flow_storage_charge,
                       -flow_storage_discharge,
                       carrier_import,
                       -carrier_export,
                       -endogenous_carrier_demand,
                       carrier_shed_demand)

        return lhs == carrier_demand
