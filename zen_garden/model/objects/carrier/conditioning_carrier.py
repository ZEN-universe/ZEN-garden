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

import numpy as np
import xarray as xr
import time

from zen_garden.utils import ZenIndex
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
        constraints = optimization_setup.constraints
        rules = ConditioningCarrierRules(optimization_setup)
        # limit import flow by availability
        t0 = time.perf_counter()
        constraints.add_constraint_block(model, name="constraint_carrier_demand_coupling",
                                         constraint=rules.get_constraint_carrier_demand_coupling(*cls.create_custom_set(["set_conditioning_carrier_parents", "set_nodes", "set_time_steps_operation"], optimization_setup)),
                                         doc='coupling model endogenous and exogenous carrier demand', )
        t1 = time.perf_counter()
        logging.debug(f"Conditioningn Carrier: constraint_carrier_demand_coupling took {t1 - t0:.4f} seconds")
        # overwrite energy balance when conditioning carriers are included
        constraints.remove_constraint(model, "constraint_nodal_energy_balance")
        constraints.add_constraint_block(model, name="constraint_nodal_energy_balance_conditioning",
                                         constraint=rules.get_constraint_nodal_energy_balance_conditioning(*cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup)),
                                         doc='node- and time-dependent energy balance for each carrier', )
        t2 = time.perf_counter()
        logging.debug(f"Conditioningn Carrier: constraint_nodal_energy_balance_conditioning took {t2 - t1:.4f} seconds")


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

    def get_constraint_carrier_demand_coupling(self, index_values, index_names):
        """ sum conditioning Carriers"""

        model = self.optimization_setup.model
        sets = self.optimization_setup.sets

        # get all the constraints
        constraints = []
        index = ZenIndex(index_values, index_names)
        for parent_carrier in index.get_unique([0]):
            constraints.append(model.variables["endogenous_carrier_demand"].loc[parent_carrier]
                               - model.variables["endogenous_carrier_demand"].loc[np.array(sets["set_conditioning_carrier_children"][parent_carrier])].sum("set_conditioning_carriers")
                               == 0)
        return constraints

    def get_constraint_nodal_energy_balance_conditioning(self, index_values, index_names):
        """
        nodal energy balance for each time step.
        """
        params = self.optimization_setup.parameters
        sets = self.optimization_setup.sets
        model = self.optimization_setup.model

        # get all the constraints
        constraints = []
        index = ZenIndex(index_values, index_names)
        for carrier, node in index.get_unique([0, 1]):

            # carrier input and output conversion technologies
            in_techs = [tech for tech in sets["set_conversion_technologies"] if carrier in sets["set_input_carriers"][tech]]
            carrier_conversion_in = 0
            if len(in_techs) > 0:
                carrier_conversion_in = (model.variables["input_flow"].loc[in_techs, carrier, node]).sum("set_conversion_technologies")
            out_techs = [tech for tech in sets["set_conversion_technologies"] if carrier in sets["set_output_carriers"][tech]]
            carrier_conversion_out = 0
            if len(out_techs) > 0:
                carrier_conversion_out = (model.variables["output_flow"].loc[out_techs, carrier, node]).sum("set_conversion_technologies")
            # carrier flow transport technologies
            carrier_flow_in, carrier_flow_out = 0, 0
            set_edges_in = xr.DataArray(self.energy_system.calculate_connected_edges(node, "in"), dims=["set_edges"])
            set_edges_out = xr.DataArray(self.energy_system.calculate_connected_edges(node, "out"), dims=["set_edges"])
            techs = xr.DataArray([tech for tech in sets["set_transport_technologies"] if carrier in sets["set_reference_carriers"][tech]], dims=["set_transport_technologies"])
            if len(techs) > 0:
                if len(set_edges_in) > 0:
                    carrier_flow_in = model.variables["carrier_flow"].loc[techs, set_edges_in].sum(["set_transport_technologies", "set_edges"])
                    carrier_flow_in -= model.variables["carrier_loss"].loc[techs, set_edges_in].sum(["set_transport_technologies", "set_edges"])
                if len(set_edges_out) > 0:
                    carrier_flow_out = model.variables["carrier_flow"].loc[techs, set_edges_out].sum(["set_transport_technologies", "set_edges"])
            # carrier flow storage technologies
            carrier_flow_discharge, carrier_flow_charge = 0, 0
            techs = [tech for tech in sets["set_storage_technologies"] if carrier in sets["set_reference_carriers"][tech]]
            if len(techs) > 0:
                carrier_flow_discharge = (model.variables["carrier_flow_discharge"].loc[techs, node]).sum("set_storage_technologies")
                carrier_flow_charge = (model.variables["carrier_flow_charge"].loc[techs, node]).sum("set_storage_technologies")
            # carrier import, demand and export
            carrier_import = model.variables["import_carrier_flow"].loc[carrier, node]
            carrier_export = model.variables["export_carrier_flow"].loc[carrier, node]
            carrier_demand = params.demand_carrier.loc[carrier, node]
            endogenous_carrier_demand = 0
            # check if carrier is conditioning carrier:
            if carrier in sets["set_conditioning_carriers"]:
                # check if carrier is parent_carrier of a conditioning_carrier
                if carrier in sets["set_conditioning_carrier_parents"]:
                    endogenous_carrier_demand = -model.variables["endogenous_carrier_demand"].loc[carrier, node]
                else:
                    endogenous_carrier_demand = model.variables["endogenous_carrier_demand"].loc[carrier, node]

            # aggregate, note some terms might be 0 so we need to make sure we don't add variabl
            terms = [carrier_conversion_out, -carrier_conversion_in, carrier_flow_in, -carrier_flow_out, carrier_flow_discharge,
                     -carrier_flow_charge, carrier_import, -carrier_export, -endogenous_carrier_demand, -carrier_demand]
            lhs = sum([term for term in terms if not isinstance(term, (xr.DataArray, float, int))])
            rhs = -sum([term for term in terms if isinstance(term, (xr.DataArray, float, int))])

            constraints.append(lhs == rhs)
        return constraints
