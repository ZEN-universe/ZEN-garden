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
import time

import linopy as lp
import numpy as np
import xarray as xr

from ..component import ZenIndex
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
        t0 = time.perf_counter()
        constraints.add_constraint_block(model, name="constraint_carrier_demand_coupling",
                                         constraint=rules.get_constraint_carrier_demand_coupling(*cls.create_custom_set(
                                             ["set_conditioning_carrier_parents", "set_nodes",
                                              "set_time_steps_operation"], optimization_setup)),
                                         doc='coupling model endogenous and exogenous carrier demand', )
        t1 = time.perf_counter()
        logging.debug(f"Conditioningn Carrier: constraint_carrier_demand_coupling took {t1 - t0:.4f} seconds")
        # overwrite energy balance when conditioning carriers are included
        constraints.remove_constraint(model, "constraint_nodal_energy_balance")
        constraints.add_constraint_block(model, name="constraint_nodal_energy_balance_conditioning",
                                         constraint=rules.get_constraint_nodal_energy_balance_conditioning(),
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
                               - model.variables["endogenous_carrier_demand"].loc[
                                   np.array(sets["set_conditioning_carrier_children"][parent_carrier])].sum(
                "set_conditioning_carriers")
                               == 0)
        return constraints

    def get_constraint_nodal_energy_balance_conditioning(self):
        """
        nodal energy balance for each time step.
        """

        # TODO: adapt like carrier routine
        params = self.optimization_setup.parameters
        sets = self.optimization_setup.sets
        model = self.optimization_setup.model

        # get the index
        index_values, index_names = ConditioningCarrier.create_custom_set(
            ["set_carriers", "set_nodes", "set_time_steps_operation"], self.optimization_setup)

        # define the groups that are summed up (-1 is a dummy value)
        carrier_conversion_in_group = xr.DataArray(-1, coords=model.variables["input_flow"].coords)
        carrier_conversion_out_group = xr.DataArray(-1, coords=model.variables["output_flow"].coords)
        carrier_flow_in_group = xr.DataArray(-1, coords=model.variables["carrier_flow"].coords)
        carrier_flow_out_group = xr.DataArray(-1, coords=model.variables["carrier_flow"].coords)
        carrier_flow_discharge_group = xr.DataArray(-1, coords=model.variables["carrier_flow_discharge"].coords)
        carrier_flow_charge_group = xr.DataArray(-1, coords=model.variables["carrier_flow_charge"].coords)
        carrier_import_group = xr.DataArray(-1, coords=model.variables["import_carrier_flow"].coords)
        carrier_export_group = xr.DataArray(-1, coords=model.variables["export_carrier_flow"].coords)
        endogenous_carrier_demand_group_pos = xr.DataArray(-1,
                                                           coords=model.variables["endogenous_carrier_demand"].coords)
        endogenous_carrier_demand_group_neg = xr.DataArray(-1,
                                                           coords=model.variables["endogenous_carrier_demand"].coords)
        carrier_demand_group = xr.DataArray(-1, coords=params.demand_carrier.coords)
        for num, (carrier, node, time) in enumerate(index_values):

            # carrier input and output conversion technologies
            for tech in sets["set_conversion_technologies"]:
                if carrier in sets["set_input_carriers"][tech]:
                    carrier_conversion_in_group.loc[tech, carrier, node, time] = num
                if carrier in sets["set_output_carriers"][tech]:
                    carrier_conversion_out_group.loc[tech, carrier, node, time] = num

            # carrier flow transport technologies
            set_edges_in = self.energy_system.calculate_connected_edges(node, "in")
            set_edges_out = self.energy_system.calculate_connected_edges(node, "out")
            for tech in sets["set_transport_technologies"]:
                if carrier in sets["set_reference_carriers"][tech]:
                    carrier_flow_in_group.loc[tech, set_edges_in, time] = num
                    carrier_flow_out_group.loc[tech, set_edges_out, time] = num

            # carrier flow storage technologies
            for tech in sets["set_storage_technologies"]:
                if carrier in sets["set_reference_carriers"][tech]:
                    carrier_flow_discharge_group.loc[tech, node, time] = num
                    carrier_flow_charge_group.loc[tech, node, time] = num

            # carrier import, demand and export
            carrier_import_group.loc[carrier, node, time] = num
            carrier_export_group.loc[carrier, node, time] = num
            carrier_demand_group.loc[carrier, node, time] = num
            # check if carrier is conditioning carrier:
            if carrier in sets["set_conditioning_carriers"]:
                # check if carrier is parent_carrier of a conditioning_carrier
                if carrier in sets["set_conditioning_carrier_parents"]:
                    endogenous_carrier_demand_group_pos.loc[carrier, node, time] = num
                else:
                    endogenous_carrier_demand_group_neg.loc[carrier, node, time] = num

        # sum over all groups, with merge and broadcast to handle missing dims correctly
        lhs = lp.expressions.merge(
            # carrier_conversion_out - carrier_conversion_in
            (model.variables["output_flow"]).groupby_sum(carrier_conversion_out_group),
            - (model.variables["input_flow"]).groupby_sum(carrier_conversion_in_group),
            # carrier_flow_out - carrier_flow_in
            (model.variables["carrier_flow"] - model.variables["carrier_loss"]).groupby_sum(carrier_flow_in_group),
            - (model.variables["carrier_flow"]).groupby_sum(carrier_flow_out_group),
            # carrier_flow_discharge - carrier_flow_charge
            (model.variables["carrier_flow_discharge"]).groupby_sum(carrier_flow_discharge_group),
            - (model.variables["carrier_flow_charge"]).groupby_sum(carrier_flow_charge_group),
            # carrier_import - carrier_export
            (model.variables["import_carrier_flow"]).groupby_sum(carrier_import_group),
            - (model.variables["export_carrier_flow"]).groupby_sum(carrier_export_group),
            # endogenous_carrier_demand
            (model.variables["endogenous_carrier_demand"]).groupby_sum(endogenous_carrier_demand_group_pos),
            - (model.variables["endogenous_carrier_demand"]).groupby_sum(endogenous_carrier_demand_group_neg),
            compat="broadcast_equals")
        rhs = params.demand_carrier.groupby(carrier_demand_group).map(lambda x: x.sum())
        sign = xr.DataArray("==", coords=rhs.coords)
        # to a nice constraint proper dims
        return self.optimization_setup.constraints.reorder_group(lhs, sign, rhs, index_values, index_names, model,
                                                                 drop=-1)
