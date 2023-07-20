"""
:Title:          ZEN-GARDEN
:Created:        October-2021
:Authors:        Alissa Ganter (aganter@ethz.ch),
                Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Class defining compressable energy carriers.
The class takes as inputs the abstract optimization model. The class adds parameters, variables and
constraints of a generic carrier and returns the abstract optimization model.
"""
import logging

import linopy as lp
import numpy as np
import xarray as xr

from .carrier import Carrier
from ..component import ZenIndex
from ..element import Element, GenericRule


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
                                         constraint=rules.constraint_carrier_demand_coupling_block(),
                                         doc='coupling model endogenous and exogenous carrier demand', )
        # overwrite energy balance when conditioning carriers are included
        constraints.remove_constraint(model, "constraint_nodal_energy_balance")
        constraints.add_constraint_block(model, name="constraint_nodal_energy_balance_conditioning",
                                         constraint=rules.constraint_nodal_energy_balance_conditioning_block(),
                                         doc='node- and time-dependent energy balance for each carrier', )


class ConditioningCarrierRules(GenericRule):
    """
    Rules for the ConditioningCarrier class
    """

    def __init__(self, optimization_setup):
        """
        Inits the rules for a given EnergySystem

        :param optimization_setup: The OptimizationSetup the element is part of
        """

        super().__init__(optimization_setup)

    # Rule-based constraints
    # -----------------------

    # Block-based constraints
    # -----------------------

    def constraint_carrier_demand_coupling_block(self):
        """ sum conditioning carriers

        .. math::
            d_{c^p,n,t}^\mathrm{endogenous} = \sum_{c^c\in\mathcal{C}_i^c} d_{c^c,n,t}^\mathrm{endogenous}

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_conditioning_carrier_parents", "set_nodes", "set_time_steps_operation"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we only loop over the parent carrier and vectorize over nodes and time steps
        constraints = []
        for parent_carrier in index.get_unique(["set_conditioning_carrier_parents"]):
            ### auxiliary calculations
            term_parent_carrier_sum = self.variables["endogenous_carrier_demand"].loc[list(self.sets["set_conditioning_carrier_children"][parent_carrier])].sum("set_conditioning_carriers")

            ### formulate constraint
            lhs = self.variables["endogenous_carrier_demand"].loc[parent_carrier] - term_parent_carrier_sum
            rhs = 0
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_conditioning_carrier_parents"]),
                                                  index_names=["set_conditioning_carrier_parents"])

    def constraint_nodal_energy_balance_conditioning_block(self):
        """
        nodal energy balance for each time step.

        .. math::
            0 = -(d_{c,n,t}^\mathrm{endogenous} + d_{c,n,t}-D_{c,n,t})
            + \\sum_{i\\in\mathcal{I}}(\\overline{G}_{c,i,n,t}-\\underline{G}_{c,i,n,t})
            + \\sum_{j\\in\mathcal{J}}\\sum_{e\\in\\underline{\mathcal{E}}}F_{j,e,t}-F^\mathrm{l}_{j,e,t})-\\sum_{e'\\in\\overline{\mathcal{E}}}F_{j,e',t})
            + \\sum_{k\\in\mathcal{K}}(\\overline{H}_{k,n,t}-\\underline{H}_{k,n,t})
            + U_{c,n,t} - V_{c,n,t}

        :return: #TODO describe parameter/return

        """

        ### index sets
        index_values, index_names = Carrier.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"],
                                                              self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # This constraints does not have a central index loop, but multiple in the auxiliary calculations

        ### auxiliary calculations
        # carrier flow transport technologies
        if self.variables["flow_transport"].size > 0:
            # recalculate all the edges
            edges_in = {node: self.energy_system.calculate_connected_edges(node, "in") for node in
                        self.sets["set_nodes"]}
            edges_out = {node: self.energy_system.calculate_connected_edges(node, "out") for node in
                         self.sets["set_nodes"]}
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
            term_carrier_conversion_in.append(
                self.variables["flow_conversion_input"].loc[techs_in, carrier_in, nodes].sum(
                    self.variables["flow_conversion_input"].dims[:2]))
            term_carrier_conversion_out.append(
                self.variables["flow_conversion_output"].loc[techs_out, carrier_out, nodes].sum(
                    self.variables["flow_conversion_output"].dims[:2]))
        # merge and regroup
        term_carrier_conversion_in = lp.merge(*term_carrier_conversion_in, dim="group")
        term_carrier_conversion_in = self.optimization_setup.constraints.reorder_group(term_carrier_conversion_in, None,
                                                                                       None,
                                                                                       index.get_unique([0]),
                                                                                       index_names[:1], self.model)
        term_carrier_conversion_out = lp.merge(*term_carrier_conversion_out, dim="group")
        term_carrier_conversion_out = self.optimization_setup.constraints.reorder_group(term_carrier_conversion_out,
                                                                                        None, None,
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
            term_flow_storage_discharge = self.optimization_setup.constraints.reorder_group(term_flow_storage_discharge,
                                                                                            None,
                                                                                            None, index.get_unique([0]),
                                                                                            index_names[:1], self.model)
            term_flow_storage_charge = lp.merge(*term_flow_storage_charge, dim="group")
            term_flow_storage_charge = self.optimization_setup.constraints.reorder_group(term_flow_storage_charge, None,
                                                                                         None,
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

        # check if carrier is conditioning carrier:
        term_endogenous_carrier_demand = []
        for carrier in index.get_unique([0]):
            # check if carrier is conditioning carrier
            if carrier in self.sets["set_conditioning_carriers"]:
                # check if carrier is parent_carrier of a conditioning_carrier
                if carrier in self.sets["set_conditioning_carrier_parents"]:
                    term_endogenous_carrier_demand.append(
                        -1.0 * self.variables["endogenous_carrier_demand"].loc[carrier])
                else:
                    term_endogenous_carrier_demand.append(
                        1.0 * self.variables["endogenous_carrier_demand"].loc[carrier])
            else:
                # something empty with the right shape
                term_endogenous_carrier_demand.append(lp.LinearExpression(data=None, model=self.model))
        # merge and regroup
        term_endogenous_carrier_demand = lp.merge(*term_endogenous_carrier_demand, dim="group")
        term_endogenous_carrier_demand = self.optimization_setup.constraints.reorder_group(term_endogenous_carrier_demand,
                                                                                      None, None,
                                                                                      index.get_unique([0]),
                                                                                      index_names[:1], self.model)
        # shed demand
        term_carrier_shed_demand = self.variables["shed_demand"].to_linexpr()

        ### form constraints
        lhs = lp.merge(term_carrier_conversion_out,
                       -term_carrier_conversion_in,
                       term_flow_transport_in,
                       -term_flow_transport_out,
                       -term_flow_storage_charge,
                       term_flow_storage_discharge,
                       term_carrier_import,
                       -term_carrier_export,
                       -term_endogenous_carrier_demand,
                       term_carrier_shed_demand,
                       compat="broadcast_equals")
        rhs = term_carrier_demand
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)
