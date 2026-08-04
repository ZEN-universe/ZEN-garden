from typing import TYPE_CHECKING

import linopy as lp
import numpy as np
import xarray as xr
from linopy.expressions import LinearExpression

if TYPE_CHECKING:
    from zen_garden.elements.energy_system import EnergySystem
    from zen_garden.model.components.multi_index_helper import MultiIndexHelper
    from zen_garden.model.zen_model import ZenModel


class NodalEnergyBalanceConstraint:
    def __init__(
        self,
        zen_model: "ZenModel",
        energy_system: "EnergySystem",
    ):
        self.zen_model = zen_model
        self.energy_system = energy_system

    def build(self, index: "MultiIndexHelper", first_index_name: list[str]):
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
        if self.zen_model.variables["flow_transport"].size > 0:
            # recalculate all the edges
            edges_in = {
                node: self.energy_system.calculate_connected_edges(node, "in")
                for node in self.zen_model.sets["set_nodes"]
            }
            edges_out = {
                node: self.energy_system.calculate_connected_edges(node, "out")
                for node in self.zen_model.sets["set_nodes"]
            }
            max_edges = max(
                [len(edges_in[node]) for node in self.zen_model.sets["set_nodes"]]
                + [len(edges_out[node]) for node in self.zen_model.sets["set_nodes"]]
            )

            # create the variables
            flow_transport_in_vars = xr.DataArray(
                -1,
                coords=[
                    self.zen_model.parameters.demand.coords["set_carriers"],
                    self.zen_model.parameters.demand.coords["set_nodes"],
                    self.zen_model.parameters.demand.coords["set_time_steps_operation"],
                    xr.DataArray(
                        np.arange(
                            len(self.zen_model.sets["set_transport_technologies"])
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
                    for tech in self.zen_model.sets["set_transport_technologies"]
                    if carrier in self.zen_model.sets["set_reference_carriers"][tech]
                ]
                edges_in = self.energy_system.calculate_connected_edges(node, "in")
                edges_out = self.energy_system.calculate_connected_edges(node, "out")

                # get the variables for the in flow
                in_vars_plus = (
                    self.zen_model.variables["flow_transport"]
                    .labels.loc[techs, edges_in, :]
                    .data
                )
                in_vars_plus = in_vars_plus.reshape((-1, in_vars_plus.shape[-1])).T
                in_coefs_plus = np.ones_like(in_vars_plus)
                in_vars_minus = (
                    self.zen_model.variables["flow_transport_loss"]
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
                    self.zen_model.variables["flow_transport"]
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
                self.zen_model.lp_model,
            )
            term_flow_transport_out = lp.LinearExpression(
                xr.Dataset(
                    {
                        "coeffs": flow_transport_out_coeffs,
                        "vars": flow_transport_out_vars,
                    }
                ),
                self.zen_model.lp_model,
            )
        else:
            # if there is no carrier flow we just create empty arrays
            term_flow_transport_in = (
                self.zen_model.variables["flow_import"]
                .where(xr.DataArray(False))
                .to_linexpr()
            )
            term_flow_transport_out = (
                self.zen_model.variables["flow_import"]
                .where(xr.DataArray(False))
                .to_linexpr()
            )

        # carrier input and output conversion technologies
        term_carrier_conversion_in = []
        term_carrier_conversion_out = []
        nodes = list(self.zen_model.sets["set_nodes"])
        for carrier in index.get_unique([0]):
            techs_in = [
                tech
                for tech in self.zen_model.sets["set_conversion_technologies"]
                if carrier in self.zen_model.sets["set_input_carriers"][tech]
            ]
            # we need to catch emtpy lookups
            carrier_in = [carrier] if len(techs_in) > 0 else []
            techs_out = [
                tech
                for tech in self.zen_model.sets["set_conversion_technologies"]
                if carrier in self.zen_model.sets["set_output_carriers"][tech]
            ]
            # we need to catch emtpy lookups
            carrier_out = [carrier] if len(techs_out) > 0 else []
            term_carrier_conversion_in.append(
                self.zen_model.variables["flow_conversion_input"]
                .loc[techs_in, carrier_in, nodes]
                .sum(self.zen_model.variables["flow_conversion_input"].dims[:2])
            )
            term_carrier_conversion_out.append(
                self.zen_model.variables["flow_conversion_output"]
                .loc[techs_out, carrier_out, nodes]
                .sum(self.zen_model.variables["flow_conversion_output"].dims[:2])
            )
        # merge and regroup
        term_carrier_conversion_in = lp.merge(
            term_carrier_conversion_in, dim="group", join="outer", cls=LinearExpression
        )
        term_carrier_conversion_in = self.zen_model.constraints.reorder_group(
            term_carrier_conversion_in,
            None,
            None,
            index.get_unique([0]),
            first_index_name,
            self.zen_model.lp_model,
        )
        term_carrier_conversion_out = lp.merge(
            term_carrier_conversion_out, dim="group", join="outer", cls=LinearExpression
        )
        term_carrier_conversion_out = self.zen_model.constraints.reorder_group(
            term_carrier_conversion_out,
            None,
            None,
            index.get_unique([0]),
            first_index_name,
            self.zen_model.lp_model,
        )

        # carrier flow storage technologies
        if self.zen_model.variables["flow_storage_discharge"].size > 0:
            term_flow_storage_discharge = []
            term_flow_storage_charge = []
            for carrier in index.get_unique([0]):
                storage_techs = [
                    tech
                    for tech in self.zen_model.sets["set_storage_technologies"]
                    if carrier in self.zen_model.sets["set_reference_carriers"][tech]
                ]
                term_flow_storage_discharge.append(
                    self.zen_model.variables["flow_storage_discharge"]
                    .loc[storage_techs]
                    .sum("set_storage_technologies")
                )
                term_flow_storage_charge.append(
                    self.zen_model.variables["flow_storage_charge"]
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
            term_flow_storage_discharge = self.zen_model.constraints.reorder_group(
                term_flow_storage_discharge,
                None,
                None,
                index.get_unique([0]),
                first_index_name,
                self.zen_model.lp_model,
            )
            term_flow_storage_charge = lp.merge(
                term_flow_storage_charge,
                dim="group",
                join="outer",
                cls=LinearExpression,
            )
            term_flow_storage_charge = self.zen_model.constraints.reorder_group(
                term_flow_storage_charge,
                None,
                None,
                index.get_unique([0]),
                first_index_name,
                self.zen_model.lp_model,
            )
        else:
            # if there is no carrier flow we just create empty arrays
            term_flow_storage_discharge = (
                self.zen_model.variables["flow_import"]
                .where(xr.DataArray(False))
                .to_linexpr()
            )
            term_flow_storage_charge = (
                self.zen_model.variables["flow_import"]
                .where(xr.DataArray(False))
                .to_linexpr()
            )

        # carrier import, demand and export
        term_carrier_import = self.zen_model.variables["flow_import"].to_linexpr()
        term_carrier_export = self.zen_model.variables["flow_export"].to_linexpr()
        term_carrier_demand = self.zen_model.parameters.demand
        # shed demand
        term_carrier_shed_demand = self.zen_model.variables["shed_demand"].to_linexpr()

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
        self.zen_model.add_constraint("constraint_nodal_energy_balance", constraints)
