import linopy as lp
import numpy as np
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.model.registries.multi_index_helper import MultiIndexHelper


class NodalEnergyBalanceConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Nodal energy balance for each time step.

        Formulation:

        .. math::
            \\begin{aligned}
            0={}&-(d_{c,n,t}-F^{\\mathrm{shed}}_{c,n,t})\\\\
            &+\\sum_{h\\in\\mathcal{H}^{\\mathrm{conv,out}}_c}
              F^{\\mathrm{conv,out}}_{h,c,n,t}
            -\\sum_{h\\in\\mathcal{H}^{\\mathrm{conv,in}}_c}
              F^{\\mathrm{conv,in}}_{h,c,n,t}\\\\
            &+\\sum_{h\\in\\mathcal{H}^{\\mathrm{trans}}_c}
              \\left[
                \\sum_{e\\in\\mathcal{E}^{\\mathrm{in}}_n}
                  (F^{\\mathrm{trans}}_{h,e,t}-F^{\\mathrm{loss}}_{h,e,t})
                -\\sum_{e'\\in\\mathcal{E}^{\\mathrm{out}}_n}
                  F^{\\mathrm{trans}}_{h,e',t}
              \\right]\\\\
            &+\\sum_{h\\in\\mathcal{H}^{\\mathrm{stor}}_c}
              (F^{\\mathrm{dis}}_{h,n,t}-F^{\\mathrm{ch}}_{h,n,t})
            +F^{\\mathrm{imp}}_{c,n,t}-F^{\\mathrm{exp}}_{c,n,t}.
            \\end{aligned}

        The carrier-specific sets restrict each sum to technologies that actually
        use carrier :math:`c`: :math:`\\mathcal{H}^{\\mathrm{conv,in}}_c` and
        :math:`\\mathcal{H}^{\\mathrm{conv,out}}_c` contain conversion technologies
        consuming and producing :math:`c`, respectively, while
        :math:`\\mathcal{H}^{\\mathrm{trans}}_c` and
        :math:`\\mathcal{H}^{\\mathrm{stor}}_c` contain transport and storage
        technologies whose reference carrier is :math:`c`.

        Notation:

        Sources of carrier :math:`c` at node :math:`n` in time step :math:`t`
        of year :math:`y`:

        :math:`F^{\\mathrm{conv,out}}_{h,c,n,t}`: output flow of
        carrier :math:`c` from conversion technology :math:`h`
        :math:`F^{\\mathrm{trans}}_{h,e,t}`: transported flow on ingoing edge
        :math:`e`, minus loss :math:`F^{\\mathrm{loss}}_{h,e,t}`, for transport
        technology :math:`h`
        :math:`F^{\\mathrm{dis}}_{h,n,t}`: output flow from storage technology
        :math:`h`
        :math:`F^{\\mathrm{imp}}_{c,n,t}`: imported carrier flow

        Sinks of carrier :math:`c` at node :math:`n` in time step :math:`t`
        of year :math:`y`:
        :math:`d_{c,n,t}-F^{\\mathrm{shed}}_{c,n,t}`: served demand
        :math:`F^{\\mathrm{conv,in}}_{h,c,n,t}`: input flow to
        conversion technology
        :math:`h`
        :math:`F^{\\mathrm{trans}}_{h,e',t}`: transported flow on outgoing
        edge :math:`e'`
        :math:`F^{\\mathrm{ch}}_{h,n,t}`: input flow to storage technology :math:`h`
        :math:`F^{\\mathrm{exp}}_{c,n,t}`: exported carrier flow
        """
        index_values, index_names = (
            model_constructor.optimization_model.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"]
            )
        )
        index = MultiIndexHelper(index_values, index_names)
        first_index_name = index_names[:1]

        ### masks
        # not necessary

        ### index loop
        # This constraints does not have a central index loop, but multiple in the
        # auxiliary calculations

        ### auxiliary calculations
        # carrier flow transport technologies
        if model_constructor.optimization_model.variables["flow_transport"].size > 0:
            # recalculate all the edges
            edges_in = {
                node: model_constructor.network_topology.calculate_connected_edges(
                    node, "in"
                )
                for node in model_constructor.optimization_model.sets["set_nodes"]
            }
            edges_out = {
                node: model_constructor.network_topology.calculate_connected_edges(
                    node, "out"
                )
                for node in model_constructor.optimization_model.sets["set_nodes"]
            }
            max_edges = max(
                [
                    len(edges_in[node])
                    for node in model_constructor.optimization_model.sets["set_nodes"]
                ]
                + [
                    len(edges_out[node])
                    for node in model_constructor.optimization_model.sets["set_nodes"]
                ]
            )

            # create the variables
            flow_transport_in_vars = xr.DataArray(
                -1,
                coords=[
                    model_constructor.optimization_model.parameters.demand.coords[
                        "set_carriers"
                    ],
                    model_constructor.optimization_model.parameters.demand.coords[
                        "set_nodes"
                    ],
                    model_constructor.optimization_model.parameters.demand.coords[
                        "set_time_steps_operation"
                    ],
                    xr.DataArray(
                        np.arange(
                            len(
                                model_constructor.optimization_model.sets[
                                    "set_transport_technologies"
                                ]
                            )
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
                    for tech in model_constructor.optimization_model.sets[
                        "set_transport_technologies"
                    ]
                    if carrier
                    in model_constructor.optimization_model.sets[
                        "set_reference_carriers"
                    ][tech]
                ]
                edges_in = model_constructor.network_topology.calculate_connected_edges(
                    node, "in"
                )
                edges_out = (
                    model_constructor.network_topology.calculate_connected_edges(
                        node, "out"
                    )
                )

                # get the variables for the in flow
                in_vars_plus = (
                    model_constructor.optimization_model.variables["flow_transport"]
                    .labels.loc[techs, edges_in, :]
                    .data
                )
                in_vars_plus = in_vars_plus.reshape((-1, in_vars_plus.shape[-1])).T
                in_coefs_plus = np.ones_like(in_vars_plus)
                in_vars_minus = (
                    model_constructor.optimization_model.variables[
                        "flow_transport_loss"
                    ]
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
                    model_constructor.optimization_model.variables["flow_transport"]
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
                model_constructor.optimization_model.lp_model,
            )
            term_flow_transport_out = lp.LinearExpression(
                xr.Dataset(
                    {
                        "coeffs": flow_transport_out_coeffs,
                        "vars": flow_transport_out_vars,
                    }
                ),
                model_constructor.optimization_model.lp_model,
            )
        else:
            # if there is no carrier flow we just create empty arrays
            term_flow_transport_in = (
                model_constructor.optimization_model.variables["flow_import"]
                .where(xr.DataArray(False))
                .to_linexpr()
            )
            term_flow_transport_out = (
                model_constructor.optimization_model.variables["flow_import"]
                .where(xr.DataArray(False))
                .to_linexpr()
            )

        # carrier input and output conversion technologies
        term_carrier_conversion_in = []
        term_carrier_conversion_out = []
        nodes = list(model_constructor.optimization_model.sets["set_nodes"])
        for carrier in index.get_unique([0]):
            techs_in = [
                tech
                for tech in model_constructor.optimization_model.sets[
                    "set_conversion_technologies"
                ]
                if carrier
                in model_constructor.optimization_model.sets["set_input_carriers"][tech]
            ]
            # we need to catch emtpy lookups
            carrier_in = [carrier] if len(techs_in) > 0 else []
            techs_out = [
                tech
                for tech in model_constructor.optimization_model.sets[
                    "set_conversion_technologies"
                ]
                if carrier
                in model_constructor.optimization_model.sets["set_output_carriers"][
                    tech
                ]
            ]
            # we need to catch emtpy lookups
            carrier_out = [carrier] if len(techs_out) > 0 else []
            term_carrier_conversion_in.append(
                model_constructor.optimization_model.variables["flow_conversion_input"]
                .loc[techs_in, carrier_in, nodes]
                .sum(
                    model_constructor.optimization_model.variables[
                        "flow_conversion_input"
                    ].dims[:2]
                )
            )
            term_carrier_conversion_out.append(
                model_constructor.optimization_model.variables["flow_conversion_output"]
                .loc[techs_out, carrier_out, nodes]
                .sum(
                    model_constructor.optimization_model.variables[
                        "flow_conversion_output"
                    ].dims[:2]
                )
            )
        # merge and regroup
        term_carrier_conversion_in = lp.merge(
            term_carrier_conversion_in, dim="group", join="outer", cls=LinearExpression
        )
        term_carrier_conversion_in = (
            model_constructor.optimization_model.constraints.reorder_group(
                term_carrier_conversion_in,
                None,
                None,
                index.get_unique([0]),
                first_index_name,
                model_constructor.optimization_model.lp_model,
            )
        )
        term_carrier_conversion_out = lp.merge(
            term_carrier_conversion_out, dim="group", join="outer", cls=LinearExpression
        )
        term_carrier_conversion_out = (
            model_constructor.optimization_model.constraints.reorder_group(
                term_carrier_conversion_out,
                None,
                None,
                index.get_unique([0]),
                first_index_name,
                model_constructor.optimization_model.lp_model,
            )
        )

        # carrier flow storage technologies
        if (
            model_constructor.optimization_model.variables[
                "flow_storage_discharge"
            ].size
            > 0
        ):
            term_flow_storage_discharge = []
            term_flow_storage_charge = []
            for carrier in index.get_unique([0]):
                storage_techs = [
                    tech
                    for tech in model_constructor.optimization_model.sets[
                        "set_storage_technologies"
                    ]
                    if carrier
                    in model_constructor.optimization_model.sets[
                        "set_reference_carriers"
                    ][tech]
                ]
                term_flow_storage_discharge.append(
                    model_constructor.optimization_model.variables[
                        "flow_storage_discharge"
                    ]
                    .loc[storage_techs]
                    .sum("set_storage_technologies")
                )
                term_flow_storage_charge.append(
                    model_constructor.optimization_model.variables[
                        "flow_storage_charge"
                    ]
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
            term_flow_storage_discharge = (
                model_constructor.optimization_model.constraints.reorder_group(
                    term_flow_storage_discharge,
                    None,
                    None,
                    index.get_unique([0]),
                    first_index_name,
                    model_constructor.optimization_model.lp_model,
                )
            )
            term_flow_storage_charge = lp.merge(
                term_flow_storage_charge,
                dim="group",
                join="outer",
                cls=LinearExpression,
            )
            term_flow_storage_charge = (
                model_constructor.optimization_model.constraints.reorder_group(
                    term_flow_storage_charge,
                    None,
                    None,
                    index.get_unique([0]),
                    first_index_name,
                    model_constructor.optimization_model.lp_model,
                )
            )
        else:
            # if there is no carrier flow we just create empty arrays
            term_flow_storage_discharge = (
                model_constructor.optimization_model.variables["flow_import"]
                .where(xr.DataArray(False))
                .to_linexpr()
            )
            term_flow_storage_charge = (
                model_constructor.optimization_model.variables["flow_import"]
                .where(xr.DataArray(False))
                .to_linexpr()
            )

        # carrier import, demand and export
        term_carrier_import = model_constructor.optimization_model.variables[
            "flow_import"
        ].to_linexpr()
        term_carrier_export = model_constructor.optimization_model.variables[
            "flow_export"
        ].to_linexpr()
        term_carrier_demand = model_constructor.optimization_model.parameters.demand
        # shed demand
        term_carrier_shed_demand = model_constructor.optimization_model.variables[
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
        model_constructor.optimization_model.add_constraint(
            "constraint_nodal_energy_balance", constraints
        )
