from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zen_garden.postprocess.results.results import Results, CostEmissionMode

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import splu
import os
from typing import Optional
import pickle
from pathlib import Path

REQUIRED_COMPONENTS = {
    "parameter": ["demand", "carbon_intensity_technology"],
    "variable": [
        "cost_capex_yearly",
        "cost_opex_yearly",
        "flow_conversion_input",
        "flow_conversion_output",
        "shed_demand",
        "flow_import",
        "flow_export",
        "cost_carrier",
        "cost_shed_demand",
        "carbon_emissions_technology",
        "carbon_emissions_carrier",
    ],
}
REQUIRED_TRANSPORT_COMPONENTS = ["flow_transport", "flow_transport_loss"]
REQUIRED_STORAGE_COMPONENTS = ["flow_storage_charge", "flow_storage_discharge"]


class CostEmissionCalculation:
    def __init__(self, r: Results):
        self.r = r
        self.path = Path(r.solution_loader.path)
        self._leontief_cache = {"cost": {}, "emissions": {}}
        self.conversion_technologies = self.r.get_system().set_conversion_technologies
        self.transport_technologies = self.r.get_system().set_transport_technologies
        self.storage_technologies = self.r.get_system().set_storage_technologies

    def _get_leontief_raw_totals(self, scenario_name: str) -> dict[str, pd.DataFrame]:
        """fetch all raw totals needed to build the Leontief cost system.

        Args:
            scenario_name: The name of the scenario

        Returns:
            A dictionary containing the raw totals needed to
            build the Leontief cost system.

        """
        for component_type in REQUIRED_COMPONENTS:
            all_components = self.r.get_component_names(component_type=component_type)
            for required_component in REQUIRED_COMPONENTS[component_type]:
                if required_component not in all_components:
                    raise ValueError(
                        f"Required component '{required_component}' "
                        f"not found in scenario '{scenario_name}'."
                        " This component is required for cost/emission calculations."
                    )

        capex = (
            self.r.get_total("cost_capex_yearly", scenario_name=scenario_name)
            .groupby(["technology", "location"])
            .sum()
        )
        opex = (
            self.r.get_total("cost_opex_yearly", scenario_name=scenario_name)
            .groupby(["technology", "location"])
            .sum()
        )
        flow_in_conversion = (
            self.r.get_total("flow_conversion_input", scenario_name=scenario_name)
            .groupby(["technology", "carrier", "node"])
            .sum()
        )
        flow_out_conversion = (
            self.r.get_total("flow_conversion_output", scenario_name=scenario_name)
            .groupby(["technology", "carrier", "node"])
            .sum()
        )
        demand = (
            self.r.get_total("demand", scenario_name=scenario_name)
            .groupby(["carrier", "node"])
            .sum()
        )
        shed_demand = (
            self.r.get_total("shed_demand", scenario_name=scenario_name)
            .groupby(["carrier", "node"])
            .sum()
        )
        flow_import = (
            self.r.get_total("flow_import", scenario_name=scenario_name)
            .groupby(["carrier", "node"])
            .sum()
        )
        flow_export = (
            self.r.get_total("flow_export", scenario_name=scenario_name)
            .groupby(["carrier", "node"])
            .sum()
        )
        cost_carrier = (
            self.r.get_total("cost_carrier", scenario_name=scenario_name)
            .groupby(["carrier", "node"])
            .sum()
        )
        cost_shed_demand = (
            self.r.get_total("cost_shed_demand", scenario_name=scenario_name)
            .groupby(["carrier", "node"])
            .sum()
        )
        carbon_emissions_technology = (
            self.r.get_total("carbon_emissions_technology", scenario_name=scenario_name)
            .groupby(["technology", "location"])
            .sum()
        )
        carbon_intensity_technology = (
            self.r.get_total("carbon_intensity_technology", scenario_name=scenario_name)
            .groupby(["technology", "location"])
            .sum()
        )
        carbon_emissions_carrier = (
            self.r.get_total("carbon_emissions_carrier", scenario_name=scenario_name)
            .groupby(["carrier", "node"])
            .sum()
        )
        if self.transport_technologies:
            for component in REQUIRED_TRANSPORT_COMPONENTS:
                if component not in self.r.get_component_names(
                    component_type="variable"
                ):
                    raise ValueError(
                        f"Required transport variable '{component}' "
                        f"not found in scenario '{scenario_name}'."
                        " This component is required for cost/emission calculations."
                    )
            flow_transport = (
                self.r.get_total("flow_transport", scenario_name=scenario_name)
                .groupby(["technology", "edge"])
                .sum()
            )
            flow_transport_loss = (
                self.r.get_total("flow_transport_loss", scenario_name=scenario_name)
                .groupby(["technology", "edge"])
                .sum()
            )
        else:
            empty_index = pd.MultiIndex.from_arrays(
                [[], []], names=["technology", "edge"]
            )
            flow_transport = pd.DataFrame(
                index=empty_index, columns=self.r.get_years(scenario_name=scenario_name)
            )
            flow_transport_loss = pd.DataFrame(
                index=empty_index, columns=self.r.get_years(scenario_name=scenario_name)
            )
        if self.storage_technologies:
            for component in REQUIRED_STORAGE_COMPONENTS:
                if component not in self.r.get_component_names(
                    component_type="variable"
                ):
                    raise ValueError(
                        f"Required storage variable '{component}' "
                        f"not found in scenario '{scenario_name}'."
                        " This component is required for cost/emission calculations."
                    )
            flow_storage_charge = (
                self.r.get_total("flow_storage_charge", scenario_name=scenario_name)
                .groupby(["technology", "node"])
                .sum()
            )
            flow_storage_discharge = (
                self.r.get_total("flow_storage_discharge", scenario_name=scenario_name)
                .groupby(["technology", "node"])
                .sum()
            )
        else:
            empty_index = pd.MultiIndex.from_arrays(
                [[], []], names=["technology", "node"]
            )
            flow_storage_charge = pd.DataFrame(
                index=empty_index, columns=self.r.get_years(scenario_name=scenario_name)
            )
            flow_storage_discharge = pd.DataFrame(
                index=empty_index, columns=self.r.get_years(scenario_name=scenario_name)
            )
        return {
            "capex": capex,
            "opex": opex,
            "flow_in_conversion": flow_in_conversion,
            "flow_out_conversion": flow_out_conversion,
            "demand": demand,
            "shed_demand": shed_demand,
            "flow_import": flow_import,
            "flow_export": flow_export,
            "cost_carrier": cost_carrier,
            "cost_shed_demand": cost_shed_demand,
            "carbon_emissions_technology": carbon_emissions_technology,
            "carbon_intensity_technology": carbon_intensity_technology,
            "carbon_emissions_carrier": carbon_emissions_carrier,
            "flow_transport": flow_transport,
            "flow_transport_loss": flow_transport_loss,
            "flow_storage_charge": flow_storage_charge,
            "flow_storage_discharge": flow_storage_discharge,
        }

    def _build_leontief_X(
        self,
        raw: dict[str, pd.DataFrame],
        sector_index: pd.MultiIndex,
        nodes_on_edges: list[str],
        ref_carrier: str,
        rtol: float = 1e-4,
        atol: float = 1e-3,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """total gross throughput per (carrier, node) sector, per year, from the
        use side (served demand + local intermediate use + gross transport-out +
        storage charge + export).

        Args:
            raw: A dictionary containing the raw totals needed to
                build the Leontief cost system.
            sector_index: A MultiIndex of (carrier, node) representing
                all sectors with recorded flows.
            nodes_on_edges: A list of nodes on edges.
            ref_carrier: A Series mapping each technology to its reference carrier.
            rtol: Relative tolerance for self-test.
            atol: Absolute tolerance for self-test.

        Returns:
            A tuple of two DataFrames:
                - X_use: A DataFrame with a MultiIndex of (carrier, node) representing
                    the total gross throughput per sector from the use side.
                - demand_served: A DataFrame with a MultiIndex of (carrier, node)
                    representing the total demand served per sector.
        """
        idx = sector_index
        demand_served = (
            (
                raw["demand"]
                - raw["shed_demand"].reindex(raw["demand"].index)
                + raw["flow_export"].reindex(raw["demand"].index)
            )
            .fillna(0)
            .reindex(idx)
            .fillna(0)
        )
        flow_in_conversion_local = (
            raw["flow_in_conversion"]
            .groupby(level=["carrier", "node"])
            .sum()
            .reindex(idx)
            .fillna(0)
        )
        flow_out_conversion_local = (
            raw["flow_out_conversion"]
            .groupby(level=["carrier", "node"])
            .sum()
            .reindex(idx)
            .fillna(0)
        )
        # export = raw["flow_export"].reindex(idx).fillna(0)
        imp = raw["flow_import"].reindex(idx).fillna(0)
        storage_charge = (
            self._leontief_storage_by_carrier_node(
                raw["flow_storage_charge"], ref_carrier
            )
            .reindex(idx)
            .fillna(0)
        )
        storage_discharge = (
            self._leontief_storage_by_carrier_node(
                raw["flow_storage_discharge"], ref_carrier
            )
            .reindex(idx)
            .fillna(0)
        )
        transport_out_gross, transport_in_net = (
            self._leontief_transport_by_carrier_node(
                raw["flow_transport"],
                raw["flow_transport_loss"],
                nodes_on_edges,
                ref_carrier,
            )
        )
        transport_out_gross = transport_out_gross.reindex(idx).fillna(0)
        transport_in_net = transport_in_net.reindex(idx).fillna(0)

        X_use = (
            demand_served
            + flow_in_conversion_local
            + transport_out_gross
            + storage_charge
        )
        X_supply = (
            flow_out_conversion_local + transport_in_net + imp + storage_discharge
        )

        emissions_tech_pos = (
            raw["carbon_emissions_technology"]
            .clip(lower=0)
            .sum()
            .to_frame(name="emissions")
            .T
        )
        emissions_carrier_pos = (
            raw["carbon_emissions_carrier"]
            .clip(lower=0)
            .sum()
            .to_frame(name="emissions")
            .T
        )

        emissions_tech_pos.index = pd.MultiIndex.from_tuples(
            [("emissions", "global")], names=["carrier", "node"]
        )
        emissions_carrier_pos.index = pd.MultiIndex.from_tuples(
            [("emissions", "global")], names=["carrier", "node"]
        )
        emissions_pos = emissions_tech_pos.add(emissions_carrier_pos, fill_value=0)
        X_use.loc[("emissions", "global")] = emissions_pos.loc[("emissions", "global")]
        X_supply.loc[("emissions", "global")] = emissions_pos.loc[
            ("emissions", "global")
        ]

        abs_diff = (X_use - X_supply).abs()
        tolerance = atol + rtol * X_use.abs()
        bad = abs_diff[(abs_diff > tolerance).any(axis=1)]
        if len(bad) > 0:
            print(
                f"WARNING: Leontief X use-side/supply-side self-test mismatch for "
                f"{len(bad)} sector(s) (max abs diff {abs_diff.to_numpy().max():.3g}, "
                f"tol=atol+rtol*|X|). This indicates a bug in this aggregation, since "
                f"the nodal energy balance constraint guarantees these match. First "
                f"offenders: {list(bad.index[:10])}"
            )
        return X_use.astype(float), demand_served.astype(float)

    def _get_or_build_leontief_systems(
        self, 
        scenario_name: str, overwrite: bool=False, is_cost: bool=True
    ) -> dict[int, dict[str, any]]:
        """builds or loads the Leontief system for every year of a scenario. 
        
        Args:
            scenario_name: The name of the scenario
            overwrite: If True, the Leontief system will be rebuilt even if 
                it exists in the cache.
            is_cost: If True, the cost Leontief system will be built. 
                If False, the emissions Leontief system will be built.

        Returns:
            A dictionary mapping each year to its corresponding Leontief system.
        """
        cache_key = scenario_name
        if is_cost:
            str_type = "cost"
            if cache_key in self._leontief_cache["cost"]:
                return self._leontief_cache["cost"][cache_key]
        else:
            str_type = "emissions"
            if cache_key in self._leontief_cache["emissions"]:
                return self._leontief_cache["emissions"][cache_key]

        cache_file = None
        if self.path is not None:
            cache_file = (
                self.path
                / f"{self.r.name}_{scenario_name}_{str_type}_leontief_system.pickle"
            )
        if cache_file is not None and os.path.exists(cache_file) and not overwrite:
            with open(cache_file, "rb") as f:
                picklable_state = pickle.load(f)
            systems = {}
            for step, state in picklable_state.items():
                lu = splu(identity(state["A"].shape[0], format="csc") - state["A"])
                systems[step] = {**state, "lu": lu}
            if is_cost:
                self._leontief_cache["cost"][cache_key] = systems
            else:
                self._leontief_cache["emissions"][cache_key] = systems
            return systems

        ref_carrier = self.r.get_df(
            "set_reference_carriers", scenario_name=scenario_name
        ).squeeze()
        nodes_on_edges = self.r.get_df(
            "set_nodes_on_edges", scenario_name=scenario_name
        ).str.split(",", expand=True)
        raw = self._get_leontief_raw_totals(scenario_name)
        sector_index = self._build_leontief_sector_index(raw)
        X, demand_served = self._build_leontief_X(
            raw, sector_index, nodes_on_edges, ref_carrier
        )

        input_carriers = self.r.get_df(
            "set_input_carriers", scenario_name=scenario_name
        )
        output_carriers = self.r.get_df(
            "set_output_carriers", scenario_name=scenario_name
        )
        set_nodes = self.r.get_system(scenario_name=scenario_name).set_nodes
        systems, picklable_state = {}, {}
        for year in self.r.get_years(scenario_name=scenario_name):
            if is_cost:
                fkt = self._build_leontief_cost_year_system
            else:
                fkt = self._build_leontief_emissions_year_system
            sys_step = fkt(
                year,
                raw,
                ref_carrier,
                input_carriers,
                output_carriers,
                set_nodes,
                nodes_on_edges,
                sector_index,
                X[year],
                demand_served[year],
            )
            systems[year] = sys_step
            picklable_state[year] = {k: v for k, v in sys_step.items() if k != "lu"}

        if cache_file is not None:
            with open(cache_file, "wb") as f:
                pickle.dump(picklable_state, f)
        if is_cost:
            self._leontief_cache["cost"][cache_key] = systems
        else:
            self._leontief_cache["emissions"][cache_key] = systems
        return systems

    def _leontief_solve_targets(
            self, sys_step: dict, target_sectors: list, is_cost: bool=True
        ) -> np.ndarray:
        """solves (I-A) x = e_j for each requested target sector j (batched into a
        single sparse solve), returning the component-by-target contribution matrix
        (rows=(component,cost_type), columns=target_sectors) as a dense numpy array.
        
        Args:
            sys_step: The Leontief system for a specific year.
            target_sectors: A list of target sectors to solve for.
            is_cost: If True, the cost Leontief system will be used ('v_matrix'); 
                otherwise, the emissions system will be used ('e_matrix').
        
        Returns:
            A dense numpy array representing the component-by-target contribution
            matrix, where rows correspond to (component, cost_type) and columns
            correspond to target_sectors.
        
        """
        if is_cost:
            value_key = "v_matrix"
        else:
            value_key = "e_matrix"
        S = len(sys_step["pos_to_sector"])
        positions = [sys_step["sector_to_pos"][s] for s in target_sectors]
        E = np.zeros((S, len(positions)))
        for col, pos in enumerate(positions):
            E[pos, col] = 1.0
        Xsol = sys_step["lu"].solve(E)
        return sys_step[value_key].dot(Xsol)

    def _leontief_assemble_frame(
        self, 
        contrib: np.ndarray, 
        comp_to_pos: dict, 
        target_sectors: list, 
        spatially_resolved: bool, 
        is_cost: bool
        ) -> pd.DataFrame:
        """ Turns a (component, target) contribution array into a (component, cost_type,
        origin_node[, node]) indexed Series for one year. 
        
        Args:
            contrib: A numpy array of contributions from components to target sectors.
            comp_to_pos: A dictionary mapping (component, cost_type, origin_node) to
                their corresponding row positions in the contrib array.
            target_sectors: A list of target sectors corresponding to the columns of
                the contrib array.
            spatially_resolved: If True, the output will be spatially resolved
                (indexed by carrier and node). If False, the output will be aggregated
                over nodes (indexed by carrier only).
            is_cost: If True, the output will be for cost contributions. If False,
                the output will be for emissions contributions.
        
        Returns:
            A pandas DataFrame representing the contributions from components to
                target sectors, indexed by (component, cost_type, origin_node[, node]).      
        
        """
        if is_cost:
            value_type_name = "cost_type"
        else:
            value_type_name = "emission_type"

        comp_index = pd.MultiIndex.from_tuples(
            list(comp_to_pos.keys()),
            names=["component", value_type_name, "origin_node"],
        )
        comp_order = np.argsort(list(comp_to_pos.values()))
        comp_index = comp_index[comp_order]
        columns = pd.MultiIndex.from_tuples(target_sectors, names=["carrier", "node"])
        out = pd.DataFrame(contrib, index=comp_index, columns=columns)
        if spatially_resolved:
            out = out.stack(["carrier", "node"], future_stack=True)
            out = out.reorder_levels(
                ["carrier", "node", "component", value_type_name, "origin_node"]
            ).sort_index()
        else:
            out = out.T.groupby(level="carrier").sum().T
            out = out.stack("carrier", future_stack=True)
            out = out.reorder_levels(
                ["carrier", "component", value_type_name, "origin_node"]
            ).sort_index()
            out = out.groupby(level=["carrier", "component", value_type_name]).sum()

        return out[out != 0]

    def _build_leontief_cost_year_system(
        self,
        step: int,
        raw: dict[str, pd.DataFrame],
        ref_carrier: pd.Series,
        input_carriers: list[str],
        output_carriers: list[str],
        set_nodes: list[str],
        nodes_on_edges: pd.DataFrame,
        sector_index: list[str],
        X_year: pd.Series,
        demand_served_year: pd.Series,
        eps: float = 1e-6,
    ) -> dict[str, any]:
        """ builds the sparse technical-coefficient matrix A, the component-by-sector
        direct-cost matrix V_comp, and the LU factorization of (I - A), for a single
        optimization year (a single column `step` of the totals in `raw`). 
        
        Args:
            step: The optimization year for which the Leontief system is being built.
            raw: A dictionary containing the raw totals needed to 
                build the Leontief cost system.
            ref_carrier: A Series mapping each technology to its reference carrier.
            input_carriers: A list of input carriers for conversion technologies.
            output_carriers: A list of output carriers for conversion technologies.
            set_nodes: A list of nodes in the system.
            nodes_on_edges: A DataFrame of nodes on edges in the system.
            sector_index: A MultiIndex of (carrier, node) representing
                all sectors with recorded flows.
            X_year: A Series representing the total gross throughput per sector
                from the use side for the given year.
            demand_served_year: A Series representing the total demand served per sector
                for the given year.
            eps: A small threshold value to determine active sectors.

        Returns:
            A dictionary containing the following keys:
                - "A": The sparse technical-coefficient matrix A.
                - "v_matrix": The component-by-sector direct-cost matrix V_comp.
                - "lu": The LU factorization of (I - A).        
        """
        active_sectors = [s for s in sector_index if X_year.loc[s] > eps]
        sector_to_pos = {s: i for i, s in enumerate(active_sectors)}
        S = len(active_sectors)
        X_active = X_year.loc[active_sectors]

        A_rows, A_cols, A_vals = [], [], []
        V_rows, V_cols, V_vals = [], [], []
        comp_to_pos = {}
        tech_flow_in_conversion = (
            {}
        )  # (tech, input_carrier, node) -> flow value, kept for single_tech isolation

        def add_v(component, cost_type, sector, value):
            if value == 0 or sector not in sector_to_pos:
                return
            key = (component, cost_type, sector[1])
            pos = comp_to_pos.setdefault(key, len(comp_to_pos))
            V_rows.append(pos)
            V_cols.append(sector_to_pos[sector])
            V_vals.append(value)

        def add_a(sector_in, sector_out, value):
            if (
                value == 0
                or sector_in not in sector_to_pos
                or sector_out not in sector_to_pos
            ):
                return
            A_rows.append(sector_to_pos[sector_in])
            A_cols.append(sector_to_pos[sector_out])
            A_vals.append(value)

        # --- conversion technologies (node-local only) ---
        for tech in self.conversion_technologies:
            if tech not in ref_carrier.index:
                continue
            c_out = self._get_carriers_of_tech(output_carriers, tech)
            c_ref = ref_carrier.loc[tech]
            sector_out_preset = None
            if len(c_out) == 1:
                c_out = c_out[0]
            elif c_ref in c_out:
                c_out = c_ref
            elif (
                len(c_out) == 0
                and raw["carbon_intensity_technology"].loc[tech].sum() < 0
            ):
                sector_out_preset = ("emissions", "global")
            else:
                raise NotImplementedError(
                    f"Leontief cost calculation not implemented for technology {tech} "
                    f"with multiple output carriers {c_out} that does not include its "
                    f"reference carrier {c_ref} or that is a carbon sink."
                )
            in_cs = self._get_carriers_of_tech(input_carriers, tech)
            for node in set_nodes:
                key = (tech, node)
                if key not in raw["capex"].index:
                    continue
                if sector_out_preset is None:
                    sector_out = (c_out, node)
                else:
                    sector_out = sector_out_preset
                if sector_out not in sector_to_pos:
                    continue
                add_v(tech, "capex", sector_out, raw["capex"].loc[key, step])
                add_v(tech, "opex", sector_out, raw["opex"].loc[key, step])
                x_out = X_active.loc[sector_out]
                for c_in in in_cs:
                    fi_key = (tech, c_in, node)
                    if fi_key not in raw["flow_in_conversion"].index:
                        continue
                    flow_val = raw["flow_in_conversion"].loc[fi_key, step]
                    if flow_val == 0:
                        continue
                    tech_flow_in_conversion[(tech, c_in, node)] = flow_val
                    sector_in = (c_in, node)

                    if sector_in not in sector_to_pos or x_out <= eps:
                        continue
                    add_a(sector_in, sector_out, flow_val / x_out)
                # emissions
                if key in raw["carbon_emissions_technology"].index:
                    if raw["carbon_emissions_technology"].loc[key, step] < 0:
                        continue
                    add_a(
                        ("emissions", "global"),
                        sector_out,
                        raw["carbon_emissions_technology"].loc[key, step] / x_out,
                    )
        # --- storage technologies: direct capex/opex ---
        # --- add A entries for charge flow, which is the only flow that creates a
        # --- cross-node dependency ---
        for tech in self.storage_technologies:
            if tech not in ref_carrier.index:
                continue
            c_ref = ref_carrier.loc[tech]
            for node in set_nodes:
                key = (tech, node)
                if key in raw["flow_storage_charge"].index:
                    flow_charge = raw["flow_storage_charge"].loc[key, step]
                    add_a(
                        (c_ref, node),
                        (c_ref, node),
                        flow_charge / X_active.loc[(c_ref, node)],
                    )
                if key not in raw["capex"].index:
                    continue
                sector = (c_ref, node)
                if sector not in sector_to_pos:
                    continue
                add_v(tech, "capex", sector, raw["capex"].loc[key, step])
                add_v(tech, "opex", sector, raw["opex"].loc[key, step])

        # --- transport technologies: capex/opex split 50/50 across endpoints; ---
        # --- gross flow creates the cross-node A entry that resolves cycles.  ---
        for tech in self.transport_technologies:
            if tech not in ref_carrier.index:
                continue
            c_ref = ref_carrier.loc[tech]
            for edge in nodes_on_edges.index:
                key = (tech, edge)
                if key not in raw["capex"].index:
                    continue
                n_from, n_to = nodes_on_edges.loc[edge, 0], nodes_on_edges.loc[edge, 1]
                sector_from, sector_to = (c_ref, n_from), (c_ref, n_to)
                capex_val, opex_val = (
                    raw["capex"].loc[key, step],
                    raw["opex"].loc[key, step],
                )
                add_v(tech, "capex", sector_from, capex_val / 2)
                add_v(tech, "opex", sector_from, opex_val / 2)
                add_v(tech, "capex", sector_to, capex_val / 2)
                add_v(tech, "opex", sector_to, opex_val / 2)
                flow_key = (tech, edge)
                if (
                    flow_key not in raw["flow_transport"].index
                    or sector_to not in sector_to_pos
                ):
                    continue
                flow_val = raw["flow_transport"].loc[flow_key, step]
                x_to = X_active.loc[sector_to] if sector_to in X_active.index else 0
                if flow_val == 0 or x_to <= eps:
                    continue
                add_a(sector_from, sector_to, flow_val / x_to)

        # --- carrier import/export cost and (optionally) shed demand penalty ---
        for sector in active_sectors:
            c, n = sector
            ck = (c, n)
            if ck in raw["cost_carrier"].index:
                add_v(c, "fuel", sector, raw["cost_carrier"].loc[ck, step])
            if ck in raw["cost_shed_demand"].index:
                add_v(c, "shed_demand", sector, raw["cost_shed_demand"].loc[ck, step])
            # add emissions
            x_active = X_active.loc[ck]
            if ck in raw["carbon_emissions_carrier"].index:
                add_a(
                    ("emissions", "global"),
                    sector,
                    raw["carbon_emissions_carrier"].loc[ck, step] / x_active,
                )

        A = coo_matrix((A_vals, (A_rows, A_cols)), shape=(S, S)).tocsc()
        n_comp = len(comp_to_pos)
        V_comp = coo_matrix((V_vals, (V_rows, V_cols)), shape=(n_comp, S)).tocsr()
        inv_X = 1.0 / X_active.to_numpy()
        v_matrix = V_comp.multiply(inv_X[np.newaxis, :]).tocsr()

        I_minus_A = identity(S, format="csc") - A
        lu = splu(I_minus_A)
        min_pivot = np.abs(lu.U.diagonal()).min() if S > 0 else np.inf
        if min_pivot < 1e-8:
            print(
                f"WARNING: Leontief system for year step {step} is near-singular "
                f"(smallest pivot {min_pivot:.3g}) -- results may be unreliable."
            )

        demand_from_AX = X_active - A.dot(X_active)
        diff_demand = (
            demand_served_year.reindex_like(demand_from_AX) - demand_from_AX
        ).abs()
        eps_diff = 1e-3
        assert (diff_demand <= eps_diff).all(), (
            f"Leontief demand self-test failed for year step {step}: max abs diff "
            f"{diff_demand.max():.3g} > {eps_diff}. This indicates a bug in the "
            f"A-matrix assembly."
        )
        return {
            "sector_to_pos": sector_to_pos,
            "pos_to_sector": active_sectors,
            "comp_to_pos": comp_to_pos,
            "X": X_active,
            "A": A,
            "V_comp": V_comp,
            "v_matrix": v_matrix,
            "lu": lu,
            "tech_flow_in_conversion": tech_flow_in_conversion,
            "demand_served": demand_served_year,
        }

    def _build_leontief_emissions_year_system(
        self,
        step: int,
        raw: dict[str, pd.DataFrame],
        ref_carrier: pd.DataFrame,
        input_carriers: pd.DataFrame,
        output_carriers: pd.DataFrame,
        set_nodes: list,
        nodes_on_edges: pd.DataFrame,
        sector_index: pd.MultiIndex,
        X_year: pd.Series,
        demand_served_year: pd.Series,
        eps=1e-6,
    ):
        """ Counterpart to _build_leontief_cost_year_system for embodied-
        emissions tracing. 
        
        Builds the sparse technical-coefficient matrix A, the component-by-sector
        direct-emissions matrix E_comp, and the LU factorization of (I - A), for a single
        optimization year (a single column `step` of the totals in `raw`). 

        Args:
            step: The optimization year for which the Leontief system is being built.
            raw: A dictionary containing the raw totals needed to build the Leontief 
                emissions system.
            ref_carrier: A Series mapping each technology to its reference carrier.
            input_carriers: A list of input carriers for conversion technologies.
            output_carriers: A list of output carriers for conversion technologies.
            set_nodes: A list of nodes in the system.
            nodes_on_edges: A DataFrame of nodes on edges in the system.
            sector_index: A MultiIndex of (carrier, node) representing
                all sectors with recorded flows.
            X_year: A Series representing the total gross throughput per sector
                from the use side for the given year.
            demand_served_year: A Series representing the total demand served per sector
                for the given year.

        Returns:
            A dictionary containing the following keys:
                - "A": The sparse technical-coefficient matrix A.
                - "e_matrix": The component-by-sector direct-emissions matrix E_comp.
                - "lu": The LU factorization of (I - A).
        """
        active_sectors = [s for s in sector_index if X_year.loc[s] > eps]
        sector_to_pos = {s: i for i, s in enumerate(active_sectors)}
        S = len(active_sectors)
        X_active = X_year.loc[active_sectors]

        A_rows, A_cols, A_vals = [], [], []
        E_rows, E_cols, E_vals = [], [], []
        comp_to_pos = {}
        tech_flow_in_conversion = {}
        tech_sector_out_preset = {}

        def add_e(component, emission_type, sector, value):
            if value == 0 or sector not in sector_to_pos:
                return
            key = (component, emission_type, sector[1])
            pos = comp_to_pos.setdefault(key, len(comp_to_pos))
            E_rows.append(pos)
            E_cols.append(sector_to_pos[sector])
            E_vals.append(value)

        def add_a(sector_in, sector_out, value):
            if (
                value == 0
                or sector_in not in sector_to_pos
                or sector_out not in sector_to_pos
            ):
                return
            A_rows.append(sector_to_pos[sector_in])
            A_cols.append(sector_to_pos[sector_out])
            A_vals.append(value)

        # --- conversion technologies:
        for tech in self.conversion_technologies:
            if tech not in ref_carrier.index:
                continue
            c_out = self._get_carriers_of_tech(output_carriers, tech)
            c_ref = ref_carrier.loc[tech]
            sector_out_preset = None
            if len(c_out) == 1:
                c_out = c_out[0]
            elif c_ref in c_out:
                c_out = c_ref
            elif (
                len(c_out) == 0
                and raw["carbon_intensity_technology"].loc[tech].sum() < 0
            ):
                sector_out_preset = ("emissions", "global")
            else:
                raise NotImplementedError(
                    f"Leontief emissions calculation not implemented for technology "
                    f"{tech} with multiple output carriers {c_out} that does not "
                    f"include its reference carrier {c_ref} or that is a carbon sink."
                )
            tech_sector_out_preset[tech] = sector_out_preset
            in_cs = self._get_carriers_of_tech(input_carriers, tech)
            for node in set_nodes:
                key = (tech, node)
                if key not in raw["capex"].index:
                    continue
                if sector_out_preset is None:
                    sector_out = (c_out, node)
                else:
                    sector_out = sector_out_preset
                if sector_out not in sector_to_pos:
                    continue
                if key in raw["carbon_emissions_technology"].index:
                    add_e(
                        tech,
                        "process",
                        sector_out,
                        raw["carbon_emissions_technology"].loc[key, step],
                    )
                x_out = X_active.loc[sector_out]
                for c_in in in_cs:
                    fi_key = (tech, c_in, node)
                    if fi_key not in raw["flow_in_conversion"].index:
                        continue
                    flow_val = raw["flow_in_conversion"].loc[fi_key, step]
                    if flow_val == 0:
                        continue
                    tech_flow_in_conversion[(tech, c_in, node)] = flow_val
                    sector_in = (c_in, node)
                    if sector_in not in sector_to_pos or x_out <= eps:
                        continue
                    add_a(sector_in, sector_out, flow_val / x_out)
                if key in raw["carbon_emissions_technology"].index:
                    if raw["carbon_emissions_technology"].loc[key, step] < 0:
                        continue
                    add_a(
                        ("emissions", "global"),
                        sector_out,
                        raw["carbon_emissions_technology"].loc[key, step] / x_out,
                    )

        # --- storage technologies: same self-loop as cost; no direct emissions
        # --- booked ---
        for tech in self.storage_technologies:
            if tech not in ref_carrier.index:
                continue
            c_ref = ref_carrier.loc[tech]
            for node in set_nodes:
                key = (tech, node)
                if key in raw["flow_storage_charge"].index:
                    flow_charge = raw["flow_storage_charge"].loc[key, step]
                    add_a(
                        (c_ref, node),
                        (c_ref, node),
                        flow_charge / X_active.loc[(c_ref, node)],
                    )
                if key not in raw["carbon_emissions_technology"].index:
                    continue
                sector = (c_ref, node)
                if sector not in sector_to_pos:
                    continue
                add_e(
                    tech,
                    "storage",
                    sector,
                    raw["carbon_emissions_technology"].loc[key, step],
                )
        # --- transport technologies: same gross-flow edge as cost; no direct
        # --- emissions ---
        for tech in self.transport_technologies:
            if tech not in ref_carrier.index:
                continue
            c_ref = ref_carrier.loc[tech]
            for edge in nodes_on_edges.index:
                key = (tech, edge)
                if key not in raw["carbon_emissions_technology"].index:
                    continue
                n_from, n_to = nodes_on_edges.loc[edge, 0], nodes_on_edges.loc[edge, 1]
                sector_from, sector_to = (c_ref, n_from), (c_ref, n_to)
                add_e(
                    tech,
                    "transport",
                    sector_from,
                    raw["carbon_emissions_technology"].loc[key, step] / 2,
                )
                add_e(
                    tech,
                    "transport",
                    sector_to,
                    raw["carbon_emissions_technology"].loc[key, step] / 2,
                )
                flow_key = (tech, edge)
                if flow_key not in raw["flow_transport"].index:
                    continue
                flow_val = raw["flow_transport"].loc[flow_key, step]
                x_to = X_active.loc[sector_to] if sector_to in X_active.index else 0
                if flow_val == 0 or x_to <= eps or sector_to not in sector_to_pos:
                    continue
                add_a(sector_from, sector_to, flow_val / x_to)

        # --- carrier-level direct ("fuel") emissions:
        for sector in active_sectors:
            c, n = sector
            if sector in raw["carbon_emissions_carrier"].index:
                add_e(
                    c, "fuel", sector, raw["carbon_emissions_carrier"].loc[sector, step]
                )
                add_a(
                    ("emissions", "global"),
                    sector,
                    raw["carbon_emissions_carrier"].loc[sector, step]
                    / X_active.loc[sector],
                )

        A = coo_matrix((A_vals, (A_rows, A_cols)), shape=(S, S)).tocsc()
        n_comp = len(comp_to_pos)
        E_comp = coo_matrix((E_vals, (E_rows, E_cols)), shape=(n_comp, S)).tocsr()
        inv_X = 1.0 / X_active.to_numpy()
        e_matrix = E_comp.multiply(inv_X[np.newaxis, :]).tocsr()

        I_minus_A = identity(S, format="csc") - A
        lu = splu(I_minus_A)
        min_pivot = np.abs(lu.U.diagonal()).min() if S > 0 else np.inf
        if min_pivot < 1e-8:
            print(
                f"WARNING: Leontief emissions system for year step {step} is "
                f"near-singular (smallest pivot {min_pivot:.3g}) -- results may be "
                f"unreliable."
            )

        demand_from_AX = X_active - A.dot(X_active)
        diff_demand = (demand_served_year.loc[active_sectors] - demand_from_AX).abs()
        eps_diff = 1e-4
        assert (diff_demand <= eps_diff).all(), (
            f"Leontief emissions demand self-test failed for year step {step}: "
            f"max abs diff {diff_demand.max():.3g} > {eps_diff}. This indicates a bug "
            f"in the physical-only A-matrix assembly."
        )
        return {
            "sector_to_pos": sector_to_pos,
            "pos_to_sector": active_sectors,
            "comp_to_pos": comp_to_pos,
            "X": X_active,
            "A": A,
            "E_comp": E_comp,
            "e_matrix": e_matrix,
            "lu": lu,
            "tech_flow_in_conversion": tech_flow_in_conversion,
            "tech_sector_out_preset": tech_sector_out_preset,
            "demand_served": demand_served_year,
        }

    @staticmethod
    def _leontief_storage_by_carrier_node(
        flow_df: pd.DataFrame, ref_carrier: pd.Series
    ) -> pd.DataFrame:
        """ remaps a (technology, node) indexed storage flow to (carrier, node), using
        each storage technology's own reference carrier, summing technologies that
        share a (carrier, node) (e.g. two different storage techs for electricity
        at the same node).

        Args:
            flow_df: A DataFrame with a MultiIndex of (technology, node)
                representing the storage flows.
            ref_carrier: A Series mapping each technology to its reference carrier.

        Returns:
            A DataFrame with a MultiIndex of (carrier, node) representing the
                summed storage flows.
        """
        if len(flow_df) == 0:
            return flow_df.rename_axis(["carrier", "node"])
        techs = flow_df.index.get_level_values("technology")
        nodes = flow_df.index.get_level_values("node")
        carriers = ref_carrier.loc[techs].values
        out = flow_df.set_axis(
            pd.MultiIndex.from_arrays([carriers, nodes], names=["carrier", "node"])
        )
        return out.groupby(level=["carrier", "node"]).sum()

    @staticmethod
    def _leontief_transport_by_carrier_node(
        flow_transport: pd.DataFrame,
        flow_transport_loss: pd.DataFrame,
        nodes_on_edges: pd.DataFrame,
        ref_carrier: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """remaps (technology, edge) indexed transport flows to (carrier, node),
        returning (gross_out, net_in): gross_out(c,n) = total flow leaving n via
        transport of carrier c (before losses); net_in(c,n) = total flow arriving
        at n via transport of carrier c (net of losses).

        Args:
            flow_transport: A DataFrame with a MultiIndex of (technology, edge)
                representing the transport flows.
            flow_transport_loss: A DataFrame with a MultiIndex of (technology, edge)
                representing the transport losses.
            nodes_on_edges: A DataFrame mapping each edge to its corresponding nodes.
            ref_carrier: A Series mapping each technology to its reference carrier.

        Returns:
            A tuple of two DataFrames:
                - gross_out: A DataFrame with a MultiIndex of (carrier, node)
                    representing the total flow leaving each node via transport.
                - net_in: A DataFrame with a MultiIndex of (carrier, node)
                    representing the total flow arriving at each node via transport.

        """
        if len(flow_transport) == 0:
            empty = flow_transport.rename_axis(["carrier", "node"])
            return empty, empty
        techs = flow_transport.index.get_level_values("technology")
        edges = flow_transport.index.get_level_values("edge")
        carriers = ref_carrier.loc[techs].values
        n_from = nodes_on_edges.loc[edges, 0].values
        n_to = nodes_on_edges.loc[edges, 1].values
        flow_transport_in_net = flow_transport - flow_transport_loss.reindex(
            flow_transport.index
        ).fillna(0)
        gross_out = flow_transport.set_axis(
            pd.MultiIndex.from_arrays([carriers, n_from], names=["carrier", "node"])
        )
        net_in = flow_transport_in_net.set_axis(
            pd.MultiIndex.from_arrays([carriers, n_to], names=["carrier", "node"])
        )
        return (
            gross_out.groupby(level=["carrier", "node"]).sum(),
            net_in.groupby(level=["carrier", "node"]).sum(),
        )

    @staticmethod
    def _build_leontief_sector_index(raw: dict[str, pd.DataFrame]) -> pd.MultiIndex:
        """union of every (carrier, node) pair with any recorded flow
        Args:
            raw: A dictionary containing the raw totals needed to
                build the Leontief cost system.

        Returns:
            A MultiIndex of (carrier, node) representing
                all sectors with recorded flows.
        """
        idx = raw["demand"].index
        for key in ("flow_import", "flow_export", "cost_carrier"):
            idx = idx.union(raw[key].index)
        idx = idx.union(
            raw["flow_in_conversion"].groupby(level=["carrier", "node"]).sum().index
        )
        idx = idx.union(
            raw["flow_out_conversion"].groupby(level=["carrier", "node"]).sum().index
        )
        emissions_idx = pd.MultiIndex.from_tuples(
            [("emissions", "global")], names=["carrier", "node"]
        )
        idx = idx.union(emissions_idx)
        return idx.set_names(["carrier", "node"]).sort_values()

    @staticmethod
    def _get_carriers_of_tech(carrier_mapping: pd.Series, tech: str) -> list[str]:
        """returns the list of carriers associated with a given technology
        
        Args:
            carrier_mapping: A Series mapping each technology to its associated
                carriers (comma-separated string).
            tech: The technology for which to retrieve the associated carriers.

        Returns:
            A list of carriers associated with the given technology. If the
            technology has no associated carriers, an empty list is returned.
        """
        c = carrier_mapping.loc[tech]
        if c == "":
            return []
        return c.split(",")

    ### user-facing functions
    def calculate_leontief_data(
        self,
        scenario_name: str,
        carrier: Optional[str] = None,
        spatially_resolved: bool = False,
        mode: CostEmissionMode = "final_demand",
        overwrite: bool = False,
        is_cost: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calculates the sectoral data of a scenario through
        Leontief Input-Output tables. The capital and operational expenditures of each
        technology and the fuel cost of each carrier are allocated to the sectors
        that use them.

        When specifying a carrier, only the cost of producing that carrier is returned.
        Note that the tables are formulated for all sectors, so returning the data for
        all sectors does not add any overhead.
        The sectoral data are either returned aggregated over all locations or
        spatially resolved for each location (`spatially_resolved = True`).
        The data of transport technologies are 50/50 allocated to the connecting nodes.
        By default, the data to produce the final demand of each sector are returned
        (`mode = "final_demand"`), but the data of the total production of each carrier
        can also be returned (`mode = "total_production"`). Finally, the relative
        production data of each sector can be returned (`mode = "relative"`).

        Args:
            scenario_name: The scenario for which the sectoral data should be
                calculated
            carrier: The carrier for which the sectoral data should be calculated. If
                None, the data of all carriers are returned.
            spatially_resolved: Whether the sectoral data should be returned
                spatially resolved for each node or aggregated over all nodes.
            mode: The mode of calculation for the sectoral data
                ("final_demand", "total_production", or "relative").
            overwrite: Whether to rebuild the leontief input-output tables even if
                they have already been built and saved.
            is_cost: Whether to calculate data (True) or emissions (False).

        Returns:
            Tuple of two DataFrames:
                - The first DataFrame contains the total upstream/downstream data
                  (cost or emissions) of each sector.
                - The second DataFrame contains the direct data of each sector.
                  These are the costs or emissions that are directly associated with
                  the sector itself, without considering the upstream/downstream
                  effects.
        """
        if mode == "relative" and not spatially_resolved:
            raise ValueError(
                "Relative production data can only be returned "
                "spatially resolved. Please set `spatially_resolved = True`."
            )
        if scenario_name is None:
            scenario_name = next(iter(self.r.solution_loader.scenarios))
        if mode not in ["final_demand", "total_production", "relative"]:
            raise ValueError(
                f"Invalid mode {mode}. "
                "Must be one of 'final_demand', 'total_production', "
                "or 'relative'."
            )
        systems = self._get_or_build_leontief_systems(
            scenario_name, overwrite=overwrite, is_cost=is_cost
        )

        year_series, direct_year_series = {}, {}
        for year in self.r.get_years(scenario_name=scenario_name):
            sys_step = systems[year]
            if mode == "final_demand":
                qty_key = "demand_served"
            else:
                qty_key = "X"

            if carrier is None:
                target_qty = sys_step[qty_key].groupby(level=0).sum()
                target_qty = target_qty[target_qty > 0]
                target_sectors = [
                    (c, n)
                    for c in target_qty.index
                    for n in self.r.get_system(scenario_name=scenario_name).set_nodes
                    if (c, n) in sys_step["sector_to_pos"]
                ]
            else:
                target_sectors = [
                    (carrier, n)
                    for n in self.r.get_system(scenario_name=scenario_name).set_nodes
                    if (carrier, n) in sys_step["sector_to_pos"]
                ]
            if not target_sectors:
                year_series[year] = pd.Series(dtype=float)
                direct_year_series[year] = pd.Series(dtype=float)
                continue
            target_qty_by_node = {s: sys_step[qty_key].loc[s] for s in target_sectors}
            contrib = self._leontief_solve_targets(
                sys_step, target_sectors, is_cost=is_cost
            )
            if is_cost:
                matrix_key = "v_matrix"
            else:
                matrix_key = "e_matrix"
            direct = sys_step[matrix_key][
                :, [sys_step["sector_to_pos"][s] for s in target_sectors]
            ].toarray()

            qty = np.array([target_qty_by_node[s] for s in target_sectors])
            if mode == "relative":
                contrib_scaled = contrib
                direct_scaled = direct
            else:
                contrib_scaled = contrib * qty[np.newaxis, :]
                direct_scaled = direct * qty[np.newaxis, :]

            year_series[year] = self._leontief_assemble_frame(
                contrib_scaled,
                sys_step["comp_to_pos"],
                target_sectors,
                spatially_resolved,
                is_cost=is_cost,
            )
            direct_year_series[year] = self._leontief_assemble_frame(
                direct_scaled,
                sys_step["comp_to_pos"],
                target_sectors,
                spatially_resolved,
                is_cost=is_cost,
            )

        tot_data = pd.concat(year_series, axis=1).fillna(0)
        direct_data = pd.concat(direct_year_series, axis=1).fillna(0)
        return tot_data, direct_data
