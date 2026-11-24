"""Class is defining the postprocessing of the results.
The class takes as inputs the optimization problem (model) and the system
configurations (system). The class contains methods to read the results and
save them in a result dictionary (resultDict).
"""

import itertools
import json
import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Hashable, Literal, cast

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from filelock import FileLock
from linopy.expressions import LinearExpression
from linopy.variables import Variable
from tables import NaturalNameWarning

from zen_garden.elements.conversion_technology.sets.relation_conversion_technology_dependent_carrier import (  # noqa: E501
    RelationConversionTechnologyDependentCarrier,
)
from zen_garden.elements.conversion_technology.sets.relation_conversion_technology_input_carrier import (  # noqa: E501
    RelationConversionTechnologyInputCarrier,
)
from zen_garden.elements.conversion_technology.sets.relation_conversion_technology_output_carrier import (  # noqa: E501
    RelationConversionTechnologyOutputCarrier,
)
from zen_garden.elements.energy_system.sets.relation_edge_node import (
    RelationEdgeEndNode,
    RelationEdgeStartNode,
)
from zen_garden.elements.energy_system.sets.relation_year_representative_hour_storage import (  # noqa: E501
    RelationYearRepresentativeHourStorage,
)
from zen_garden.elements.energy_system.sets.set_representative_hours import (
    SetRepresentativeHours,
)
from zen_garden.elements.retrofitting_technology.sets.relation_retrofitting_technology_base_technology import (  # noqa: E501
    RelationRetrofittingTechnologyBaseTechnology,
)
from zen_garden.elements.technology.sets.relation_technology_existing_capacity import (
    RelationTechnologyExistingCapacity,
)
from zen_garden.elements.technology.sets.relation_technology_reference_carrier import (  # noqa: E501
    RelationTechnologyReferenceCarrier,
)
from zen_garden.model.registries.set import RelationSet
from zen_garden.model.registries.variable import align_mask

if TYPE_CHECKING:
    from zen_garden.input.unit_converter import UnitConverter
    from zen_garden.model.optimization_model import OptimizationModel
    from zen_garden.model.schema import ModelSchema
    from zen_garden.model.time_steps import TimeStepsDicts
    from zen_garden.workflow.scaling import Scaling

logger = logging.getLogger(__name__)

# Warnings
warnings.filterwarnings("ignore", category=NaturalNameWarning)

H5_COMP_LEVEL: int = 4
H5_COMP_LIB: Literal["zlib", "lzo", "bzip2", "blosc"] = "blosc"

ENERGY_PARAMETER_COMPONENTS = {
    "capacity_existing_energy": "capacity_existing",
    "capacity_investment_existing_energy": "capacity_investment_existing",
    "capacity_addition_min_energy": "capacity_addition_min",
    "capacity_addition_max_energy": "capacity_addition_max",
    "capacity_addition_unbounded_energy": "capacity_addition_unbounded",
    "capacity_limit_energy": "capacity_limit",
    "capacity_lower_limit_energy": "capacity_lower_limit",
    "opex_specific_fixed_energy": "opex_specific_fixed",
    "capex_specific_storage_energy": "capex_specific_storage",
    "capex_capacity_existing_energy": "capex_capacity_existing",
    "existing_capacities_energy": "existing_capacities",
    "existing_capex_energy": "existing_capex",
}

ENERGY_VARIABLE_COMPONENTS = {
    "capacity_energy": "capacity",
    "capacity_previous_energy": "capacity_previous",
    "capacity_addition_energy": "capacity_addition",
    "capacity_investment_energy": "capacity_investment",
    "cost_capex_overnight_energy": "cost_capex_overnight",
    "cost_capex_yearly_energy": "cost_capex_yearly",
    "technology_installation_energy": "technology_installation",
}

ENERGY_EXPRESSION_COMPONENTS = {
    "capacity_previous_energy": "capacity_previous",
    "cost_capex_overnight_energy": "cost_capex_overnight",
    "cost_capex_yearly_energy": "cost_capex_yearly",
}

ENERGY_DUAL_COMPONENTS = {
    "constraint_storage_energy_capacity_limit_not_reached": (
        "constraint_technology_capacity_limit_not_reached"
    ),
    "constraint_storage_energy_capacity_limit_reached": (
        "constraint_technology_capacity_limit_reached"
    ),
    "constraint_storage_energy_capacity_lower_limit": (
        "constraint_technology_capacity_lower_limit"
    ),
    "constraint_storage_energy_min_capacity_addition": (
        "constraint_technology_min_capacity_addition"
    ),
    "constraint_storage_energy_max_capacity_addition": (
        "constraint_technology_max_capacity_addition"
    ),
    "constraint_storage_energy_construction_time": (
        "constraint_technology_construction_time"
    ),
    "constraint_storage_energy_construction_time_outside": (
        "constraint_technology_construction_time_outside"
    ),
    "constraint_storage_energy_lifetime": "constraint_technology_lifetime",
    "constraint_storage_energy_lifetime_previous": (
        "constraint_technology_lifetime_previous"
    ),
    "constraint_storage_energy_diffusion_limit_total": (
        "constraint_technology_diffusion_limit_total"
    ),
    "constraint_storage_energy_diffusion_limit": (
        "constraint_technology_diffusion_limit"
    ),
    "constraint_cost_capex_yearly_energy": "constraint_cost_capex_yearly",
    "constraint_storage_technology_capex_energy": "constraint_storage_technology_capex",
}

LEGACY_INDEXED_SET_RELATIONS = {
    "set_input_carriers": RelationConversionTechnologyInputCarrier,
    "set_output_carriers": RelationConversionTechnologyOutputCarrier,
    "set_dependent_carriers": RelationConversionTechnologyDependentCarrier,
    "set_reference_carriers": RelationTechnologyReferenceCarrier,
    "set_retrofitting_base_technologies": (
        RelationRetrofittingTechnologyBaseTechnology
    ),
    "set_technologies_existing": RelationTechnologyExistingCapacity,
    "set_representative_hours_storage": RelationYearRepresentativeHourStorage,
}

LEGACY_COMPONENT_DIMENSIONS = {
    "flow_conversion_input": {"set_carriers": "set_input_carriers"},
    "flow_conversion_output": {"set_carriers": "set_output_carriers"},
    "conversion_factor": {"set_carriers": "set_dependent_carriers"},
    "constraint_carrier_conversion": {"set_carriers": "set_dependent_carriers"},
}

UNRELATED_INDEXES_FOR_UNITS = set(
    [
        "set_location",
        "set_nodes",
        "set_edges",
        "set_time_steps_operation",
        "set_time_steps_storage",
        "set_time_steps_yearly",
        "set_years",
        "set_hours",
        "set_representative_hours",
        "set_representative_hours_storage",
    ]
)


def _append_suffix(file_name: Path, suffix: str) -> Path:
    """Append ``suffix`` to ``file_name`` unless it is already present.

    Unlike ``Path.with_suffix``, this does not strip a trailing ``.<x>`` that
    is part of the stem (e.g. a scenario name like ``price_import_1.3``).

    :param file_name: The target path, with or without the final suffix
    :param suffix: The suffix to ensure, including the leading dot
    :return: ``file_name`` guaranteed to end in ``suffix``
    """
    if file_name.suffix == suffix:
        return file_name
    return file_name.with_name(file_name.name + suffix)


def drop_unit_levels(series: pd.Series) -> pd.Series:
    """Drops the unrelated index levels from the series.

    Args:
        series: The series to drop the levels from
    Returns:
        The series with the unrelated index levels dropped.
    """
    levels_to_drop = cast(
        list[Hashable],
        list(UNRELATED_INDEXES_FOR_UNITS & set(series.index.names)),
    )
    if levels_to_drop and len(levels_to_drop) < len(series.index.names):
        reduced_series = series.droplevel(levels_to_drop)
        reduced_series = reduced_series[~reduced_series.index.duplicated(keep="first")]
    elif levels_to_drop:
        # if there are no index levels left, we must only keep one entry
        reduced_series = series.reset_index(drop=True).loc[[0]]
    else:
        reduced_series = series
    return reduced_series


class Postprocess:
    """Class is defining the postprocessing of the results."""

    def __init__(
        self,
        model_schema: "ModelSchema",
        unit_converter: "UnitConverter",
        optimization_model: "OptimizationModel",
        scaling: "Scaling",
        time_steps: "TimeStepsDicts",
        optimized_time_steps: list[int],
        scenarios,
        model_name: str,
        subfolder: tuple[Path, Path] | Path,
        param_map,
    ):
        """Postprocessing of the results of the optimization.

        :param model: optimization model
        :param model_name: The name of the model used to name the output folder
        :param subfolder: The subfolder used for the results
        :param scenario_name: The name of the current scenario
        :param param_map: A dictionary mapping the parameters to the scenario names
        """
        logger.info("\n--- Postprocess results ---\n")
        # get the necessary stuff from the model
        self.model_schema = model_schema
        self.unit_converter = unit_converter
        self.optimization_model = optimization_model
        self.energy_system = model_schema.energy_system

        self.lp_model = optimization_model.lp_model

        self.optimized_time_steps = optimized_time_steps
        self.scenarios = scenarios
        self.param_map = param_map
        self.scaling = scaling
        self.time_steps = time_steps

        # get name or directory
        self.model_name: str = model_name
        self.name_dir: Path = Path(self.config.analysis.folder_output) / self.model_name

        # deal with the subfolder
        self.subfolder = subfolder
        # here we make use of the fact that None and "" both evaluate to
        # False but any non-empty string doesn't
        if subfolder != Path(""):
            # check if mf within scenario analysis
            if isinstance(self.subfolder, tuple):
                scenario_dir = self.name_dir.joinpath(self.subfolder[0])
                os.makedirs(scenario_dir, exist_ok=True)
                mf_in_scenario_dir = self.subfolder[0].joinpath(self.subfolder[1])
                self.name_dir = self.name_dir.joinpath(mf_in_scenario_dir)
            else:
                self.name_dir = self.name_dir.joinpath(self.subfolder)
        # create the output directory
        os.makedirs(self.name_dir, exist_ok=True)

        # check if we should overwrite output
        self.overwrite = self.config.analysis.overwrite_output
        # get the compression param
        self.output_format = self.config.analysis.output_format

    def save_results(self, scenario_name: str | None):
        """Saves the results of the optimization to a folder.

        :param scenario_name: name of scenario for which results are postprocessed
        """

        self.optimization_model.expressions.select(
            self.config.analysis.selected_saved_expressions
        )

        # save components
        component_map: dict[str, list[Hashable]] = {}
        component_map["sets"] = self.save_sets()
        component_map["parameter"] = self.save_param()
        component_map["variable"] = self.save_var()
        component_map["expression"] = self.save_expressions()
        component_map["dual"] = self.save_duals()
        component_map["reduced_cost"] = self.save_reduced_costs()
        self.save_component_map(component_map)

        self.save_system()
        self.save_analysis()
        self.save_scenarios()
        self.save_solver()
        self.save_unit_definitions()
        self.save_sequence_time_steps(scenario=scenario_name)
        self.save_param_map()
        if self.config.solver.run_diagnostics:
            self.save_benchmarking_data()

    @property
    def config(self):
        """Return the canonical configuration from the model schema."""
        return self.model_schema.config

    def save_benchmarking_data(self):
        """Saves the benchmarking data to a json file."""
        # initialize dictionary
        benchmarking_data = dict()
        # get the benchmarking data
        benchmarking_data["objective_value"] = self.lp_model.objective.value
        if self.config.solver.name == "gurobi":
            benchmarking_data["solving_time"] = self.lp_model.solver_model.Runtime
            if "Method" in self.config.solver.solver_options:
                if self.config.solver.solver_options["Method"] == 2:
                    benchmarking_data["number_iterations"] = (
                        self.lp_model.solver_model.BarIterCount
                    )
                else:
                    benchmarking_data["number_iterations"] = (
                        self.lp_model.solver_model.IterCount
                    )
            benchmarking_data["solver_status"] = self.lp_model.solver_model.Status
            benchmarking_data["number_constraints"] = (
                self.lp_model.solver_model.NumConstrs
            )
            benchmarking_data["number_variables"] = self.lp_model.solver_model.NumVars
        elif self.config.solver.name == "highs":
            benchmarking_data["solver_status"] = (
                self.lp_model.solver_model.getModelStatus().name
            )
            benchmarking_data["solving_time"] = self.lp_model.solver_model.getRunTime()
            benchmarking_data["number_iterations"] = (
                self.lp_model.solver_model.getInfo().simplex_iteration_count
            )
            benchmarking_data["number_constraints"] = (
                self.lp_model.solver_model.getNumRow()
            )
            benchmarking_data["number_variables"] = (
                self.lp_model.solver_model.getNumCol()
            )
        else:
            logger.info(
                f"Saving benchmarking data for solver {self.config.solver.name} has "
                "not been implemented yet"
            )

        benchmarking_data["scaling_time"] = self.scaling.scaling_time
        # get numerical range
        range_lhs, range_rhs = self.scaling.print_numerics(
            0, no_scaling=False, benchmarking_output=True
        )
        benchmarking_data["numerical_range_lhs"] = range_lhs
        benchmarking_data["numerical_range_rhs"] = range_rhs
        fname = self.name_dir.joinpath("benchmarking.json")
        self._write_json_file(fname, benchmarking_data)

    def save_sets(self) -> list[Hashable]:
        """Persist the stable version-4 set view reconstructed from relations."""

        series: dict[str, pd.Series] = {}
        set_registry = self.optimization_model.sets
        start_nodes = set_registry.relations[
            RelationEdgeStartNode.name
        ].combinations.to_frame(index=False)
        end_nodes = set_registry.relations[
            RelationEdgeEndNode.name
        ].combinations.to_frame(index=False)
        edge_nodes = start_nodes.merge(
            end_nodes,
            on="set_edges",
            suffixes=("_start", "_end"),
            validate="one_to_one",
        ).set_index("set_edges")
        edge_nodes = edge_nodes.reindex(set_registry.sets["set_edges"].members)
        serialized_edge_nodes = edge_nodes.apply(
            lambda row: f"{row['set_nodes_start']},{row['set_nodes_end']}",
            axis=1,
        )
        serialized_edge_nodes.name = "set_nodes_on_edges"
        serialized_edge_nodes.index.name = "set_nodes_on_edges"
        series["set_nodes_on_edges"] = serialized_edge_nodes

        for set_name, relation_class in LEGACY_INDEXED_SET_RELATIONS.items():
            if relation_class.name not in set_registry.relations:
                continue
            series[set_name] = self._serialize_legacy_indexed_set(
                set_name,
                set_registry.relations[relation_class.name],
            )

        representative_hours = list(set_registry[SetRepresentativeHours])
        series[SetRepresentativeHours.name] = pd.Series(
            representative_hours,
            name=SetRepresentativeHours.name,
            index=pd.RangeIndex(
                len(representative_hours),
                name=SetRepresentativeHours.name,
            ),
        )

        self._write_h5_file(self.name_dir / "sets.h5", series)
        docs = dict(self.optimization_model.sets.docs)
        docs["set_nodes_on_edges"] = "Start and end node of each directed edge."
        for set_name, relation_class in LEGACY_INDEXED_SET_RELATIONS.items():
            if relation_class.name in set_registry.relations:
                docs[set_name] = set_registry.relations[relation_class.name].doc
        self._write_json_file(
            self.name_dir / "sets_docs",
            docs,
        )

        return list(series.keys())

    @staticmethod
    def _serialize_legacy_indexed_set(
        set_name: str,
        relation: RelationSet,
    ) -> pd.Series:
        """Serialize a two-dimensional relation using the version-4 set shape."""
        grouped: dict[Hashable, list[object]] = {}
        for parent, child in relation.combinations:
            grouped.setdefault(parent, []).append(child)
        data = [",".join(str(value) for value in values) for values in grouped.values()]
        indices_list = list(grouped)
        indices: pd.Index
        if indices_list and isinstance(indices_list[0], tuple):
            tuple_indices = [
                cast(tuple[Hashable, ...], value) for value in indices_list
            ]
            indices = pd.MultiIndex.from_tuples(tuple_indices, names=[set_name])
        else:
            indices = pd.Index(indices_list, name=set_name)
        return pd.Series(data, name=set_name, index=indices)

    def save_param(self) -> list[Hashable]:
        """Saves the Param values to a json file which can then be
        post-processed immediately or loaded and postprocessed at some other
        time.
        """
        if not self.config.solver.save_parameters:
            logger.info("Parameters are not saved")
            return []

        parameters = xr.Dataset()
        legacy_names = {
            "weight_representative_hours": "time_steps_operation_duration",
            "duration_representative_hours_storage": "time_steps_storage_duration",
        }
        for param in self.optimization_model.parameters.docs.keys():
            output_name = legacy_names.get(param, param)
            selected_name = ENERGY_PARAMETER_COMPONENTS.get(output_name, output_name)
            if (
                self.config.solver.selected_saved_parameters
                and param not in self.config.solver.selected_saved_parameters
                and output_name not in self.config.solver.selected_saved_parameters
                and selected_name not in self.config.solver.selected_saved_parameters
            ):
                continue
            # get the values
            vals = getattr(self.optimization_model.parameters, param)
            # data frame
            if isinstance(vals, xr.DataArray):
                parameters[output_name] = vals
            # we have a scalar
            else:
                parameters[output_name] = xr.DataArray(data=[vals], dims=["scalar"])

        parameters = self._merge_capacity_type_components(
            parameters, ENERGY_PARAMETER_COMPONENTS
        )
        parameters = self._to_legacy_set_dimensions(parameters)
        parameters = self._to_legacy_time_dimensions(parameters)
        self._write_netcdf_file(self.name_dir / "parameters.nc", parameters)
        units: dict[str, pd.Series | None] = {
            legacy_names.get(name, name): pd.Series(value)
            for name, value in self.optimization_model.parameters.units.items()
            if value is not None
        }
        merged_units = self._merge_capacity_type_units(
            units, ENERGY_PARAMETER_COMPONENTS
        )
        merged_units = self._to_legacy_unit_dimensions(merged_units)
        self._write_units_to_file(self.name_dir / "parameters_units.h5", merged_units)
        self._write_json_file(
            self.name_dir / "parameters_docs.json",
            {
                legacy_names.get(
                    ENERGY_PARAMETER_COMPONENTS.get(name, name),
                    ENERGY_PARAMETER_COMPONENTS.get(name, name),
                ): self._legacy_component_doc(
                    legacy_names.get(
                        ENERGY_PARAMETER_COMPONENTS.get(name, name),
                        ENERGY_PARAMETER_COMPONENTS.get(name, name),
                    ),
                    doc,
                )
                for name, doc in self.optimization_model.parameters.docs.items()
                if name not in ENERGY_PARAMETER_COMPONENTS
            },
        )

        return list(parameters.keys())

    def save_var(self) -> list[Hashable]:
        """Saves the variable values to a json file which can then be
        post-processed immediately or loaded and postprocessed at some other
        time.
        """
        raw_solution = self.lp_model.solution.copy()
        requested = self.config.solver.selected_saved_variables
        if requested:
            solution_names = [cast(str, name) for name in raw_solution]
            available = set(solution_names).union(
                ENERGY_VARIABLE_COMPONENTS.get(name, name) for name in solution_names
            )
            unknown = sorted(set(requested).difference(available))
            if unknown:
                raise ValueError(
                    f"Unknown selected_saved_variables: {unknown}. "
                    f"Available solver variables: {sorted(available)}"
                )
            selected_names = [
                name
                for name in solution_names
                if name in requested
                or ENERGY_VARIABLE_COMPONENTS.get(name, name) in requested
            ]
            raw_solution = raw_solution[selected_names]
        variables = self.optimization_model.variables
        for name, default in variables.mask_default_values.items():
            if name not in raw_solution:
                continue
            structural = align_mask(
                variables.structural_masks[name], raw_solution[name]
            )
            performance = align_mask(
                variables.performance_masks[name], raw_solution[name]
            )
            restore = structural & ~performance
            raw_solution[name] = raw_solution[name].where(~restore, default)
        solution = self._merge_capacity_type_components(
            raw_solution, ENERGY_VARIABLE_COMPONENTS
        )
        solution = self._to_legacy_set_dimensions(solution)
        solution = self._to_legacy_time_dimensions(solution)
        self._write_netcdf_file(self.name_dir / "variables.nc", solution)

        units: dict[str, pd.Series | None] = {
            cast(str, name): self.optimization_model.variables.units[cast(str, name)]
            for name in raw_solution.keys()
        }
        merged_units = self._merge_capacity_type_units(
            units, ENERGY_VARIABLE_COMPONENTS
        )
        merged_units = self._to_legacy_unit_dimensions(merged_units)
        units_filtered = {
            name: series
            for name, series in merged_units.items()
            if series is not None and not series.empty
        }
        self._write_units_to_file(self.name_dir / "variables_units.h5", units_filtered)
        self._write_json_file(
            self.name_dir / "variables_docs.json",
            {
                ENERGY_VARIABLE_COMPONENTS.get(name, name): self._legacy_component_doc(
                    ENERGY_VARIABLE_COMPONENTS.get(name, name), doc
                )
                for name, doc in self.optimization_model.variables.docs.items()
                if name in raw_solution and name not in ENERGY_VARIABLE_COMPONENTS
            },
        )
        return list(solution.keys())

    def save_expressions(self) -> list[Hashable]:
        """Evaluate and persist configured expressions from the solved model."""
        registry = self.optimization_model.expressions
        selected = registry.select(self.config.analysis.selected_saved_expressions)
        values: dict[str, xr.DataArray] = {}
        units: dict[str, pd.Series | None] = {}
        for name in selected:
            value = self._evaluate_expression(name)
            values[name] = value

            expected = registry.indices[name]
            unit_category = registry.unit_categories[name]
            if not expected:
                units[name] = self._scalar_expression_unit(unit_category)
            else:
                coordinate_values = [value.coords[index].values for index in expected]
                index_values: list[object]
                if len(expected) == 1:
                    index_values = list(coordinate_values[0])
                else:
                    index_values = list(itertools.product(*coordinate_values))
                units[name] = self.optimization_model.variables._get_var_units(
                    unit_category,
                    index_values,
                    list(expected),
                    value.notnull(),
                )

        objective_name = self.config.analysis.objective
        if objective_name not in units:
            units[objective_name] = self._scalar_expression_unit(
                registry.unit_categories[objective_name]
            )

        dataset = xr.Dataset(values)
        dataset = self._merge_capacity_type_components(
            dataset, ENERGY_EXPRESSION_COMPONENTS
        )
        dataset = self._to_legacy_set_dimensions(dataset)
        dataset = self._to_legacy_time_dimensions(dataset)
        self._write_netcdf_file(self.name_dir / "expressions.nc", dataset)
        merged_units = self._merge_capacity_type_units(
            units, ENERGY_EXPRESSION_COMPONENTS
        )
        merged_units = self._to_legacy_unit_dimensions(merged_units)
        self._write_units_to_file(
            self.name_dir / "expressions_units.h5",
            {
                name: unit
                for name, unit in merged_units.items()
                if unit is not None and not unit.empty
            },
        )
        docs: dict[str, str] = {}
        for name in selected:
            public_name = ENERGY_EXPRESSION_COMPONENTS.get(name, name)
            docs.setdefault(
                public_name,
                self._legacy_component_doc(public_name, registry.docs[name]),
            )
        self._write_json_file(self.name_dir / "expressions_docs.json", docs)
        return list(dataset.keys())

    def _scalar_expression_unit(
        self, unit_category: dict[str, int]
    ) -> pd.Series | None:
        """Resolve the unit of an expression without public index dimensions."""
        if not unit_category:
            return None

        unit_converter = self.optimization_model.variables.unit_converter
        unit = unit_converter.ureg("dimensionless")
        base_dimensions = {
            "money": "[currency]",
            "distance": "[length]",
            "time": "[time]",
            "emissions": "[mass]",
        }
        for dimension, base_dimension in base_dimensions.items():
            if dimension not in unit_category:
                continue
            base_unit = next(
                unit_name
                for unit_name, dimensionality in unit_converter.base_units.items()
                if dimensionality == base_dimension
            )
            unit *= unit_converter.ureg(base_unit) ** unit_category[dimension]

        if "energy_quantity" in unit_category:
            energy_units = {
                str(quantity.units): quantity
                for quantity in unit_converter.carrier_energy_quantities.values()
            }
            if len(energy_units) != 1:
                raise ValueError(
                    "A scalar expression with an energy_quantity unit requires all "
                    "model carriers to use the same energy-quantity unit."
                )
            energy_unit = next(iter(energy_units.values()))
            unit *= energy_unit ** unit_category["energy_quantity"]

        return pd.Series([str(unit.units)])

    def _evaluate_expression(self, name: str) -> xr.DataArray:
        """Evaluate one expression and restore performance-masked defaults."""
        registry = self.optimization_model.expressions
        expression = registry[name]
        if isinstance(expression, LinearExpression):
            value = expression.solution
        elif isinstance(expression, Variable):
            value = (1 * expression).solution
        else:
            value = expression
        if not isinstance(value, xr.DataArray):
            value = xr.DataArray(value)
        expected = registry.indices[name]
        public_dimensions = tuple(dim for dim in value.dims if dim != "_term")
        if public_dimensions != expected:
            raise ValueError(
                f"Expression {name!r} produced dimensions {public_dimensions}, "
                f"but declares {expected}"
            )
        value = value.transpose(*expected)
        if name in registry.mask_default_values:
            structural = align_mask(registry.structural_masks[name], value)
            performance = align_mask(registry.performance_masks[name], value)
            restore = structural & ~performance
            value = value.where(~restore, registry.mask_default_values[name])
        return value

    def save_duals(self) -> list[Hashable]:
        """Saves the dual variable values to a h5 file."""
        if not self.config.solver.save_duals:
            logger.info("Duals are not saved")
            return []

        dual = self._merge_capacity_type_components(
            self.lp_model.dual, ENERGY_DUAL_COMPONENTS
        )
        dual = self._to_legacy_set_dimensions(dual)
        dual = self._to_legacy_time_dimensions(dual)
        self._write_netcdf_file(self.name_dir / "duals.nc", dual)
        self._write_json_file(
            self.name_dir / "duals_docs",
            {
                ENERGY_DUAL_COMPONENTS.get(name, name): self._legacy_component_doc(
                    ENERGY_DUAL_COMPONENTS.get(name, name), doc
                )
                for name, doc in self.optimization_model.constraints.docs.items()
                if name not in ENERGY_DUAL_COMPONENTS
            },
        )
        return list(dual.keys())

    @staticmethod
    def _merge_capacity_type_components(
        dataset: xr.Dataset, energy_components: dict[str, str]
    ) -> xr.Dataset:
        """Restore split power and energy components to the public result schema."""
        converted = dataset.copy()
        for energy_name, power_name in energy_components.items():
            has_power = power_name in converted
            has_energy = energy_name in converted
            if not has_power and not has_energy:
                continue

            power = converted[power_name] if has_power else None
            energy = converted[energy_name] if has_energy else None
            if energy is not None and energy.size == 0:
                energy = None
                if power is None:
                    converted = converted.drop_vars(energy_name)
                    continue
            if energy is not None:
                rename = {}
                if "set_storage_technologies" in energy.dims and (
                    power is None or "set_technologies" in power.dims
                ):
                    rename["set_storage_technologies"] = "set_technologies"
                if "set_nodes" in energy.dims and (
                    power is None or "set_location" in power.dims
                ):
                    rename["set_nodes"] = "set_location"
                energy = energy.rename(rename)

            reference = power if power is not None else energy
            assert reference is not None
            technology_dimension = next(
                (
                    dimension
                    for dimension in ("set_technologies", "set_storage_technologies")
                    if dimension in reference.dims
                ),
                reference.dims[0],
            )
            axis = reference.dims.index(technology_dimension) + 1
            components = []
            if power is not None:
                components.append(
                    power.expand_dims({"set_capacity_types": ["power"]}, axis=axis)
                )
            if energy is not None:
                components.append(
                    energy.expand_dims({"set_capacity_types": ["energy"]}, axis=axis)
                )
            combined = xr.concat(components, dim="set_capacity_types", join="outer")
            converted[power_name] = combined.transpose(*combined.dims)
            if has_energy:
                converted = converted.drop_vars(energy_name)
        return converted

    @staticmethod
    def _merge_capacity_type_units(
        units: dict[str, pd.Series | None], energy_components: dict[str, str]
    ) -> dict[str, pd.Series]:
        """Restore capacity-type indices in persisted unit metadata."""
        converted = dict(units)
        for energy_name, power_name in energy_components.items():
            has_power = power_name in converted
            has_energy = energy_name in converted
            if not has_power and not has_energy:
                continue
            power_unit = converted.get(power_name)
            energy_unit = converted.get(energy_name)
            power = (
                power_unit.rename_axis(
                    index={
                        "set_storage_technologies": "set_technologies",
                        "set_nodes": "set_location",
                    }
                )
                if power_unit is not None
                else None
            )
            energy = (
                energy_unit.rename_axis(
                    index={
                        "set_storage_technologies": "set_technologies",
                        "set_nodes": "set_location",
                    }
                )
                if energy_unit is not None
                else None
            )
            pieces = {
                capacity_type: unit
                for capacity_type, unit in (("power", power), ("energy", energy))
                if unit is not None
            }
            if not pieces:
                if has_energy:
                    del converted[energy_name]
                continue
            combined = pd.concat(
                pieces,
                names=["set_capacity_types"],
            )
            names = list(combined.index.names)
            if "set_technologies" in names:
                names.remove("set_capacity_types")
                position = names.index("set_technologies") + 1
                names.insert(position, "set_capacity_types")
                combined = combined.reorder_levels(names)
            converted[power_name] = combined
            if has_energy:
                del converted[energy_name]
        return {name: unit for name, unit in converted.items() if unit is not None}

    @staticmethod
    def _to_legacy_set_dimensions(dataset: xr.Dataset) -> xr.Dataset:
        """Rename canonical internal domains on a persistence-only copy."""
        converted = xr.Dataset()
        for name, array in dataset.data_vars.items():
            component_name = cast(str, name)
            rename = dict(LEGACY_COMPONENT_DIMENSIONS.get(component_name, {}))
            if "set_existing_capacity_ids" in array.dims:
                rename["set_existing_capacity_ids"] = "set_technologies_existing"
            converted[component_name] = array.rename(rename)
        return converted

    @staticmethod
    def _to_legacy_unit_dimensions(
        units: dict[str, pd.Series],
    ) -> dict[str, pd.Series]:
        """Apply persistence-only domain aliases to unit-series index names."""
        converted = {}
        for name, unit in units.items():
            rename = dict(LEGACY_COMPONENT_DIMENSIONS.get(name, {}))
            if "set_existing_capacity_ids" in unit.index.names:
                rename["set_existing_capacity_ids"] = "set_technologies_existing"
            converted[name] = unit.rename_axis(index=rename) if rename else unit
        return converted

    def _to_legacy_time_dimensions(self, dataset: xr.Dataset) -> xr.Dataset:
        """Flatten internal year/hour dimensions to the stable result schema."""
        converted = xr.Dataset()
        for name, array in dataset.data_vars.items():
            if {
                "set_years",
                "set_representative_hours",
            }.issubset(array.dims):
                converted[name] = self._flatten_time_dimension(
                    array,
                    representative_dimension="set_representative_hours",
                    legacy_dimension="set_time_steps_operation",
                    pair_to_step=self.time_steps.operation_pair_to_step,
                )
            elif {
                "set_years",
                "set_representative_hours_storage",
            }.issubset(array.dims):
                converted[name] = self._flatten_time_dimension(
                    array,
                    representative_dimension="set_representative_hours_storage",
                    legacy_dimension="set_time_steps_storage",
                    pair_to_step=self.time_steps.storage_pair_to_step,
                )
            else:
                converted[name] = array
        return converted

    def _flatten_time_dimension(
        self,
        array: xr.DataArray,
        *,
        representative_dimension: str,
        legacy_dimension: str,
        pair_to_step: dict[tuple[int, int], int] | None,
    ) -> xr.DataArray:
        """Flatten one explicit year/representative-hour pair."""
        assert pair_to_step is not None
        target_dims = list(array.dims)
        year_position = target_dims.index("set_years")
        target_dims.remove("set_years")
        target_dims.remove(representative_dimension)
        target_dims.insert(year_position, legacy_dimension)
        pieces = []
        for year in array.coords["set_years"].values:
            pairs = [
                pair
                for pair in pair_to_step
                if pair[0] == int(year)
                and pair[1] in array.coords[representative_dimension]
            ]
            if not pairs:
                continue
            hours = [pair[1] for pair in pairs]
            steps = [pair_to_step[pair] for pair in pairs]
            piece = array.sel(
                {"set_years": year, representative_dimension: hours}
            ).drop_vars("set_years", errors="ignore")
            piece = piece.assign_coords({representative_dimension: steps}).rename(
                {representative_dimension: legacy_dimension}
            )
            pieces.append(piece)
        if not pieces:
            return array
        return xr.concat(pieces, dim=legacy_dimension).transpose(*target_dims)

    @staticmethod
    def _legacy_time_doc(doc: str) -> str:
        """Translate internal time names in persisted component documentation."""
        return (
            doc.replace(
                "set_years, set_representative_hours_storage",
                "set_time_steps_storage",
            )
            .replace(
                "set_years, set_representative_hours",
                "set_time_steps_operation",
            )
            .replace("weight_representative_hours", "time_steps_operation_duration")
            .replace(
                "duration_representative_hours_storage",
                "time_steps_storage_duration",
            )
        )

    @classmethod
    def _legacy_component_doc(cls, component_name: str, doc: str) -> str:
        """Translate canonical domain names in persisted documentation."""
        converted = doc.replace(
            "set_existing_capacity_ids",
            "set_technologies_existing",
        )
        for canonical, legacy in LEGACY_COMPONENT_DIMENSIONS.get(
            component_name, {}
        ).items():
            converted = converted.replace(canonical, legacy)
        return cls._legacy_time_doc(converted)

    def save_reduced_costs(self) -> list[Hashable]:
        """Saves the reduced cost values of variables to a h5 file."""
        if self.config.solver.name != "gurobi":
            logger.info("Reduced costs are only supported for gurobi solver")
            return []

        if not self.config.solver.save_reduced_costs:
            logger.info("Reduced costs are not saved")
            return []

        reduced_costs = xr.Dataset()
        for name in self.lp_model.variables:
            selected_name = ENERGY_VARIABLE_COMPONENTS.get(name, name)
            # skip variables not selected to be saved
            if (
                self.config.solver.selected_saved_reduced_costs
                and name not in self.config.solver.selected_saved_reduced_costs
                and selected_name not in self.config.solver.selected_saved_reduced_costs
            ):
                continue

            # get reduced costs from solver
            if name in self.lp_model.variables:
                arr = self.lp_model.variables[name].get_solver_attribute("RC")
            else:
                logger.warning(f"Variable {name} not found in the model")
                continue

            # rescale
            if self.config.solver.use_scaling:
                arr = self.scaling.rescale_dataarray(arr, name)

            reduced_costs[name] = arr

        reduced_costs = self._merge_capacity_type_components(
            reduced_costs, ENERGY_VARIABLE_COMPONENTS
        )
        reduced_costs = self._to_legacy_set_dimensions(reduced_costs)
        reduced_costs = self._to_legacy_time_dimensions(reduced_costs)
        self._write_netcdf_file(self.name_dir / "reduced_costs.nc", reduced_costs)
        return list(reduced_costs.keys())

    def save_component_map(self, component_map: dict[str, list[Hashable]]):
        """Saves a list of components per type."""
        self._write_json_file(self.name_dir / "component_map.json", component_map)

    def save_system(self):
        """Saves the system dict as json."""
        if self.config.system.use_rolling_horizon:
            dirname = self.name_dir.parent
        else:
            dirname = self.name_dir
        self._write_json_file(dirname / "system.json", self.config.system.model_dump())

    def save_analysis(self):
        """Saves the analysis dict as json."""
        if self.config.system.use_rolling_horizon:
            dirname = self.name_dir.parent
        else:
            dirname = self.name_dir
        # remove cwd path part to avoid saving the absolute path
        if os.path.isabs(self.config.analysis.dataset):
            cwd = os.getcwd()
            self.config.analysis.dataset = os.path.relpath(
                self.config.analysis.dataset, cwd
            )
            self.config.analysis.folder_output = os.path.relpath(
                self.config.analysis.folder_output, cwd
            )
        self._write_json_file(
            dirname / "analysis.json", self.config.analysis.model_dump()
        )

    def save_solver(self):
        """Saves the solver dict as json."""
        # This we only need to save once
        if self.config.system.use_rolling_horizon:
            dirname = self.name_dir.parent
        else:
            dirname = self.name_dir

        # remove cwd path part to avoid saving the absolute path
        if os.path.isabs(self.config.solver.solver_dir):
            cwd = os.getcwd()
            self.config.solver.solver_dir = os.path.relpath(
                self.config.solver.solver_dir, cwd
            )
        # save
        self._write_json_file(dirname / "solver.json", self.config.solver.model_dump())

    def save_scenarios(self):
        """Saves the scenario dict as json."""
        # only save the scenarios at the highest level
        fname = (
            Path(self.config.analysis.folder_output)
            / self.model_name
            / "scenarios.json"
        )
        self._write_json_file(fname, self.scenarios)

    def save_unit_definitions(self):
        """Saves the user-defined units as txt."""
        if self.config.system.use_rolling_horizon:
            dirname = self.name_dir.parent
        else:
            dirname = self.name_dir

        # Only save user-defined units (skip base units like 'meter')
        all_units = self.unit_converter.ureg._units
        default_unit_items = self.unit_converter.default_unit_items
        user_units = list(set(all_units.items()).difference(default_unit_items))
        lines = list(set(unit.raw for _, unit in user_units if hasattr(unit, "raw")))  # type: ignore[attr-defined]

        self._write_txt_file(dirname / "unit_definitions.txt", "\n".join(lines))

    def save_param_map(self):
        """Saves the param_map dict as yaml."""
        if self.param_map is None:
            return

        # This we only need to save once
        if (
            self.config.system.use_rolling_horizon
            and self.config.system.conduct_scenario_analysis
        ):
            fname = self.name_dir.parent.parent.joinpath("param_map")
        elif self.subfolder != Path(""):
            fname = self.name_dir.parent.joinpath("param_map")
        else:
            fname = self.name_dir.joinpath("param_map")

        self._write_yml_file(fname.with_suffix(".yml"), self.param_map)

    def save_sequence_time_steps(self, scenario: str | None = None):
        """Saves the dict_all_sequence_time_steps dict as json.

        :param scenario: name of scenario for which results are postprocessed
        """
        # extract and save sequence time steps, we transform the arrays to lists
        dict_sequence_time_steps = self.flatten_dict(
            self.time_steps.get_sequence_time_steps_dict()
        )
        dict_sequence_time_steps["optimized_time_steps"] = self.optimized_time_steps
        dict_sequence_time_steps["time_steps_operation_duration"] = (
            self.time_steps.time_steps_operation_duration
        )
        dict_sequence_time_steps["time_steps_storage_duration"] = (
            self.time_steps.time_steps_storage_duration
        )
        dict_sequence_time_steps["time_steps_storage_level_startend_year"] = (
            self.time_steps.time_steps_storage_level_startend_year
        )
        dict_sequence_time_steps["time_steps_year2operation"] = (
            self.get_time_steps_year2operation()
        )
        dict_sequence_time_steps["time_steps_year2storage"] = (
            self.get_time_steps_year2storage()
        )

        # add the scenario name
        if scenario is not None:
            add_on = f"_{scenario}"
        else:
            add_on = ""

            # This we only need to save once
        if self.config.system.use_rolling_horizon:
            fname = self.name_dir.parent.joinpath(
                f"dict_all_sequence_time_steps{add_on}"
            )
        else:
            fname = self.name_dir.joinpath(f"dict_all_sequence_time_steps{add_on}")
        dict_formatted = {}
        for k, v in dict_sequence_time_steps.items():
            if isinstance(v, np.ndarray):
                dict_formatted[k] = v.tolist()
            elif isinstance(v, dict):
                dict_formatted[k] = {
                    str(kk): vv.tolist() if isinstance(vv, np.ndarray) else str(vv)
                    for kk, vv in v.items()
                }
            elif isinstance(v, list):
                dict_formatted[k] = v
            else:
                NotImplementedError(f"Type {type(v)} not supported for key {k}")

        self._write_json_file(fname, dict_formatted)

    def flatten_dict(self, dictionary):
        """Creates a copy of the dictionary where all numpy arrays are
        recursively flattened to lists such that it can be saved as json file.

        :param dictionary: The input dictionary
        :return: A copy of the dictionary containing lists instead of arrays
        """
        # create a copy of the dict to avoid overwrite
        out_dict = dict()

        # falten all arrays
        for k, v in dictionary.items():
            # transform the key None to 'null'
            if k is None:
                k = "null"

            # recursive call
            if isinstance(v, dict):
                out_dict[k] = self.flatten_dict(v)  # flatten the array to list
            elif isinstance(v, pd.Series):
                # Note: list(v) creates a list of np objects v.tolist() not
                out_dict[k] = v.values.tolist()
            # take as is
            else:
                out_dict[k] = v

        return out_dict

    def get_time_steps_year2operation(self):
        """Returns a HDF5-Serializable version of the
        dict_time_steps_year2operation dictionary.
        """
        assert self.time_steps.time_steps_year2operation is not None
        return {
            str(year): time_steps
            for year, time_steps in self.time_steps.time_steps_year2operation.items()
        }

    def get_time_steps_year2storage(self):
        """Returns a HDF5-Serializable version of the
        dict_time_steps_year2storage dictionary.
        """
        assert self.time_steps.time_steps_year2storage is not None
        return {
            str(year): time_steps
            for year, time_steps in self.time_steps.time_steps_year2storage.items()
        }

    def _write_units_to_file(self, f_name: Path, units_dict: dict[str, pd.Series]):
        """Writes the units dictionary to a h5 file.

        Args:
            name: Filename without extension
            units_dict: The dictionary to save
        """
        units_dict_reduced = {}
        for key, series in units_dict.items():
            reduced_series = drop_unit_levels(series)
            units_dict_reduced[key] = reduced_series
        self._write_h5_file(f_name, units_dict_reduced)

    def _write_json_file(self, file_name: Path, dict: dict[str, Any]):
        """Writes the dictionary to a json file.

        :param file_name: The name of the file
        :param dict: The dictionary to save
        """
        file_name = _append_suffix(file_name, ".json")
        if file_name.exists() and not self.overwrite:
            return

        with FileLock(file_name.with_suffix(".json.lock")).acquire(timeout=300):
            with open(file_name, "w+") as file:
                json.dump(dict, file, indent=2)

    def _write_yml_file(self, file_name: Path, dict: dict[str, Any]):
        """Writes the dictionary to a yml file.

        :param file_name: The name of the file
        :param dict: The dictionary to save
        """
        file_name = _append_suffix(file_name, ".yml")
        if file_name.exists() and not self.overwrite:
            return

        with FileLock(file_name.with_suffix(".yml.lock")).acquire(timeout=300):
            with open(file_name, "w") as file:
                yaml.dump(dict, file)

    def _write_txt_file(self, file_name: Path, txt: str):
        """Writes the text to a txt file.

        :param file_name: The name of the file
        :param txt: The text to save
        """
        file_name = _append_suffix(file_name, ".txt")
        if file_name.exists() and not self.overwrite:
            return

        with FileLock(file_name.with_suffix(".txt.lock")).acquire(timeout=300):
            with open(file_name, "w+", encoding="utf-8") as outfile:
                outfile.write(txt)

    def _write_h5_file(
        self,
        file_name: Path,
        named_series: dict[str, pd.Series],
        mode: Literal["a", "w", "r", "r+"] = "w",
    ):
        """Writes the dictionary to a hdf5 file.

        :param file_name: The name of the file
        :param named_series: The dictionary to save
        :param mode: Writting mode for python file. The two options are 'w' and
            'a'. The former create a new file while the latter will append to an
            existing file.
        """
        file_name = _append_suffix(file_name, ".h5")
        if file_name.exists() and mode == "w" and not self.overwrite:
            raise FileExistsError(
                "File already exists. Please set overwrite=True to overwrite the file."
            )

        with FileLock(file_name.with_suffix(".h5.lock")).acquire(timeout=300):
            with pd.HDFStore(
                file_name, mode=mode, complevel=H5_COMP_LEVEL, complib=H5_COMP_LIB
            ) as store:
                for key, series in named_series.items():
                    if not isinstance(series, pd.Series):
                        raise TypeError(
                            (
                                f"Expected a pandas Series for key '{key}', "
                                f"but got {type(series)}"
                            )
                        )
                    # PyTables expects a NumPy dtype and cannot serialize the
                    # extension StringDtype inferred by pandas 3.0.
                    if isinstance(series.dtype, pd.StringDtype):
                        series = series.astype(object)
                    store.put(key, series)

    def _write_netcdf_file(self, file_name: Path, dataset: xr.Dataset):
        """Writes the dataset to a netcdf file.

        :param file_name: The name of the file
        :param dataset: The dataset to save
        """
        file_name = _append_suffix(file_name, ".nc")
        if file_name.exists() and not self.overwrite:
            return

        with FileLock(file_name.with_suffix(".nc.lock")).acquire(timeout=300):
            encoding = {
                var: {"zlib": True, "complevel": H5_COMP_LEVEL}
                for var in dataset.data_vars
            }
            dataset.to_netcdf(file_name, encoding=encoding)
