import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from linopy import Model
from pint import UnitRegistry

from zen_garden.config import Analysis, Solver, System
from zen_garden.elements.carrier.expressions import CARRIER_EXPRESSIONS
from zen_garden.elements.carrier.variables import CARRIER_VARIABLES
from zen_garden.elements.energy_system.expressions import ENERGY_SYSTEM_EXPRESSIONS
from zen_garden.elements.energy_system.variables import ENERGY_SYSTEM_VARIABLES
from zen_garden.elements.storage_technology.expressions import (
    STORAGE_TECHNOLOGY_EXPRESSIONS,
)
from zen_garden.elements.storage_technology.expressions.capacity_previous_energy import (  # noqa: E501
    CapacityPreviousEnergy,
)
from zen_garden.elements.storage_technology.masks import (
    EnergyCapacityAvailableMask,
)
from zen_garden.elements.storage_technology.variables import (
    STORAGE_TECHNOLOGY_VARIABLES,
)
from zen_garden.elements.storage_technology.variables.capacity_addition_energy import (  # noqa: E501
    CapacityAdditionEnergy,
)
from zen_garden.elements.storage_technology.variables.capacity_energy import (
    CapacityEnergy,
)
from zen_garden.elements.technology.expressions import TECHNOLOGY_EXPRESSIONS
from zen_garden.elements.technology.expressions.capacity_previous import (
    CapacityPrevious,
)
from zen_garden.elements.technology.masks import CapacityAvailableMask
from zen_garden.elements.technology.variables import TECHNOLOGY_VARIABLES
from zen_garden.elements.technology.variables.capacity import Capacity
from zen_garden.elements.technology.variables.capacity_addition import CapacityAddition
from zen_garden.elements.transport_technology.expressions import (
    TRANSPORT_TECHNOLOGY_EXPRESSIONS,
)
from zen_garden.elements.transport_technology.variables import (
    TRANSPORT_TECHNOLOGY_VARIABLES,
)
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.construction_service import ModelConstructionService
from zen_garden.model.registries.expression import ExpressionRegistry
from zen_garden.postprocess.postprocess import Postprocess
from zen_garden.postprocess.results.component_map import ComponentMap
from zen_garden.postprocess.results.scenario import Scenario
from zen_garden.postprocess.results.timestep_map import TimestepMap

MIGRATED_EXPRESSIONS = [
    "cost_carrier",
    "cost_shed_demand",
    "cost_carrier_total",
    "carbon_emissions_carrier",
    "carbon_emissions_carrier_total",
    "cost_opex_variable",
    "cost_opex_yearly",
    "cost_opex_yearly_total",
    "cost_capex_overnight",
    "cost_capex_yearly",
    "cost_capex_yearly_total",
    "carbon_emissions_technology",
    "carbon_emissions_technology_total",
    "cost_capex_overnight_energy",
    "cost_capex_yearly_energy",
    "carbon_emissions_annual",
    "cost_carbon_emissions_total",
    "cost_total",
    "net_present_cost",
    "flow_transport_loss",
    "carbon_emissions_cumulative",
    "capacity_previous",
    "capacity_previous_energy",
]


class _DimensionlessExpression(GenericExpression):
    name = "dimensionless_expression"
    indices = ()
    unit_category = {}
    doc = "The dimensionless constant one."

    @classmethod
    def get_expression(cls, model_constructor):
        return xr.DataArray(1.0)


class _IndexedExpression(GenericExpression):
    name = "indexed_expression"
    indices = ("set_test",)
    unit_category = {"money": 1}
    doc = "A test expression indexed by the test set."

    @classmethod
    def get_expression(cls, model_constructor):
        return xr.DataArray([1.0], dims="set_test")


def test_expression_contract_and_selection():
    with pytest.raises(TypeError, match="unit_category"):

        class _MissingUnits(GenericExpression):
            """Expression intentionally missing its unit declaration."""

            name = "missing_units"
            indices = ()

            @classmethod
            def get_expression(cls, model_constructor):
                return xr.DataArray(1.0)

    with pytest.raises(TypeError, match="must not define 'symbol'"):

        class _ExpressionWithSymbol(GenericExpression):
            """Expression intentionally declaring a forbidden symbol."""

            name = "expression_with_symbol"
            indices = ()
            unit_category = {}
            symbol = "X"

            @classmethod
            def get_expression(cls, model_constructor):
                return xr.DataArray(1.0)

    assert _DimensionlessExpression.unit_category == {}
    assert Analysis().selected_saved_expressions == MIGRATED_EXPRESSIONS
    expression_specs = {
        expression.name
        for expression in (
            CARRIER_EXPRESSIONS
            + ENERGY_SYSTEM_EXPRESSIONS
            + STORAGE_TECHNOLOGY_EXPRESSIONS
            + TECHNOLOGY_EXPRESSIONS
            + TRANSPORT_TECHNOLOGY_EXPRESSIONS
        )
    }
    variable_specs = {
        variable.name
        for variable in (
            CARRIER_VARIABLES
            + ENERGY_SYSTEM_VARIABLES
            + STORAGE_TECHNOLOGY_VARIABLES
            + TECHNOLOGY_VARIABLES
            + TRANSPORT_TECHNOLOGY_VARIABLES
        )
    }
    assert set(MIGRATED_EXPRESSIONS).issubset(expression_specs)
    assert set(MIGRATED_EXPRESSIONS).isdisjoint(variable_specs)

    registry = ExpressionRegistry()
    registry.add(_DimensionlessExpression, xr.DataArray(1.0))
    registry.add(_IndexedExpression, xr.DataArray([1.0], dims="set_test"))
    assert registry.select([]) == [
        "dimensionless_expression",
        "indexed_expression",
    ]
    assert registry.select(["indexed_expression"]) == ["indexed_expression"]
    with pytest.raises(ValueError, match="unknown_one.*unknown_two"):
        registry.select(["unknown_two", "unknown_one"])


def test_expression_result_evaluation_and_masks():
    coordinates = {"set_test": ["active", "inactive", "invalid"]}
    structural = xr.DataArray([True, True, False], coords=coordinates, dims="set_test")
    performance = xr.DataArray(
        [True, False, False], coords=coordinates, dims="set_test"
    )
    lp_model = Model()
    variable = lp_model.add_variables(
        lower=xr.DataArray([4.0, 0.0, 0.0], coords=coordinates, dims="set_test"),
        upper=xr.DataArray([4.0, 0.0, 0.0], coords=coordinates, dims="set_test"),
        coords=[pd.Index(coordinates["set_test"], name="set_test")],
        name="masked_variable",
    )
    lp_model.add_objective(variable.sum())
    lp_model.solve(solver_name="highs")
    registry = ExpressionRegistry()
    registry.add(
        _IndexedExpression,
        (1 * variable).where(performance),
        structural_mask=structural,
        performance_mask=performance,
        mask_default_value=0.0,
    )
    registry.add(_DimensionlessExpression, xr.DataArray(2.0))
    postprocess = Postprocess.__new__(Postprocess)
    postprocess.optimization_model = SimpleNamespace(expressions=registry)

    evaluated = postprocess._evaluate_expression("indexed_expression")
    assert evaluated.dims == ("set_test",)
    assert evaluated.coords["set_test"].values.tolist() == [
        "active",
        "inactive",
        "invalid",
    ]
    assert evaluated.sel(set_test="active").item() == 4.0
    assert evaluated.sel(set_test="inactive").item() == 0.0
    assert np.isnan(evaluated.sel(set_test="invalid").item())
    assert postprocess._evaluate_expression("dimensionless_expression").item() == 2.0


def test_expression_result_round_trip():
    result_path = Path("tests/unit_tests/fixtures/expression_round_trip")
    generated_files = [
        path for path in result_path.iterdir() if path.name != ".gitkeep"
    ]
    assert not generated_files
    try:
        _assert_expression_result_round_trip(result_path)
    finally:
        for path in result_path.iterdir():
            if path.name != ".gitkeep":
                path.unlink()


def _assert_expression_result_round_trip(tmp_path):
    xr.Dataset(
        {
            "solver_variable": xr.DataArray(
                [3.0], coords={"set_years": [0]}, dims="set_years"
            )
        }
    ).to_netcdf(tmp_path / "variables.nc")
    xr.Dataset(
        {
            "saved_expression": xr.DataArray(
                [7.0], coords={"set_years": [0]}, dims="set_years"
            )
        }
    ).to_netcdf(tmp_path / "expressions.nc")
    pd.Series(["MW"], index=pd.Index([0], name="set_years")).to_hdf(
        tmp_path / "variables_units.h5", key="solver_variable"
    )
    pd.Series(["MEUR"], index=pd.Index([0], name="set_years")).to_hdf(
        tmp_path / "expressions_units.h5", key="saved_expression"
    )
    (tmp_path / "variables_docs.json").write_text(
        json.dumps({"solver_variable": "A solver variable."}), encoding="utf-8"
    )
    (tmp_path / "expressions_docs.json").write_text(
        json.dumps({"saved_expression": "An expanded mathematical expression."}),
        encoding="utf-8",
    )

    component_map = ComponentMap(
        sets=[],
        variable=["solver_variable"],
        parameter=[],
        dual=[],
        reduced_cost=[],
        expression=["saved_expression"],
    )
    scenario = Scenario(tmp_path, "none", "")
    scenario._analysis = Analysis(objective="saved_expression")
    scenario._system = System()
    scenario._solver = Solver()
    scenario._component_map = component_map
    scenario._ureg = UnitRegistry()
    scenario._time_steps = TimestepMap(
        operation=[],
        storage=[],
        yearly=[0],
        optimized_time_steps=[0],
        time_steps_operation_duration={},
        time_steps_storage_duration={},
        time_steps_storage_level_startend_year={},
        time_steps_year2operation={},
        time_steps_year2storage={},
    )

    assert component_map.find_type("solver_variable").value == "variable"
    assert component_map.find_type("saved_expression").value == "expression"
    assert scenario.get_raw_values("saved_expression").iloc[0] == 7.0
    expression_unit = scenario.get_unit("saved_expression", False)
    assert isinstance(expression_unit, pd.Series)
    assert expression_unit.iloc[0] == "MEUR"
    objective_unit = scenario.get_unit("objective", False)
    assert isinstance(objective_unit, pd.Series)
    assert objective_unit.equals(expression_unit)
    assert scenario.get_doc("saved_expression") == (
        "An expanded mathematical expression."
    )
    assert scenario.get_index_names("saved_expression") == ["set_years"]
    assert scenario.get_total("saved_expression").iloc[0] == 7.0
    assert scenario.get_full_ts("saved_expression").iloc[0] == 7.0

    old_map = ComponentMap(
        **{
            "sets": [],
            "variable": ["solver_variable"],
            "parameter": [],
            "dual": [],
            "reduced_cost": [],
        }
    )
    assert old_map.expression == []


def test_previous_capacity_expression():
    coordinates = {
        "set_technologies": ["existing", "new", "inactive"],
        "set_location": ["node"],
        "set_years": [0],
    }
    capacity = xr.DataArray([[[5.0]], [[4.0]], [[np.nan]]], coords=coordinates)
    additions = xr.DataArray([[[0.0]], [[1.5]], [[np.nan]]], coords=coordinates)
    model = SimpleNamespace(variables={Capacity: capacity, CapacityAddition: additions})
    constructor = SimpleNamespace(optimization_model=model)
    previous = CapacityPrevious.get_expression(constructor)
    assert previous.sel(set_technologies="existing").item() == 5.0
    assert previous.sel(set_technologies="new").item() == 2.5
    assert np.isnan(previous.sel(set_technologies="inactive").item())
    assert CapacityPrevious.masks == (CapacityAvailableMask,)

    energy_coordinates = {
        "set_storage_technologies": ["battery"],
        "set_nodes": ["node"],
        "set_years": [0],
    }
    energy_model = SimpleNamespace(
        variables={
            CapacityEnergy: xr.DataArray([[[8.0]]], coords=energy_coordinates),
            CapacityAdditionEnergy: xr.DataArray([[[3.0]]], coords=energy_coordinates),
        }
    )
    energy_previous = CapacityPreviousEnergy.get_expression(
        SimpleNamespace(optimization_model=energy_model)
    )
    assert energy_previous.item() == 5.0
    assert CapacityPreviousEnergy.masks == (EnergyCapacityAvailableMask,)


@pytest.mark.parametrize("sense", ["min", "max"])
def test_objective_requires_scalar_expression(sense):
    lp_model = Model()
    variable = lp_model.add_variables(name="objective_variable")
    registry = ExpressionRegistry()
    registry.add(_DimensionlessExpression, variable + 2.0)
    optimization_model = SimpleNamespace(
        expressions=registry,
        lp_model=lp_model,
        objective_constant=0.0,
    )
    service = ModelConstructionService.__new__(ModelConstructionService)
    service.optimization_model = optimization_model
    service.model_schema = SimpleNamespace(
        config=SimpleNamespace(
            analysis=SimpleNamespace(objective="dimensionless_expression", sense=sense),
            solver=SimpleNamespace(run_diagnostics=False),
        )
    )
    service._construct_objective()
    assert lp_model.objective.sense == sense
    assert optimization_model.objective_constant == 2.0

    service.model_schema.config.analysis.objective = "unknown_expression"
    with pytest.raises(KeyError, match="Available expressions"):
        service._construct_objective()

    dimensional_model = Model()
    dimensional_variable = dimensional_model.add_variables(
        coords=[["a", "b"]], name="dimensional_variable"
    )
    dimensional_registry = ExpressionRegistry()
    dimensional_registry.add(_IndexedExpression, dimensional_variable)
    service.optimization_model = SimpleNamespace(
        expressions=dimensional_registry,
        lp_model=dimensional_model,
        objective_constant=0.0,
    )
    service.model_schema.config.analysis.objective = "indexed_expression"
    with pytest.raises(ValueError, match="explicit sum or selection"):
        service._construct_objective()

    mismatch_registry = ExpressionRegistry()
    mismatch_registry.add(_DimensionlessExpression, dimensional_variable)
    service.optimization_model = SimpleNamespace(
        expressions=mismatch_registry,
        lp_model=dimensional_model,
        objective_constant=0.0,
    )
    service.model_schema.config.analysis.objective = "dimensionless_expression"
    with pytest.raises(ValueError, match=r"despite declaring indices=\(\)"):
        service._construct_objective()

    unit_registry = UnitRegistry()
    postprocess = Postprocess.__new__(Postprocess)
    postprocess.optimization_model = SimpleNamespace(
        variables=SimpleNamespace(
            unit_converter=SimpleNamespace(
                ureg=unit_registry,
                base_units={},
                carrier_energy_quantities={"heat": unit_registry("gigawatt_hour")},
            )
        )
    )
    objective_unit = postprocess._scalar_expression_unit({"energy_quantity": 1})
    assert objective_unit is not None
    assert objective_unit.iloc[0] == "gigawatt_hour"
