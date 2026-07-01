"""Defines the parameters, variables and constraints that hold for all technologies.
The class takes the abstract optimization model as an input, and returns the parameters,
variables and constraints that hold for all technologies.
"""

import itertools
import logging
from typing import TYPE_CHECKING, cast, override

import linopy as lp
import numpy as np
import pandas as pd
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.model.components.index_set import IndexSet
from zen_garden.model.components.zen_index import ZenIndex
from zen_garden.model.components.zen_set import ZenSet
from zen_garden.model.config import Config
from zen_garden.model.context import Context
from zen_garden.model.element import Element, ElementConstructor
from zen_garden.model.generic_rule import GenericRule
from zen_garden.model.zen_model import ZenModel

if TYPE_CHECKING:
    from zen_garden.model.energy_system import EnergySystem
    from zen_garden.preprocess.unit_handling import UnitHandling
    from zen_garden.services.element_registry import ElementRegistry

logger = logging.getLogger(__name__)


class Technology(Element):
    """Defines parameters, variables and constraints holding for all technologies."""

    # set label
    label = "set_technologies"
    location_type = None

    def __init__(
        self,
        technology_name: str,
        config: Config,
        context: Context,
        energy_system: "EnergySystem",
        element_registry: "ElementRegistry",
        unit_handling: "UnitHandling",
    ):
        """Init generic technology object.

        :param technology: technology that is added to the model
        :param optimization_setup: The OptimizationSetup the element is part of
        """
        super().__init__(
            technology_name,
            config,
            context,
            energy_system,
            element_registry,
            unit_handling,
        )

    def store_carriers(self):
        """Retrieves and stores information on reference."""
        self.reference_carrier = self.data_input.extract_carriers(
            carrier_type="reference_carrier"
        )
        self.energy_system.set_technology_of_carrier(self.name, self.reference_carrier)

    def store_input_data(self):
        """Retrieves and stores input data for element as attributes.

        Each Child class overwrites method to store different attributes.
        """
        # store scenario dict
        super().store_scenario_dict()
        # set attributes of technology
        set_location = self.location_type
        self.capacity_addition_min = self.data_input.extract_input_data(
            "capacity_addition_min",
            index_sets=[],
            unit_category={"energy_quantity": 1, "time": -1},
        )
        self.capacity_addition_max = self.data_input.extract_input_data(
            "capacity_addition_max",
            index_sets=[],
            unit_category={"energy_quantity": 1, "time": -1},
        )
        self.capacity_addition_unbounded = self.data_input.extract_input_data(
            "capacity_addition_unbounded",
            index_sets=[],
            unit_category={"energy_quantity": 1, "time": -1},
        )
        self.lifetime = self.data_input.extract_input_data(
            "lifetime", index_sets=[], unit_category={}
        )
        if "depreciation_time" in self.data_input.attribute_dict:
            self.depreciation_time = self.data_input.extract_input_data(
                "depreciation_time", index_sets=[], unit_category={}
            )
            self.depreciation_time[0] = np.max(
                (
                    self.config.system.interval_between_years,
                    self.depreciation_time[0],
                )
            )
        else:
            self.depreciation_time = self.lifetime.copy()
        self.construction_time = self.data_input.extract_input_data(
            "construction_time", index_sets=[], unit_category={}
        )
        # maximum diffusion rate
        self.max_diffusion_rate = self.data_input.extract_input_data(
            "max_diffusion_rate",
            index_sets=["set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={},
        )

        # add all raw time series to dict
        self.raw_time_series = {}
        self.raw_time_series["min_load"] = self.data_input.extract_input_data(
            "min_load",
            index_sets=[set_location, "set_time_steps"],
            time_steps="set_base_time_steps_yearly",
            unit_category={},
        )
        self.raw_time_series["max_load"] = self.data_input.extract_input_data(
            "max_load",
            index_sets=[set_location, "set_time_steps"],
            time_steps="set_base_time_steps_yearly",
            unit_category={},
        )
        self.raw_time_series["opex_specific_variable"] = (
            self.data_input.extract_input_data(
                "opex_specific_variable",
                index_sets=[set_location, "set_time_steps"],
                time_steps="set_base_time_steps_yearly",
                unit_category={"money": 1, "energy_quantity": -1},
            )
        )
        # non-time series input data
        self.capacity_limit = self.data_input.extract_input_data(
            "capacity_limit",
            index_sets=[set_location, "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"energy_quantity": 1, "time": -1},
        )

        # lower capacity limit
        self.capacity_lower_limit = self.data_input.extract_input_data(
            "capacity_lower_limit",
            index_sets=[set_location, "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"energy_quantity": 1, "time": -1},
        )

        self.carbon_intensity_technology = self.data_input.extract_input_data(
            "carbon_intensity_technology",
            index_sets=[set_location],
            unit_category={"emissions": 1, "energy_quantity": -1},
        )
        # extract existing capacity
        self.set_technologies_existing = (
            self.data_input.extract_set_technologies_existing()
        )
        self.capacity_existing = self.data_input.extract_input_data(
            "capacity_existing",
            index_sets=[set_location, "set_technologies_existing"],
            unit_category={"energy_quantity": 1, "time": -1},
        )
        self.capacity_investment_existing = self.data_input.extract_input_data(
            "capacity_investment_existing",
            index_sets=[set_location, "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        self.lifetime_existing = self.data_input.extract_lifetime_existing(
            "capacity_existing", index_sets=[set_location, "set_technologies_existing"]
        )

    def calculate_capex_of_capacities_existing(self, storage_energy=False):
        """This method calculates the annualized capex of the existing capacities.

        :param storage_energy: boolean if energy storage
        :return: capex of existing capacities
        """
        if self.__class__.__name__ == "StorageTechnology":
            if storage_energy:
                capacities_existing = self.capacity_existing_energy
            else:
                capacities_existing = self.capacity_existing
            capex_capacity_existing = capacities_existing.to_frame().apply(
                lambda _capacity_existing: self.calculate_capex_of_single_capacity(
                    _capacity_existing.squeeze(),
                    _capacity_existing.name,
                    storage_energy=storage_energy,
                ),
                axis=1,
            )
        else:
            capacities_existing = self.capacity_existing
            capex_capacity_existing = capacities_existing.to_frame().apply(
                lambda _capacity_existing: self.calculate_capex_of_single_capacity(
                    _capacity_existing.squeeze(), _capacity_existing.name
                ),
                axis=1,
            )
        return capex_capacity_existing

    def calculate_capex_of_single_capacity(self, capacity, index, **kwargs):
        """Calculates annualized capex of existing capacity, implemented in child class.

        :param args: arguments
        """
        raise NotImplementedError

    def calculate_fraction_of_year(self):
        """Calculate fraction of year."""
        # only account for fraction of year
        fraction_year = (
            self.config.system.unaggregated_time_steps_per_year
            / self.config.system.total_hours_per_year
        )
        return fraction_year

    def add_new_capacity_addition_tech(
        self, capacity_addition: pd.Series, capex: pd.Series, step_horizon: list
    ):
        """Adds the newly built capacity to the existing capacity.

        :param capacity_addition: pd.Series of newly built capacity of technology
        :param capex: pd.Series of capex of newly built capacity of technology
        :param step_horizon: current horizon step
        """
        system = self.config.system
        # reduce lifetime of existing capacities and add new remaining lifetime
        delta_lifetime = step_horizon[-1] - step_horizon[0]
        self.lifetime_existing = (
            self.lifetime_existing
            - system.interval_between_years * (delta_lifetime + 1)
        ).clip(lower=0)
        # new capacity
        new_capacity_addition = capacity_addition[step_horizon]
        new_capex = capex[step_horizon]
        # if at least one value unequal to zero
        if not (new_capacity_addition.stack() == 0).all():
            # add new index to set_technologies_existing
            index_step_horizon = list(range(len(step_horizon)))
            index_new_technology = [
                max(self.set_technologies_existing) + 1 + idx
                for idx in index_step_horizon
            ]
            self.set_technologies_existing = np.append(
                self.set_technologies_existing, index_new_technology
            )
            # add new remaining lifetime
            lifetime = self.lifetime_existing.unstack()
            lifetime[index_new_technology] = [
                self.lifetime[0]
                - system.interval_between_years * (delta_lifetime - idx + 1)
                for idx in index_step_horizon
            ]
            self.lifetime_existing = lifetime.stack()

            for type_capacity in list(
                set(new_capacity_addition.index.get_level_values(0))
            ):
                # if power
                if type_capacity == system.set_capacity_types[0]:
                    energy_string = ""
                # if energy
                else:
                    energy_string = "_energy"
                capacity_existing = getattr(self, "capacity_existing" + energy_string)
                capex_capacity_existing = getattr(
                    self, "capex_capacity_existing" + energy_string
                )
                # add new existing capacity
                capacity_existing = capacity_existing.unstack()
                capacity_existing[index_new_technology] = new_capacity_addition.loc[
                    type_capacity
                ]
                setattr(
                    self, "capacity_existing" + energy_string, capacity_existing.stack()
                )
                # calculate capex of existing capacity
                capex_capacity_existing = capex_capacity_existing.unstack()
                capex_capacity_existing[index_new_technology] = new_capex.loc[
                    type_capacity
                ]
                setattr(
                    self,
                    "capex_capacity_existing" + energy_string,
                    capex_capacity_existing.stack(),
                )

    def add_new_capacity_investment(
        self, capacity_investment: pd.Series, step_horizon: list
    ):
        """Adds the newly invested capacity to the list of invested capacity.

        :param capacity_investment: pd.Series of newly built capacity of technology
        :param step_horizon: optimization time step
        """
        system = self.config.system
        new_capacity_investment = capacity_investment[step_horizon]
        new_capacity_investment = new_capacity_investment.fillna(0)
        if not (new_capacity_investment.stack() == 0).all():
            for type_capacity in list(
                set(new_capacity_investment.index.get_level_values(0))
            ):
                # if power
                if type_capacity == system.set_capacity_types[0]:
                    energy_string = ""
                # if energy
                else:
                    energy_string = "_energy"
                capacity_investment_existing = getattr(
                    self, "capacity_investment_existing" + energy_string
                )
                # add new existing invested capacity
                capacity_investment_existing = capacity_investment_existing.unstack()
                capacity_investment_existing[step_horizon] = (
                    new_capacity_investment.loc[type_capacity]
                )
                setattr(
                    self,
                    "capacity_investment_existing" + energy_string,
                    capacity_investment_existing.stack(),
                )

    @classmethod
    def get_investment_time_step(cls, params, system, tech, year):
        """Returns investment time step of technology, considering construction time.

        returns investment time step of technology, i.e., the time step in which the
        technology is invested considering the construction time.

        :param optimization_setup: The optimization setup to add everything
        :param tech: name of technology
        :param year: yearly time step
        :return: investment time step
        """
        # get params and system
        construction_time = params.construction_time[tech]
        # conservative estimate of construction time (ceil)
        del_construction_time = int(
            np.ceil(construction_time / system.interval_between_years)
        )
        return year - del_construction_time


class TechnologyConstructor(ElementConstructor):
    element_class = Technology

    @override
    def has_elements(self) -> bool:
        """Checks if there are any elements of the class <Carrier>.

        :return: True if there are elements, False otherwise
        """
        return True

    ### --- classmethods to construct sets, parameters, variables, and constraints,
    # that correspond to Technology --- ###
    def construct_sets(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the pe.Sets of the class <Technology>.

        :param optimization_setup: The OptimizationSetup
        """
        logger.info("Constructing sets for Technology")

        # conversion technologies
        zen_model.sets.add_set(
            name="set_conversion_technologies",
            data=energy_system.set_conversion_technologies,
            doc="Set of conversion technologies",
        )
        # retrofitting technologies
        zen_model.sets.add_set(
            name="set_retrofitting_technologies",
            data=energy_system.set_retrofitting_technologies,
            doc="Set of retrofitting technologies",
        )
        # transport technologies
        zen_model.sets.add_set(
            name="set_transport_technologies",
            data=energy_system.set_transport_technologies,
            doc="Set of transport technologies",
        )
        # storage technologies
        zen_model.sets.add_set(
            name="set_storage_technologies",
            data=energy_system.set_storage_technologies,
            doc="Set of storage technologies",
        )
        # existing installed technologies
        zen_model.sets.add_set(
            name="set_technologies_existing",
            data=self.element_registry.get_attribute_of_all_elements(
                self.element_class, "set_technologies_existing"
            ),
            doc="Set of existing technologies",
            index_set="set_technologies",
        )
        # reference carriers
        zen_model.sets.add_set(
            name="set_reference_carriers",
            data=self.element_registry.get_attribute_of_all_elements(
                self.element_class, "reference_carrier"
            ),
            doc="set of all reference carriers correspondent to a technology. "
            "Indexed by set_technologies",
            index_set="set_technologies",
        )
        # # add pe.Sets of the child classes
        # for subclass in cls.__subclasses__():
        #     subclass.construct_sets(optimization_setup)

    def construct_params(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the pe.Params of the class <Technology>.

        :param optimization_setup: The OptimizationSetup
        """
        # construct pe.Param of the class <Technology>
        logger.info("Constructing parameters for Technology")

        # existing capacity
        self.add_parameter(
            zen_model,
            energy_system,
            name="capacity_existing",
            index_names=[
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_technologies_existing",
            ],
            capacity_types=True,
            doc="Parameter which specifies the existing technology size",
        )
        # existing capacity
        self.add_parameter(
            zen_model,
            energy_system,
            name="capacity_investment_existing",
            index_names=[
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_time_steps_yearly_entire_horizon",
            ],
            capacity_types=True,
            doc="Parameter specifying the size of the previously invested capacities",
        )
        # minimum capacity addition
        self.add_parameter(
            zen_model,
            energy_system,
            name="capacity_addition_min",
            index_names=["set_technologies", "set_capacity_types"],
            capacity_types=True,
            doc="Parameter which specifies the minimum capacity addition "
            "that can be installed",
        )
        # maximum capacity addition
        self.add_parameter(
            zen_model,
            energy_system,
            name="capacity_addition_max",
            index_names=["set_technologies", "set_capacity_types"],
            capacity_types=True,
            doc="Parameter which specifies the maximum capacity addition "
            "that can be installed",
        )
        # unbounded capacity addition
        self.add_parameter(
            zen_model,
            energy_system,
            name="capacity_addition_unbounded",
            index_names=["set_technologies"],
            doc="Parameter which specifies the unbounded capacity addition that can be "
            "added each year (only for delayed technology deployment)",
        )
        # lifetime existing technologies
        self.add_parameter(
            zen_model,
            energy_system,
            name="lifetime_existing",
            index_names=[
                "set_technologies",
                "set_location",
                "set_technologies_existing",
            ],
            doc="Parameter specifying the remaining lifetime of an existing technology",
        )
        # lifetime existing technologies
        self.add_parameter(
            zen_model,
            energy_system,
            name="capex_capacity_existing",
            index_names=[
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_technologies_existing",
            ],
            capacity_types=True,
            doc="Parameter which specifies the total capex of an existing technology "
            "which still has to be paid",
        )
        # variable specific opex
        self.add_parameter(
            zen_model,
            energy_system,
            name="opex_specific_variable",
            index_names=[
                "set_technologies",
                "set_location",
                "set_time_steps_operation",
            ],
            doc="Parameter which specifies the variable specific opex",
        )
        # fixed specific opex
        self.add_parameter(
            zen_model,
            energy_system,
            name="opex_specific_fixed",
            index_names=[
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_time_steps_yearly",
            ],
            capacity_types=True,
            doc="Parameter which specifies the fixed annual specific opex",
        )
        # lifetime newly built technologies
        self.add_parameter(
            zen_model,
            energy_system,
            name="lifetime",
            index_names=["set_technologies"],
            doc="Parameter which specifies the lifetime of a newly built technology",
        )
        # amortization time newly built technologies
        self.add_parameter(
            zen_model,
            energy_system,
            name="depreciation_time",
            index_names=["set_technologies"],
            doc="Parameter which specifies the depreciation time of a "
            "newly built technology",
        )
        # construction_time newly built technologies
        self.add_parameter(
            zen_model,
            energy_system,
            name="construction_time",
            index_names=["set_technologies"],
            doc="Parameter which specifies the construction time of a "
            "newly built technology",
        )
        # maximum diffusion rate, i.e., increase in capacity
        self.add_parameter(
            zen_model,
            energy_system,
            name="max_diffusion_rate",
            index_names=["set_technologies", "set_time_steps_yearly"],
            doc="Parameter which specifies the maximum diffusion rate which is the "
            "maximum increase in capacity between investment steps",
        )
        # capacity_limit of technologies
        self.add_parameter(
            zen_model,
            energy_system,
            name="capacity_limit",
            index_names=[
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_time_steps_yearly",
            ],
            capacity_types=True,
            doc="Parameter which specifies the capacity limit of technologies",
        )
        # NEW: lower capacity limit of technologies
        self.add_parameter(
            zen_model,
            energy_system,
            name="capacity_lower_limit",
            index_names=[
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_time_steps_yearly",
            ],
            capacity_types=True,
            doc="Parameter which specifies the lower capacity limit of technologies",
        )
        # minimum load relative to capacity
        self.add_parameter(
            zen_model,
            energy_system,
            name="min_load",
            index_names=[
                "set_technologies",
                "set_location",
                "set_time_steps_operation",
            ],
            doc="Parameter which specifies the minimum load of technology "
            "relative to installed capacity",
        )
        # maximum load relative to capacity
        self.add_parameter(
            zen_model,
            energy_system,
            name="max_load",
            index_names=[
                "set_technologies",
                "set_location",
                "set_time_steps_operation",
            ],
            doc="Parameter which specifies the maximum load of technology relative to "
            "installed capacity",
        )
        # carbon intensity
        self.add_parameter(
            zen_model,
            energy_system,
            name="carbon_intensity_technology",
            index_names=["set_technologies", "set_location"],
            doc="Parameter which specifies the carbon intensity of each technology",
        )
        # calculate additional existing parameters
        zen_model.parameters.add_parameter(
            name="existing_capacities",
            data=self.get_existing_quantity("capacity", zen_model, energy_system),
            doc="Parameter which specifies the total available capacity of existing "
            "technologies at the beginning of the optimization",
        )
        zen_model.parameters.add_parameter(
            name="existing_capex",
            data=self.get_existing_quantity(
                "cost_capex_overnight", zen_model, energy_system
            ),
            doc="Parameter which specifies the total capex of existing technologies at "
            "the beginning of the optimization",
        )

    def construct_vars(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the pe.Vars of the class <Technology>.

        :param optimization_setup: The OptimizationSetup
        """
        logger.info("Constructing variables for Technology")

        variables = zen_model.variables
        sets = zen_model.sets

        # TODO: This could be vectorized
        def capacity_bounds(tech, capacity_type, loc, time):
            """Return bounds of capacity for bigM expression.

            :param tech: tech index
            :param capacity_type: either power or energy
            :param loc: location of capacity
            :param time: investment time step
            :return: bounds: bounds of capacity
            """
            # bounds only needed for Big-M formulation,
            #   thus if any technology is modeled with on-off behavior
            if tech in techs_on_off:
                params = zen_model.parameters.dict_parameters
                capacity_existing = params.capacity_existing
                capacity_addition_max = params.capacity_addition_max
                capacity_limit = params.capacity_limit
                capacities_existing = 0
                for id_technology_existing in sets["set_technologies_existing"][tech]:
                    if (
                        params.lifetime_existing[tech, loc, id_technology_existing]
                        > params.lifetime[tech]
                    ):
                        if (
                            time
                            > params.lifetime_existing[
                                tech, loc, id_technology_existing
                            ]
                            - params.lifetime[tech]
                        ):
                            capacities_existing += capacity_existing[
                                tech, capacity_type, loc, id_technology_existing
                            ]
                    elif (
                        time
                        <= params.lifetime_existing[tech, loc, id_technology_existing]
                        + 1
                    ):
                        capacities_existing += capacity_existing[
                            tech, capacity_type, loc, id_technology_existing
                        ]

                capacity_addition_max = (
                    len(sets["set_time_steps_yearly"])
                    * capacity_addition_max[tech, capacity_type]
                )
                max_capacity_limit = capacity_limit[tech, capacity_type, loc, time]
                bound_capacity = min(
                    capacity_addition_max + capacities_existing,
                    max_capacity_limit + capacities_existing,
                )
                return 0, bound_capacity
            else:
                return 0, np.inf

        # bounds only needed for Big-M formulation,
        #   thus if any technology is modeled with on-off behavior
        techs_on_off = self.create_custom_set(
            ["set_technologies", "set_on_off"], zen_model, energy_system
        )[0]
        # construct pe.Vars of the class <Technology>
        # capacity technology
        variables.add_variable(
            name="capacity",
            index_sets=self.create_custom_set(
                [
                    "set_technologies",
                    "set_capacity_types",
                    "set_location",
                    "set_time_steps_yearly",
                ],
                zen_model,
                energy_system,
            ),
            bounds=capacity_bounds,
            doc="size of installed technology at location l and time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # capacity technology before current year
        variables.add_variable(
            name="capacity_previous",
            index_sets=self.create_custom_set(
                [
                    "set_technologies",
                    "set_capacity_types",
                    "set_location",
                    "set_time_steps_yearly",
                ],
                zen_model,
                energy_system,
            ),
            bounds=(0, np.inf),
            doc="size of installed technology at location l and BEFORE time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # built_capacity technology
        variables.add_variable(
            name="capacity_addition",
            index_sets=self.create_custom_set(
                [
                    "set_technologies",
                    "set_capacity_types",
                    "set_location",
                    "set_time_steps_yearly",
                ],
                zen_model,
                energy_system,
            ),
            bounds=(0, np.inf),
            doc="size of built technology (invested capacity after construction) "
            "at location l and time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # invested_capacity technology
        variables.add_variable(
            name="capacity_investment",
            index_sets=self.create_custom_set(
                [
                    "set_technologies",
                    "set_capacity_types",
                    "set_location",
                    "set_time_steps_yearly",
                ],
                zen_model,
                energy_system,
            ),
            bounds=(0, np.inf),
            doc="size of invested technology at location l and time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # capex of building capacity overnight
        variables.add_variable(
            name="cost_capex_overnight",
            index_sets=self.create_custom_set(
                [
                    "set_technologies",
                    "set_capacity_types",
                    "set_location",
                    "set_time_steps_yearly",
                ],
                zen_model,
                energy_system,
            ),
            bounds=(0, np.inf),
            doc="capex for building technology at location l and time t",
            unit_category={"money": 1},
        )
        # annual capex of having capacity
        variables.add_variable(
            name="cost_capex_yearly",
            index_sets=self.create_custom_set(
                [
                    "set_technologies",
                    "set_capacity_types",
                    "set_location",
                    "set_time_steps_yearly",
                ],
                zen_model,
                energy_system,
            ),
            bounds=(0, np.inf),
            doc="annual capex for having technology at location l",
            unit_category={"money": 1},
        )
        # total capex
        variables.add_variable(
            name="cost_capex_yearly_total",
            index_sets=sets["set_time_steps_yearly"],
            bounds=(0, np.inf),
            doc="total capex for installing all technologies in all locations "
            "at all times",
            unit_category={"money": 1},
        )
        # opex
        variables.add_variable(
            name="cost_opex_variable",
            index_sets=self.create_custom_set(
                ["set_technologies", "set_location", "set_time_steps_operation"],
                zen_model,
                energy_system,
            ),
            bounds=(0, np.inf),
            doc="opex for operating technology at location l and time t",
            unit_category={"money": 1, "time": -1},
        )
        # total opex
        variables.add_variable(
            name="cost_opex_yearly_total",
            index_sets=sets["set_time_steps_yearly"],
            bounds=(0, np.inf),
            doc="total opex all technologies and locations in year y",
            unit_category={"money": 1},
        )
        # yearly opex
        variables.add_variable(
            name="cost_opex_yearly",
            index_sets=self.create_custom_set(
                ["set_technologies", "set_location", "set_time_steps_yearly"],
                zen_model,
                energy_system,
            ),
            bounds=(0, np.inf),
            doc="yearly opex for operating technology at location l and year y",
            unit_category={"money": 1},
        )
        # carbon emissions
        variables.add_variable(
            name="carbon_emissions_technology",
            index_sets=self.create_custom_set(
                ["set_technologies", "set_location", "set_time_steps_operation"],
                zen_model,
                energy_system,
            ),
            doc="carbon emissions for operating technology at location l and time t",
            unit_category={"emissions": 1, "time": -1},
        )
        # total carbon emissions technology
        variables.add_variable(
            name="carbon_emissions_technology_total",
            index_sets=sets["set_time_steps_yearly"],
            doc="total carbon emissions for operating technology",
            unit_category={"emissions": 1},
        )

        # install technology
        # Note: binary variables are written into the lp file by linopy even if they
        # are not relevant for the optimization, which makes all problems MIPs.
        # Therefore, we only add binary variables, if really necessary. Gurobi can
        # handle this by noting that the binary variables are not part of the model,
        # however, only if there are no binary variables at all, it is possible to get
        # the dual values of the constraints.
        mask = self._technology_installation_mask(zen_model, energy_system)
        if mask.any():
            variables.add_variable(
                name="technology_installation",
                index_sets=self.create_custom_set(
                    [
                        "set_technologies",
                        "set_capacity_types",
                        "set_location",
                        "set_time_steps_yearly",
                    ],
                    zen_model,
                    energy_system,
                ),
                binary=True,
                doc="installment of a technology at location l and time t",
                mask=mask,
                unit_category=None,
            )

        # on-off variables
        # We remove the binary variables if there are any no constraints that use them
        techs_on_off, index_list = self.create_custom_set(
            [
                "set_technologies",
                "set_on_off",
                "set_location",
                "set_time_steps_operation",
            ],
            zen_model,
            energy_system,
        )
        index_list.pop(1)
        mask_on_off = zen_model.sets.indices_to_mask(techs_on_off, index_list, (0, 0))[
            0
        ]
        times = zen_model.sets["set_time_steps_operation"]
        ts = energy_system.time_steps
        time_step_year = xr.DataArray(
            [ts.convert_time_step_operation2year(t) for t in times.data],
            coords=[times],
            dims=["set_time_steps_operation"],
        )
        mask_nonzero_cap_limit = (
            zen_model.parameters.capacity_limit.sel(
                {"set_capacity_types": "power", "set_time_steps_yearly": time_step_year}
            )
            != 0
        )
        mask_on_off = mask_on_off & mask_nonzero_cap_limit.drop_vars(
            "set_capacity_types"
        )
        variables.add_variable(
            name="tech_on_var",
            index_sets=self.create_custom_set(
                ["set_technologies", "set_location", "set_time_steps_operation"],
                zen_model,
                energy_system,
            ),
            mask=mask_on_off,
            doc="Binary variable which equals 1 when technology is switched on at "
            "location l and time t",
            binary=True,
            unit_category=None,
        )
        variables.add_variable(
            name="capacity_on_off_helper_var",
            index_sets=self.create_custom_set(
                ["set_technologies", "set_location", "set_time_steps_operation"],
                zen_model,
                energy_system,
            ),
            bounds=(0, np.inf),
            mask=mask_on_off,
            doc="Helper variable substituting the product of capacity and tech_on_var",
            unit_category={"energy_quantity": 1, "time": -1},
        )

    def construct_constraints(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the Constraints of the class <Technology>.

        :param optimization_setup: The OptimizationSetup
        """
        logger.info("Constructing constraints for Technology")
        model = zen_model.lp_model

        # construct pe.Constraints of the class <Technology>
        rules = TechnologyRules(self.config, self.context, self.element_registry)
        #  technology capacity_limit
        rules.constraint_technology_capacity_limit(zen_model, energy_system)

        # NEW: technology capacity_lower_limit (Lower Limit)
        rules.constraint_technology_capacity_lower_limit(zen_model, energy_system)

        # minimum capacity
        rules.constraint_technology_min_capacity_addition(zen_model, energy_system)

        # maximum capacity
        rules.constraint_technology_max_capacity_addition(zen_model, energy_system)

        # construction period
        rules.constraint_technology_construction_time(zen_model, energy_system)

        # lifetime
        rules.constraint_technology_lifetime(zen_model, energy_system)

        # limit diffusion rate
        rules.constraint_technology_diffusion_limit(zen_model, energy_system)

        # annual capex of having capacity
        index_values, index_names = self.create_custom_set(
            [
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_time_steps_yearly",
            ],
            zen_model,
            energy_system,
        )
        rules.constraint_cost_capex_yearly(
            zen_model, ZenIndex(index_values, index_names)
        )

        # total capex of all technologies
        rules.constraint_cost_capex_yearly_total(zen_model, energy_system)

        # yearly opex
        rules.constraint_cost_opex_yearly(zen_model, energy_system)

        # total opex of all technologies
        rules.constraint_cost_opex_yearly_total(zen_model, energy_system)

        # total carbon emissions of technologies
        rules.constraint_carbon_emissions_technology_total(zen_model, energy_system)

        # min load constraints
        n_cons = len(model.constraints.items())
        techs_on_off = self.create_custom_set(
            ["set_technologies", "set_on_off"], zen_model, energy_system
        )[0]
        rules.constraint_technology_on_off(zen_model, energy_system, techs_on_off)

        # if nothing was added we can remove the tech vars again
        if len(model.constraints.items()) == n_cons:
            model.variables.remove("tech_on_var")
            model.variables.remove("capacity_on_off_helper_var")

    def _technology_installation_mask(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ) -> xr.DataArray:
        """Check if the binary variable is necessary.

        :param optimization_setup: optimization setup object
        """
        params = zen_model.parameters
        model = zen_model.lp_model
        sets = zen_model.sets

        mask = xr.DataArray(
            False,
            coords=[
                model.variables.coords["set_time_steps_yearly"],
                model.variables.coords["set_technologies"],
                model.variables.coords["set_capacity_types"],
                model.variables.coords["set_location"],
            ],
        )

        # used in transport technology
        techs = list(sets["set_transport_technologies"])
        if len(techs) > 0:
            edges = list(sets["set_edges"])
            sub_mask = (
                params.distance.loc[techs, edges]
                * params.capex_per_distance_transport.loc[techs, edges]
                != 0
            )
            sub_mask = sub_mask.rename(
                {
                    "set_transport_technologies": "set_technologies",
                    "set_edges": "set_location",
                }
            )
            mask.loc[:, techs, :, edges] |= sub_mask

        # used in constraint_technology_min_capacity_addition
        mask = mask | (
            params.capacity_addition_min.notnull() & (params.capacity_addition_min != 0)
        )

        # used in constraint_technology_max_capacity_addition
        index_values, index_names = self.create_custom_set(
            [
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_time_steps_yearly",
            ],
            zen_model,
            energy_system,
        )
        index = ZenIndex(index_values, index_names)
        sub_mask = (
            params.capacity_addition_max.notnull()
            & (params.capacity_addition_max != np.inf)
            & (params.capacity_addition_max != 0)
        )
        for tech, capacity_type in index.get_unique([0, 1]):
            locs = index.get_values(locs=[tech, capacity_type], levels=2, unique=True)
            mask.loc[:, tech, capacity_type, locs] |= sub_mask.loc[tech, capacity_type]

        return mask

    def get_existing_quantity(
        self,
        type_existing_quantity: str,
        zen_model: ZenModel,
        energy_system: "EnergySystem",
    ):
        """Get existing capacities of all technologies.

        :param optimization_setup: The OptimizationSetup the element is part of
        :param type_existing_quantity: capacity or cost_capex_overnight
        :return: The existing capacities
        """
        index_values, index_names = self.create_custom_set(
            [
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_time_steps_yearly",
            ],
            zen_model,
            energy_system,
        )
        # get all the capacities
        index_arrs = IndexSet.tuple_to_arr(index_values, index_names)
        coords = [
            zen_model.sets.get_coord(data, name)
            for data, name in zip(index_arrs, index_names, strict=False)
        ]
        existing_quantities = xr.DataArray(np.nan, coords=coords, dims=index_names)
        values = np.zeros(len(index_values))
        for i, (tech, capacity_type, loc, time) in enumerate(index_values):
            values[i] = self._get_available_existing_quantity(
                tech,
                capacity_type,
                loc,
                time,
                type_existing_quantity,
                zen_model,
                energy_system,
            )
        existing_quantities.loc[index_arrs] = values
        return existing_quantities

    def _get_available_existing_quantity(
        self,
        tech,
        capacity_type,
        loc,
        year,
        type_existing_quantity,
        zen_model: ZenModel,
        energy_system: "EnergySystem",
    ):
        """Gets the existing quantity of 'tech' at investment time step 'time'.

        returns existing quantity of 'tech', that is still available at invest
        time step 'time'. Either capacity or capex.

        :param optimization_setup: The OptimizationSetup the element is part of
        :param tech: name of technology
        :param capacity_type: type of capacity
        :param loc: location (node or edge) of existing capacity
        :param year: current yearly time step
        :param type_existing_quantity: capex or capacity
        :return: existing_quantity: existing capacity or capex of existing capacity
        """
        params = zen_model.parameters.dict_parameters
        sets = zen_model.sets
        existing_quantity = 0
        if type_existing_quantity == "capacity":
            existing_variable = params.capacity_existing
        elif type_existing_quantity == "cost_capex_overnight":
            existing_variable = params.capex_capacity_existing
        else:
            raise KeyError(f"Wrong type of existing quantity {type_existing_quantity}")

        for id_capacity_existing in sets["set_technologies_existing"][tech]:
            is_existing = self.get_if_capacity_still_existing(
                tech,
                year,
                loc=loc,
                id_capacity_existing=id_capacity_existing,
                zen_model=zen_model,
                energy_system=energy_system,
            )
            # if still available at first base time step, add to list
            if is_existing:
                existing_quantity += existing_variable[
                    tech, capacity_type, loc, id_capacity_existing
                ]
        return existing_quantity

    def get_if_capacity_still_existing(
        self,
        tech,
        year,
        loc,
        id_capacity_existing,
        zen_model: ZenModel,
        energy_system: "EnergySystem",
    ):
        """Returns boolean if capacity still exists at yearly time step 'year'.

        :param optimization_setup: The optimization setup to add everything
        :param tech: name of technology
        :param year: yearly time step
        :param loc: location
        :param id_capacity_existing: id of existing capacity
        :return: boolean if still existing
        """
        # get params and system
        params = zen_model.parameters.dict_parameters
        # get lifetime of existing capacity
        lifetime_existing = params.lifetime_existing[tech, loc, id_capacity_existing]
        lifetime = params.lifetime[tech]
        delta_lifetime = lifetime_existing - lifetime
        # reference year of current optimization horizon
        current_year_horizon = energy_system.set_time_steps_yearly[0]
        if delta_lifetime >= 0:
            cutoff_year = (
                year - current_year_horizon
            ) * self.config.system.interval_between_years
            return cutoff_year >= delta_lifetime
        else:
            cutoff_year = (
                year - current_year_horizon + 1
            ) * self.config.system.interval_between_years
            return cutoff_year <= lifetime_existing


class TechnologyRules(GenericRule):
    """Rules for the Technology class."""

    def __init__(
        self, config: Config, context: Context, element_registry: "ElementRegistry"
    ):
        """Inits the rules.

        :param optimization_setup: OptimizationSetup of the element
        """
        self.element_registry = element_registry
        super().__init__(config, context)

    def constraint_cost_capex_yearly_total(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Sums over all technologies to calculate total capex.

        .. math::
            CAPEX_y = \\sum_{h\\in\\mathcal{H}}\\sum_{p\\in\\mathcal{P}}A_{h,p,y} +
            \\sum_{k\\in\\mathcal{K}}\\sum_{n\\in\\mathcal{N}}A^\\mathrm{e}_{k,n,y}

        :math:`A_{h,p,y}`: annual capex of technology :math:`h` at location :math:`p`
        in year :math:`y`

        """
        lhs = zen_model.lp_model.variables[
            "cost_capex_yearly_total"
        ] - zen_model.lp_model.variables["cost_capex_yearly"].sum(
            ["set_technologies", "set_capacity_types", "set_location"]
        )
        rhs = 0
        constraints = lhs == rhs

        zen_model.constraints.add_constraint(
            "constraint_cost_capex_yearly_total", constraints
        )

    def constraint_cost_opex_yearly_total(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Sums over all technologies to calculate total opex.

        .. math::
            OPEX_y = \\sum_{h\\in\\mathcal{H}}\\sum_{p\\in\\mathcal{P}} OPEX_{h,p,y}

        :math:`OPEX_{h,p,y}`: opex of operating technology :math:`h` at
        location :math:`p` in year :math:`y`

        """
        lhs = zen_model.lp_model.variables[
            "cost_opex_yearly_total"
        ] - zen_model.lp_model.variables["cost_opex_yearly"].sum(
            ["set_technologies", "set_location"]
        )
        rhs = 0
        constraints = lhs == rhs

        zen_model.constraints.add_constraint(
            "constraint_cost_opex_yearly_total", constraints
        )

    def constraint_technology_capacity_limit(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Limited capacity_limit of technology.

        .. math::
            \\text{if existing capacities < capacity limit: }
            s^\\mathrm{max}_{h,p,y} \\geq S_{h,p,y}
        .. math::
            \\text{else: } \\Delta S_{h,p,y} = 0

        :math:`S_{h,p,y}`: installed capacity of technology :math:`h` at
        location :math:`p` in year :math:`y` \n
        :math:`s^\\mathrm{max}_{h,p,y}`: capacity limit of technology :math:`h` at
        location :math:`p` in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}`: size of built technology :math:`h` (invested
        capacity after construction) at location :math:`p` in year :math:`y`

        """
        # if the capacity limit is not reached by the existing capacities,
        # the capacity is constrained by the capacity limit.
        # if the capacity limit is reached, the capacity addition is 0.
        capacity_limit_not_reached = (
            zen_model.parameters.existing_capacities
            < zen_model.parameters.capacity_limit
        )
        # create mask so that skipped if capacity_limit is inf
        m = zen_model.parameters.capacity_limit != np.inf

        lhs_not_reached = (
            zen_model.lp_model.variables["capacity"]
            .where(m)
            .where(capacity_limit_not_reached)
        )
        rhs_not_reached = zen_model.parameters.capacity_limit.where(m, 0.0).where(
            capacity_limit_not_reached, 0.0
        )
        constraints_not_reached = lhs_not_reached <= rhs_not_reached
        lhs_reached = (
            zen_model.lp_model.variables["capacity_addition"]
            .where(m)
            .where(~capacity_limit_not_reached)
        )
        rhs_reached = 0
        if not self.config.system.allow_investment:
            lhs_reached = zen_model.lp_model.variables["capacity_addition"]
        constraints_reached = lhs_reached == rhs_reached

        zen_model.constraints.add_constraint(
            "constraint_technology_capacity_limit_not_reached", constraints_not_reached
        )
        zen_model.constraints.add_constraint(
            "constraint_technology_capacity_limit_reached", constraints_reached
        )

    def constraint_technology_capacity_lower_limit(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Constraint that installed capacity must be >= the defined lower limit."""

        # In TechnologyRules, we access variables and parameters directly via self
        capacity = zen_model.lp_model.variables["capacity"]
        capacity_lower_limit = zen_model.parameters.capacity_lower_limit

        # Create a mask so we only build constraints
        # where the user actually provided a number
        mask = capacity_lower_limit > 0.0

        # Apply the mask using xarray's .where() so we don't build empty/NaN constraints
        lhs = capacity.where(mask)
        rhs = capacity_lower_limit.where(mask, 0.0)

        # Total Capacity >= Lower Bound
        constraint = lhs >= rhs

        # Add the constraint to the model
        zen_model.constraints.add_constraint(
            "constraint_technology_capacity_lower_limit", constraint
        )

    def constraint_technology_min_capacity_addition(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Min capacity addition of technology.

        .. math::
            \\Delta s^\\mathrm{min}_{h} g_{i,p,y} \\le \\Delta S_{h,p,y}

        :math:`\\Delta s^\\mathrm{min}_{h}`: minimum capacity addition of
        technology :math:`h` \n
        :math:`g_{i,p,y}`: binary variable which equals 1 if technology is installed
        at location :math:`p` in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}`: size of built technology :math:`h` (invested
        capacity after construction) at location :math:`p` in year :math:`y`

        """
        capacity_addition_min = zen_model.parameters.capacity_addition_min
        mask = (capacity_addition_min != 0) & (capacity_addition_min.notnull())

        # if mask is empty, return None
        if not mask.any():
            return None

        lhs = mask * (
            capacity_addition_min
            * zen_model.lp_model.variables["technology_installation"]
            - zen_model.lp_model.variables["capacity_addition"]
        )
        rhs = 0
        constraints = lhs <= rhs

        ### return
        zen_model.constraints.add_constraint(
            "constraint_technology_min_capacity_addition", constraints
        )

    def constraint_technology_max_capacity_addition(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Max capacity addition of technology.

        .. math::
            s^\\mathrm{max}_{h} g_{i,p,y} \\ge \\Delta S_{h,p,y}

        :math:`s^\\mathrm{add, max}_{h}`: maximum capacity addition of
        technology :math:`h`  \n
        :math:`g_{i,p,y}`: binary variable which equals 1 if technology is installed
        at location :math:`p` in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}`: size of built technology :math:`h` (invested
        capacity after construction) at location :math:`p` in year :math:`y`

        """
        capacity_addition_max = zen_model.parameters.capacity_addition_max
        mask = (
            (capacity_addition_max != np.inf)
            & (capacity_addition_max != 0)
            & (capacity_addition_max.notnull())
        )

        # if mask is empty, return None
        if not mask.any():
            return None
        lhs = mask * (
            capacity_addition_max
            * zen_model.lp_model.variables["technology_installation"]
            - zen_model.lp_model.variables["capacity_addition"]
        )
        rhs = 0
        constraints = lhs >= rhs

        zen_model.constraints.add_constraint(
            "constraint_technology_max_capacity_addition", constraints
        )

    def constraint_technology_construction_time(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Construction time of technology: time between investment and availability.

        .. math::
            \\text{if start time step in set time steps yearly: } \\Delta S_{h,p,y} =
            \\Delta S_{h,p,(y-dy^{\\mathrm{construction}})}^\\mathrm{invest}
        .. math::
            \\text{elif start time step in set time steps yearly entire horizon:}
            \\Delta S_{h,p,y} =
            \\Delta s^\\mathrm{ex,invest}_{h,p,(y-dy^{\\mathrm{construction}})}
        .. math::
            \\text{else: } \\Delta S_{h,p,y} = 0

        :math:`\\Delta S_{h,p,y}`: size of built technology :math:`h` (invested
        capacity after construction) at location :math:`p` in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}^\\mathrm{invest}`: size of invested technology at
        location :math:`p` in year :math:`y` \n
        :math:`\\Delta s^\\mathrm{ex,invest}_{h,p,y}`: size of the previously invested
        capacities at location :math:`p` in year :math:`y` \n

        """
        # get investment time step
        investment_time = pd.Series(
            {
                (
                    t,
                    y,
                    Technology.get_investment_time_step(
                        zen_model.parameters.dict_parameters,
                        self.config.system,
                        t,
                        y,
                    ),
                ): 1
                for t, y in itertools.product(
                    zen_model.sets["set_technologies"],
                    zen_model.sets["set_time_steps_yearly"],
                )
            }
        )
        investment_time.index.names = [
            "set_technologies",
            "set_time_steps_yearly",
            "set_time_steps_construction",
        ]

        # select masks
        mask_current_time_steps = investment_time.index.get_level_values(
            "set_time_steps_construction"
        ).isin(zen_model.sets["set_time_steps_yearly"])
        mask_existing_time_steps = (
            investment_time.isin(zen_model.sets["set_time_steps_yearly_entire_horizon"])
            & ~mask_current_time_steps
        )
        # broadcast capacity investment and capacity investment existing
        capacity_investment = zen_model.lp_model.variables["capacity_investment"]
        investment_time_current = (
            investment_time[mask_current_time_steps]
            .dropna()
            .to_xarray()
            .broadcast_like(capacity_investment.mask)
            .fillna(0)
        )
        investment_time_existing = (
            investment_time[mask_existing_time_steps]
            .dropna()
            .to_xarray()
            .broadcast_like(capacity_investment.mask)
            .fillna(0)
        )
        # gets the time steps where no investment can be made without the
        #   addition exceeding the horizon
        investment_time_outside = (1 - investment_time_current).min(
            "set_time_steps_yearly"
        )

        capacity_investment = capacity_investment.rename(
            {"set_time_steps_yearly": "set_time_steps_construction"}
        )
        capacity_investment_addition = capacity_investment.broadcast_like(
            investment_time_current
        )
        capacity_investment_existing = zen_model.parameters.capacity_investment_existing
        capacity_investment_existing = capacity_investment_existing.rename(
            {"set_time_steps_yearly_entire_horizon": "set_time_steps_construction"}
        ).broadcast_like(investment_time_existing)

        ### formulate constraint
        lhs = lp.merge(
            [
                1 * zen_model.lp_model.variables["capacity_addition"],
                -(investment_time_current * capacity_investment_addition).sum(
                    "set_time_steps_construction"
                ),
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = (investment_time_existing * capacity_investment_existing).sum(
            "set_time_steps_construction"
        )
        rhs = xr.align(lhs.const, rhs, join="left")[1]
        constraints = lhs == rhs
        # constrain capacity_investment where no investment can be made
        #   without the addition exceeding the horizon
        lhs_outside = self.align_and_mask(capacity_investment, investment_time_outside)
        rhs_outside = 0
        constraints_outside = lhs_outside == rhs_outside

        zen_model.constraints.add_constraint(
            "constraint_technology_construction_time", constraints
        )
        zen_model.constraints.add_constraint(
            "constraint_technology_construction_time_outside", constraints_outside
        )

    def constraint_technology_lifetime(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Calculates remaining capacity of technologies based on the lifetime.

        limited lifetime of the technologies. calculates 'capacity', i.e., the
        capacity at the end of the year and 'capacity_previous', i.e., the capacity at
        the beginning of the year.

        .. math::
            S_{h,p,y} = \\sum_{\\tilde{y}=\\max(y_0,y-\\lceil\\frac{l_h}
            {\\Delta^\\mathrm{y}}\\rceil+1)}^y \\Delta S_{h,p,\\tilde{y}}
            + \\sum_{\\hat{y}=\\psi(\\min(y_0-1,y-\\lceil\\frac{l_h}
            {\\Delta^\\mathrm{y}}\\rceil+1))}^{\\psi(y_0)}
            \\Delta s^\\mathrm{ex}_{h,p,\\hat{y}}

        :math:`S_{h,p,y}`: installed capacity of technology :math:`h` at
        location :math:`p` in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}`: size of built technology :math:`h` (invested capacity
        after construction) at location :math:`p` in year :math:`y` \n
        :math:`\\Delta s^\\mathrm{ex}_{h,p,y}`: size of the previously invested
        capacities at location :math:`p` in year :math:`y`
        """
        lt_range = pd.MultiIndex.from_tuples(
            [
                (t, y, py)
                for t, y in itertools.product(
                    zen_model.sets["set_technologies"],
                    zen_model.sets["set_time_steps_yearly"],
                )
                for py in list(self.get_lifetime_range(t, y, zen_model))
            ],
            names=[
                "set_technologies",
                "set_time_steps_yearly",
                "set_time_steps_yearly_prev",
            ],
        )
        lt_range = pd.Series(index=lt_range, data=-1)
        lt_range = (
            lt_range.to_xarray()
            .broadcast_like(zen_model.lp_model.variables["capacity"].lower)
            .fillna(0)
        )
        capacity_addition = zen_model.lp_model.variables["capacity_addition"].rename(
            {"set_time_steps_yearly": "set_time_steps_yearly_prev"}
        )
        capacity_addition = capacity_addition.broadcast_like(lt_range)
        expr = (lt_range * capacity_addition).sum("set_time_steps_yearly_prev")
        lhs = lp.merge(
            [1 * zen_model.lp_model.variables["capacity"], expr],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        lhs_previous = lp.merge(
            [
                1 * zen_model.lp_model.variables["capacity_previous"],
                expr,
                1 * zen_model.lp_model.variables["capacity_addition"],
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = xr.align(
            lhs.const, zen_model.parameters.existing_capacities, join="left"
        )[1]
        constraints = lhs == rhs
        constraints_previous = lhs_previous == rhs

        ### return
        zen_model.constraints.add_constraint(
            "constraint_technology_lifetime", constraints
        )
        zen_model.constraints.add_constraint(
            "constraint_technology_lifetime_previous", constraints_previous
        )

    def constraint_technology_diffusion_limit(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Limits technology diffusion based on existing capacity in the previous year.

        For storage and conversion technologies: \n
        .. math::
            \\Delta S_{k,n,y}\\leq ((1+\\vartheta_k)^{\\mathrm{dy}}-1)(K_{k,n,y} +
            \\omega \\sum_{\\tilde{n}\\in\\tilde{\\mathcal{N}}}K_{k,\\tilde{n},y})
            +\\mathrm{dy}(\\xi\\sum_{\\tilde{k}
            \\in\\tilde{\\mathcal{K}}}S_{\\tilde{k},n,y} + \\zeta_k)

        For transport technologies: \n
        .. math::
            \\Delta S_{j,e,y}\\leq ((1+\\vartheta_j)^{\\mathrm{dy}}-1)K_{j,e,y}
            + \\mathrm{dy}(\\xi\\sum_{\\tilde{j}\\in\\tilde{\\mathcal{J}}}
            S_{\\tilde{j},e,y} + \\zeta_j)

        :math:`\\Delta S_{j,e,y}`: size of built technology :math:`j` (invested capacity
        after construction) at location :math:`e` in year :math:`y` \n
        :math:`\\vartheta_j`: maximum diffusion rate of technology :math:`j` which is
        the maximum increase in capacity between investment steps \n
        :math:`K_{j,e,y}`: existing knowledge of how to install the technology :math:`j`
        at location :math:`e` in year :math:`y` \n
        :math:`\\xi`: parameter which specifies the unbounded market share \n
        :math:`\\zeta_j`: parameter which specifies the unbounded capacity addition that
        can be added each year (only for delayed technology deployment) \n
        :math:`dy`: interval between planning periods\n
        :math:`\\omega`: parameter which specifies the knowledge spillover rate

        """
        # load variables and parameters
        capacity_addition = zen_model.lp_model.variables["capacity_addition"]
        capacity_existing = zen_model.parameters.capacity_existing
        knowledge_depreciation_rate = zen_model.parameters.knowledge_depreciation_rate
        interval_between_years = self.config.system.interval_between_years
        spillover_rate = zen_model.parameters.knowledge_spillover_rate
        # technology diffusion rate per investment period
        tdr = (
            1 + zen_model.parameters.max_diffusion_rate
        ) ** interval_between_years - 1
        tdr = tdr.broadcast_like(capacity_addition.lower)
        tdr_sum = tdr.sum("set_location")
        mask_inf_tdr = ~(tdr == np.inf)
        mask_inf_tdr_sum = ~(tdr_sum == np.inf)
        # if all tdr are inf, we can skip the constraint
        if (~mask_inf_tdr).all():
            return
        # mask for knowledge spillover rate (sr) to exclude transport technologies
        mask_technology_type = pd.Series(
            index=pd.Index(zen_model.sets["set_technologies"]), data=1
        )
        mask_technology_type.index.name = "set_technologies"
        mask_technology_type[
            mask_technology_type.index.isin(
                zen_model.sets["set_transport_technologies"]
            )
        ] = 0
        mask_technology_type = mask_technology_type.to_xarray()
        # create mask for knowledge spillover rate (sr) to exclude edges
        mask_location = pd.Series(
            index=pd.Index(capacity_addition.coords["set_location"]), data=1
        )
        mask_location.index.name = "set_location"
        mask_location[mask_location.index.isin(zen_model.sets["set_edges"])] = 0
        mask_location = mask_location.to_xarray()
        # mask match technology type and location
        mask_transport_edge = (1 - mask_technology_type) & (1 - mask_location)
        mask_not_transport_not_edge = mask_technology_type & mask_location
        mask_technology_location = mask_transport_edge | mask_not_transport_not_edge
        # create xarray for previous years
        years = pd.MultiIndex.from_tuples(
            [
                (y, py)
                for y, py in itertools.product(
                    zen_model.sets["set_time_steps_yearly"],
                    zen_model.sets["set_time_steps_yearly"],
                )
                if py < y
            ],
            names=["set_time_steps_yearly", "set_time_steps_yearly_prev"],
        )
        # only formulate term_knowledge if there are previous years
        term_knowledge_no_spillover = capacity_addition.where(
            xr.DataArray(False)
        )  # dummy term
        term_knowledge = capacity_addition.where(xr.DataArray(False))  # dummy term
        if len(years) != 0:
            # kdr for capacity additions
            kdr = {
                (y, py): (1 - knowledge_depreciation_rate)
                ** (interval_between_years * (y - 1 - py))
                for y, py in years
            }
            kdr = pd.Series(kdr)
            kdr.index.names = ["set_time_steps_yearly", "set_time_steps_yearly_prev"]
            kdr = kdr.to_xarray().fillna(0)

            years = pd.Series(index=years, data=1)
            years = years.to_xarray().fillna(0)
            # expand and sum capacity addition over all nodes for spillover
            capacity_addition_years = capacity_addition.rename(
                {"set_time_steps_yearly": "set_time_steps_yearly_prev"}
            ).broadcast_like(years)
            kdr = kdr.broadcast_like(capacity_addition_years.lower)
            term_knowledge_no_spillover = tdr * (capacity_addition_years * kdr).sum(
                "set_time_steps_yearly_prev"
            )
            # if spillover rate is not inf, calculate term knowledge with spillover
            if spillover_rate != np.inf:
                location_index = pd.Series(
                    index=pd.MultiIndex.from_product(
                        [
                            capacity_addition.coords["set_location"].values,
                            capacity_addition.coords["set_location"].values,
                        ],
                        names=["set_location", "set_location_temp"],
                    )
                ).to_xarray()
                capacity_addition_location = (
                    capacity_addition_years.rename(
                        {"set_location": "set_location_temp"}
                    )
                    .broadcast_like(location_index)
                    .sel({"set_location_temp": zen_model.sets["set_nodes"]})
                    .sum("set_location_temp")
                )
                # calculate term spillover
                term_spillover = capacity_addition_location - capacity_addition_years
                sr = xr.full_like(term_spillover.const, spillover_rate)
                sr = sr.where(mask_technology_type, 0).where(mask_location, 0)
                # annual knowledge addition
                term_knowledge = capacity_addition_years + sr * term_spillover
                term_knowledge = tdr * (term_knowledge * kdr).sum(
                    "set_time_steps_yearly_prev"
                )
        # unbounded market share --> only for same technology class
        capacity_previous = zen_model.lp_model.variables["capacity_previous"]
        market_share_unbounded = {
            (t, ot): (
                zen_model.parameters.market_share_unbounded
                if zen_model.sets["set_reference_carriers"][t][0]
                == zen_model.sets["set_reference_carriers"][ot][0]
                else 0
            )
            for t in zen_model.sets["set_technologies"]
            for ot in self._get_class_set_of_element(zen_model, t, Technology)
        }
        market_share_unbounded = pd.Series(market_share_unbounded)
        market_share_unbounded.index.names = [
            "set_technologies",
            "set_other_technologies",
        ]
        market_share_unbounded = (
            market_share_unbounded.to_xarray()
            .broadcast_like(capacity_previous.lower)
            .fillna(0)
        )
        mask_market_share_unbounded = market_share_unbounded != 0
        term_unbounded_addition = (
            (
                market_share_unbounded
                * capacity_previous.rename(
                    {"set_technologies": "set_other_technologies"}
                )
            )
            .where(mask_market_share_unbounded)
            .sum("set_other_technologies")
        )
        # existing capacities
        delta_years = interval_between_years * (
            capacity_addition.coords["set_time_steps_yearly"]
            - 1
            - energy_system.set_time_steps_yearly[0]
        )
        lifetime_existing = zen_model.parameters.lifetime_existing
        lifetime = zen_model.parameters.lifetime
        kdr_existing = (1 - knowledge_depreciation_rate) ** (
            delta_years + lifetime - lifetime_existing
        )
        capacity_existing_total_nosr = capacity_existing
        # capacity addition unbounded
        capacity_addition_unbounded = zen_model.parameters.capacity_addition_unbounded
        capacity_addition_unbounded = capacity_addition_unbounded.broadcast_like(tdr)
        capacity_addition_unbounded = capacity_addition_unbounded.where(
            mask_technology_location, 0
        )
        # build constraints for all nodes summed ("sn")
        lhs_sn = lp.merge(
            [
                1 * capacity_addition,
                -1 * term_knowledge_no_spillover,
                -1 * term_unbounded_addition,
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        ).sum("set_location")
        rhs_sn = (
            tdr
            * (capacity_existing_total_nosr * kdr_existing).sum(
                "set_technologies_existing"
            )
            + capacity_addition_unbounded
        ).sum("set_location")
        rhs_sn = rhs_sn.broadcast_like(lhs_sn.const)
        # mask for tdr == inf
        lhs_sn = self.align_and_mask(lhs_sn, mask_inf_tdr_sum)
        rhs_sn = self.align_and_mask(rhs_sn, mask_inf_tdr_sum)
        # combine constraint
        constraints_sn = lhs_sn <= rhs_sn
        zen_model.constraints.add_constraint(
            "constraint_technology_diffusion_limit_total", constraints_sn
        )
        # build constraints for all nodes ("an") if spillover rate is not inf
        if spillover_rate != np.inf:
            # existing capacities with spillover
            capacity_existing_total = capacity_existing + spillover_rate * (
                capacity_existing.sum("set_location") - capacity_existing
            ).where(mask_technology_type, 0)
            lhs_an = lp.merge(
                [
                    1 * capacity_addition,
                    -1 * term_knowledge,
                    -1 * term_unbounded_addition,
                ],
                compat="broadcast_equals",
                join="outer",
                cls=LinearExpression,
            )
            rhs_an = (
                tdr
                * (capacity_existing_total * kdr_existing).sum(
                    "set_technologies_existing"
                )
                + capacity_addition_unbounded
            )
            rhs_an = rhs_an.broadcast_like(lhs_an.const)
            # mask for tdr == inf
            lhs_an = self.align_and_mask(lhs_an, mask_inf_tdr)
            rhs_an = self.align_and_mask(rhs_an, mask_inf_tdr)
            # combine constraint
            constraints_an = lhs_an <= rhs_an
            zen_model.constraints.add_constraint(
                "constraint_technology_diffusion_limit", constraints_an
            )

    def _get_class_set_of_element(
        self, zen_model: ZenModel, element_name: str, class_name: type[Element]
    ) -> ZenSet:
        """Returns the set of all elements in the class of the element.

        :param element_name: name of element
        :param klass: class of the elements to return
        :return: class_set: set of all elements in the class of the element
        """
        element = self.element_registry.get_element(class_name, element_name)
        if element is None:
            raise ValueError(f"Element {element_name} not found in class {class_name}")
        return zen_model.sets[element.label]

    def constraint_cost_capex_yearly(self, zen_model: ZenModel, index: ZenIndex):
        """Aggregates the capex of built capacity and of existing capacity.

        .. math::
            A_{h,p,y} = f_h (\\sum_{\\tilde{y} = \\max(y_0,y-\\lceil\\frac{l_h}
            {\\mathrm{dy}}\\rceil+1)}^y \\alpha_{h,y}\\Delta S_{h,p,\\tilde{y}}
            + \\sum_{\\hat{y}=\\psi(\\min(y_0-1,y-\\lceil\\frac{l_h}
            {\\mathrm{dy}}\\rceil+1))}^{\\psi(y_0)} \\alpha_{h,y_0}
            \\Delta s^\\mathrm{ex}_{h,p,\\hat{y}})

        :math:`A_{h,p,y}`: annual capex of technology :math:`h` at location :math:`p`
        in year :math:`y` \n
        :math:`f_h`: annuity factor of technology :math:`h` \n
        :math:`\\alpha_{h,y}`: unit cost of capital investment of technology :math:`h`
        in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}`: size of built technology :math:`h` (invested capacity
        after construction) at location :math:`p` in year :math:`y` \n
        :math:`\\Delta s^\\mathrm{ex}_{h,p,y}`: size of the previously added capacities
        at location :math:`p` in year :math:`y` \n
        :math:`l_h`: depreciation time of technology :math:`h`   \n
        :math:`\\mathrm{dy}`: interval between planning periods


        """
        ### masks
        # not needed

        # Annuity factor
        dr = zen_model.parameters.discount_rate
        lt = zen_model.parameters.depreciation_time

        if dr != 0:
            a = ((1 + dr) ** lt * dr) / ((1 + dr) ** lt - 1)
        else:
            a = 1 / lt

        lt_range = pd.MultiIndex.from_tuples(
            [
                (t, y, py)
                for t, y in index.get_unique(
                    ["set_technologies", "set_time_steps_yearly"]
                )
                for py in list(
                    self.get_lifetime_range(t, y, zen_model, use_depreciation_time=True)
                )
            ]
        )

        lt_range = pd.Series(index=lt_range, data=-1)
        lt_range.index.names = [
            "set_technologies",
            "set_time_steps_yearly",
            "set_time_steps_yearly_prev",
        ]
        lt_range = (
            lt_range.to_xarray()
            .broadcast_like(zen_model.lp_model.variables["capacity"].lower)
            .fillna(0)
        )

        cost_capex_overnight = zen_model.lp_model.variables[
            "cost_capex_overnight"
        ].rename({"set_time_steps_yearly": "set_time_steps_yearly_prev"})
        cost_capex_overnight = cost_capex_overnight.broadcast_like(lt_range)
        expr = (lt_range * a * cost_capex_overnight).sum("set_time_steps_yearly_prev")
        lhs = lp.merge(
            [1 * zen_model.lp_model.variables["cost_capex_yearly"], expr],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = (a * zen_model.parameters.existing_capex).broadcast_like(lhs.const)
        constraints = lhs == rhs

        ### return
        zen_model.constraints.add_constraint(
            "constraint_cost_capex_yearly", constraints
        )

    def constraint_cost_opex_yearly(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Yearly opex for a technology at a location in each year.

        .. math::
            OPEX_{h,p,y} = \\sum_{t\\in\\mathcal{T}}\\tau_t O_{h,p,t}^t
            + \\gamma_{h,y} S_{h,p,y} + \\gamma_{k,y}^\\mathrm{e} S_{k,n,y}^\\mathrm{e}

        :math:`OPEX_{h,p,y}`: opex of operating technology :math:`h` at
        location :math:`p` in year :math:`y` \n
        :math:`\\tau_t`: duration of time step :math:`t` \n
        :math:`O_{h,p,t}^t`: variable opex of operating technology :math:`h` at
        location :math:`p` in time step :math:`t` \n
        :math:`\\gamma_{h,y}`: specific fixed opex of technology :math:`h` in
        year :math:`y` \n
        :math:`S_{h,p,y}`: installed capacity of technology :math:`h` at
        location :math:`p` in year :math:`y` \n
        :math:`\\gamma_{k,y}^\\mathrm{e}`: specific fixed opex of storage
        technology :math:`k` in year :math:`y` \n
        :math:`S_{k,n,y}^\\mathrm{e}`: installed capacity of storage
        technology :math:`k` at node :math:`n` in year :math:`y`

        """
        times_dict: dict[str, pd.Series] = {
            y: zen_model.parameters.time_steps_operation_duration.loc[
                energy_system.time_steps.get_time_steps_year2operation(y)
            ].to_series()
            for y in zen_model.sets["set_time_steps_yearly"]
        }
        times = pd.concat(times_dict, keys=times_dict.keys())
        times.index.names = ["set_time_steps_yearly", "set_time_steps_operation"]
        times = times.to_xarray().broadcast_like(
            zen_model.lp_model.variables["cost_opex_variable"].mask
        )
        term_opex_variable = (
            zen_model.lp_model.variables["cost_opex_variable"] * times
        ).sum("set_time_steps_operation")
        term_opex_fixed = (
            zen_model.parameters.opex_specific_fixed
            * zen_model.lp_model.variables["capacity"]
        ).sum("set_capacity_types")
        lhs = (
            zen_model.lp_model.variables["cost_opex_yearly"]
            - term_opex_variable
            - term_opex_fixed
        )
        rhs = 0
        constraints = lhs == rhs

        ### return
        zen_model.constraints.add_constraint("constraint_cost_opex_yearly", constraints)

    def constraint_carbon_emissions_technology_total(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Calculate total carbon emissions of each technology.

        .. math::
            E_y^{\\mathcal{H}} = \\sum_{p\\in\\mathcal{P}}
            \\sum_{t\\in\\mathcal{T}}\\sum_{h\\in\\mathcal{H}} \\theta_{h,p,t} \\tau_{t}

        :math:`E_y^{\\mathcal{H}}`: total carbon emissions of each technology in
        year :math:`y` \n
        :math:`\\theta_{h,p,t}`: carbon emissions of technology :math:`h` at
        location :math:`p` in time step :math:`t` \n
        :math:`\\tau_{t}`: duration of time step :math:`t`

        """
        term_summed_carbon_emissions_technology = (
            zen_model.lp_model.variables["carbon_emissions_technology"]
            * self.get_year_time_step_duration_array(zen_model, energy_system)
        ).sum(["set_technologies", "set_location", "set_time_steps_operation"])
        lhs = (
            zen_model.lp_model.variables["carbon_emissions_technology_total"]
            - term_summed_carbon_emissions_technology
        )
        rhs = 0
        constraints = lhs == rhs

        zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_technology_total", constraints
        )

    def constraint_technology_on_off(
        self, zen_model: ZenModel, energy_system: "EnergySystem", techs_on_off
    ):
        """If technology is on, the binary variable is 1, else 0.

        The min load constraint is expressed as six constraints
        (here for conversion technologies):

        .. math::
             m^\\mathrm{min}_{i,n,t}S^\\mathrm{approx}_{i,n,t}\\leq
             G^\\mathrm{r}_{i,n,t} \\leq S^\\mathrm{approx}_{i,n,t} \n
             0 \\leq S^\\mathrm{approx}_{i,n,t}
             \\leq s^\\mathrm{max}_{i,n,y} B_{i,n,t} \n
             S_{i,n,y} - s^\\mathrm{max}_{i,n,y}(1-B_{i,n,t})
             \\leq S^\\mathrm{approx}_{i,n,t} \\leq S_{i,n,y}

        :math:`m^\\mathrm{min}_{i,n,t}`: minimum load parameter for
        technology :math:`i`, node :math:`n`, time step :math:`t` \n
        :math:`G_{i,n,t}^\\mathrm{r}`: reference carrier flow of the
        technology :math:`i` at node :math:`n` in time step :math:`t` \n
        :math:`S_{h,p,y}`: installed capacity of technology :math:`h` at
        location :math:`p` in year :math:`y` \n
        :math:`B_{i,n,t}`: binary variable indicating whether the technology is on or
        off for technology :math:`i`, node :math:`n`, time step :math:`t` \n
        :math:`S^\\mathrm{approx}_{i,n,t}`: helper variable that represents the product
        of :math:`S_{i,n,y}` and :math:`B_{i,n,t}` \n
        :math:`s^\\mathrm{max}_{i,n,y}`: Big-M limit on :math:`S_{h,p,y}`
        """
        # sets
        conversion_techs = zen_model.sets["set_conversion_technologies"]
        storage_techs = zen_model.sets["set_storage_technologies"]
        transport_techs = zen_model.sets["set_transport_technologies"]
        nodes = zen_model.sets["set_nodes"]
        times = zen_model.sets["set_time_steps_operation"]
        ts = energy_system.time_steps
        time_step_year = xr.DataArray(
            [ts.convert_time_step_operation2year(t) for t in times.data],
            coords=[times],
            dims=["set_time_steps_operation"],
        )
        if len(techs_on_off) == 0:
            return None
        # params and variables
        min_load = zen_model.parameters.min_load
        capacity = zen_model.lp_model.variables["capacity"].sel(
            {"set_capacity_types": "power", "set_time_steps_yearly": time_step_year}
        )
        big_M = capacity.upper
        binary = zen_model.lp_model.variables["tech_on_var"]
        capacity_on_off_helper = zen_model.lp_model.variables[
            "capacity_on_off_helper_var"
        ]
        # mask for on_off variables
        mask_on_off = binary.mask
        # assert that no big-M is inf
        sel_big_M = (big_M.where(mask_on_off) == np.inf).to_series()
        big_M_elements = sel_big_M[sel_big_M].index.droplevel(2).unique().to_list()
        assert ~sel_big_M.any(), (
            f"Big-M is inf for {big_M_elements}. "
            f"Please set finite capacity limits of the technologies."
        )
        # flows
        list_flow_reference = []
        if len(conversion_techs) > 0:
            list_flow_reference.append(
                self.get_flow_expression_conversion(
                    conversion_techs, nodes, zen_model
                ).rename(
                    {
                        "set_conversion_technologies": "set_technologies",
                        "set_nodes": "set_location",
                    }
                )
            )
        if len(storage_techs) > 0:
            list_flow_reference.append(
                self.get_flow_expression_storage(zen_model, rename=True)
            )
        if len(transport_techs) > 0:
            list_flow_reference.append(
                zen_model.lp_model.variables["flow_transport"]
                .rename(
                    {
                        "set_transport_technologies": "set_technologies",
                        "set_edges": "set_location",
                    }
                )
                .to_linexpr()
            )
        flow_reference = lp.merge(
            list_flow_reference,
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        flow_reference = flow_reference.reindex_like(mask_on_off)
        # constraints
        # constraint 1, operational limit
        # 1a, lower bound
        lhs_1a = self.align_and_mask(
            min_load * capacity_on_off_helper - flow_reference, mask_on_off
        )
        rhs_1a = 0
        constraints_1a = lhs_1a <= rhs_1a
        zen_model.constraints.add_constraint(
            "constraint_technology_on_off_operation_lower_bound", constraints_1a
        )
        # 1a, upper bound
        lhs_1b = self.align_and_mask(
            -capacity_on_off_helper + flow_reference, mask_on_off
        )
        rhs_1b = 0
        constraints_1b = lhs_1b <= rhs_1b
        zen_model.constraints.add_constraint(
            "constraint_technology_on_off_operation_upper_bound", constraints_1b
        )
        # constraint 2, limit capacity helper
        # (lower bound already given by variable definition)
        lhs_2 = self.align_and_mask(
            capacity_on_off_helper - big_M * binary, mask_on_off
        )
        rhs_2 = 0
        constraints_2 = lhs_2 <= rhs_2
        zen_model.constraints.add_constraint(
            "constraint_technology_on_off_capacity_helper", constraints_2
        )
        # constraint 3, capacity helper bounds
        # 3a, lower bound
        lhs_3a = self.align_and_mask(
            capacity + big_M * binary - capacity_on_off_helper, mask_on_off
        )
        rhs_3a = big_M
        constraints_3a = lhs_3a <= rhs_3a
        zen_model.constraints.add_constraint(
            "constraint_technology_on_off_capacity_helper_lower_bound", constraints_3a
        )
        # 3b, upper bound
        lhs_3b = self.align_and_mask(capacity_on_off_helper - capacity, mask_on_off)
        rhs_3b = 0
        constraints_3b = lhs_3b <= rhs_3b
        zen_model.constraints.add_constraint(
            "constraint_technology_on_off_capacity_helper_upper_bound", constraints_3b
        )

    def get_lifetime_range(
        self,
        tech,
        year,
        zen_model: ZenModel,
        use_depreciation_time=False,
    ):
        """Get active year range of technology: either lifetime or depreciation time.

        :param optimization_setup: OptimizationSetup the technology is part of
        :param tech: name of the technology
        :param year: yearly time step
        :param use_depreciation_time: boolean indicating whether to use depreciation
            time instead of lifetime, namely for CAPEX calculation
        :return: lifetime or depreciation time range of technology
        """
        first_lifetime_year = self.get_first_lifetime_time_step(
            tech,
            year,
            zen_model,
            use_depreciation_time,
        )
        first_lifetime_year = max(
            first_lifetime_year, cast(int, zen_model.sets["set_time_steps_yearly"][0])
        )
        return range(first_lifetime_year, year + 1)

    def get_first_lifetime_time_step(
        self,
        tech,
        year,
        zen_model: ZenModel,
        use_depreciation_time=False,
    ):
        """Get first time step of active capacity of technology.

        Returns the first time step within the lifetime or depreciation time of the
        technology, i.e., the earliest past time step whose installed capacity is
        still active at the given time step.

        :param optimization_setup: OptimizationSetup the technology is part of
        :param tech: name of the technology
        :param year: current yearly time step
        :param use_depreciation_time: boolean indicating whether to use depreciation
            time instead of standard lifetime for capacity calculation
        :return: first time step where capacity or investment is still valid
        """
        # get params and system
        params = zen_model.parameters.dict_parameters
        lifetime = (
            params.depreciation_time[tech]
            if use_depreciation_time
            else params.lifetime[tech]
        )
        # conservative estimate of lifetime (floor)
        del_lifetime = (
            int(np.floor(lifetime / self.config.system.interval_between_years)) - 1
        )
        return year - del_lifetime
