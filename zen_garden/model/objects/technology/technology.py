"""
:Title:          ZEN-GARDEN
:Created:        October-2021
:Authors:        Alissa Ganter (aganter@ethz.ch),
                Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Class defining the parameters, variables and constraints that hold for all technologies.
The class takes the abstract optimization model as an input, and returns the parameters, variables and
constraints that hold for all technologies.
"""
import logging
import time

import linopy as lp
import numpy as np
import pandas as pd
import xarray as xr
from linopy.constraints import AnonymousConstraint

from zen_garden.utils import lp_sum
from ..component import ZenIndex, IndexSet
from ..element import Element, GenericRule


class Technology(Element):
    """
    Class defining the parameters, variables and constraints that hold for all technologies.
    """
    # set label
    label = "set_technologies"
    location_type = None

    def __init__(self, technology: str, optimization_setup):
        """init generic technology object

        :param technology: technology that is added to the model
        :param optimization_setup: The OptimizationSetup the element is part of """

        super().__init__(technology, optimization_setup)


    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # set attributes of technology
        _set_location = self.location_type
        _set_location_super = self.location_type_super

        set_base_time_steps_yearly = self.energy_system.set_base_time_steps_yearly
        set_time_steps_yearly = self.energy_system.set_time_steps_yearly
        self.reference_carrier = [self.data_input.extract_attribute("reference_carrier", skip_warning=True)]
        self.energy_system.set_technology_of_carrier(self.name, self.reference_carrier)
        self.capacity_addition_min = self.data_input.extract_attribute("capacity_addition_min")["value"]
        self.capacity_addition_max = self.data_input.extract_attribute("capacity_addition_max")["value"]
        self.capacity_addition_unbounded = self.data_input.extract_attribute("capacity_addition_unbounded")["value"]
        self.lifetime = self.data_input.extract_attribute("lifetime")["value"]
        self.construction_time = self.data_input.extract_attribute("construction_time")["value"]
        # maximum diffusion rate
        self.max_diffusion_rate = self.data_input.extract_input_data("max_diffusion_rate", index_sets=["set_time_steps_yearly"], time_steps=set_time_steps_yearly)

        # add all raw time series to dict
        self.raw_time_series = {}
        self.raw_time_series["min_load"] = self.data_input.extract_input_data("min_load", index_sets=[_set_location, "set_time_steps"], time_steps=set_base_time_steps_yearly)
        self.raw_time_series["max_load"] = self.data_input.extract_input_data("max_load", index_sets=[_set_location, "set_time_steps"], time_steps=set_base_time_steps_yearly)
        self.raw_time_series["opex_specific_variable"] = self.data_input.extract_input_data("opex_specific_variable", index_sets=[_set_location, "set_time_steps"], time_steps=set_base_time_steps_yearly)
        # non-time series input data
        self.opex_specific_fixed = self.data_input.extract_input_data("opex_specific_fixed", index_sets=[_set_location, "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        self.capacity_limit = self.data_input.extract_input_data("capacity_limit", index_sets=[_set_location])
        self.capacity_limit_super = self.data_input.extract_input_data("capacity_limit_super", index_sets=[_set_location_super])
        self.carbon_intensity_technology = self.data_input.extract_input_data("carbon_intensity", index_sets=[_set_location])
        # extract existing capacity
        self.set_technologies_existing = self.data_input.extract_set_technologies_existing()
        self.capacity_existing = self.data_input.extract_input_data("capacity_existing", index_sets=[_set_location, "set_technologies_existing"])
        self.capacity_investment_existing = self.data_input.extract_input_data("capacity_investment_existing", index_sets=[_set_location, "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        self.lifetime_existing = self.data_input.extract_lifetime_existing("capacity_existing", index_sets=[_set_location, "set_technologies_existing"])

    def calculate_capex_of_capacities_existing(self, storage_energy=False):
        """ this method calculates the annualized capex of the existing capacities

        :param storage_energy: #TODO describe parameter/return
        :return: #TODO describe parameter/return
        """
        if self.__class__.__name__ == "StorageTechnology":
            if storage_energy:
                capacities_existing = self.capacity_existing_energy
            else:
                capacities_existing = self.capacity_existing
            capex_capacity_existing = capacities_existing.to_frame().apply(
                lambda _capacity_existing: self.calculate_capex_of_single_capacity(_capacity_existing.squeeze(), _capacity_existing.name, storage_energy), axis=1)
        else:
            capacities_existing = self.capacity_existing
            capex_capacity_existing = capacities_existing.to_frame().apply(lambda _capacity_existing: self.calculate_capex_of_single_capacity(_capacity_existing.squeeze(), _capacity_existing.name), axis=1)
        return capex_capacity_existing

    def calculate_capex_of_single_capacity(self, *args):
        """ this method calculates the annualized capex of the existing capacities. Is implemented in child class

        :param args: #TODO describe parameter/return
        """
        raise NotImplementedError

    def calculate_fraction_of_year(self):
        """calculate fraction of year"""
        # only account for fraction of year
        _fraction_year = self.optimization_setup.system["unaggregated_time_steps_per_year"] / self.optimization_setup.system["total_hours_per_year"]
        return _fraction_year

    def overwrite_time_steps(self, base_time_steps: int):
        """ overwrites set_time_steps_operation

        :param base_time_steps: #TODO describe parameter/return
        """
        set_time_steps_operation = self.energy_system.time_steps.encode_time_step(self.name, base_time_steps=base_time_steps, time_step_type="operation", yearly=True)

        # copy invest time steps
        self.set_time_steps_operation = set_time_steps_operation.squeeze().tolist()

    def add_new_capacity_addition_tech(self, capacity_addition: pd.Series, capex: pd.Series, base_time_steps: int):
        """ adds the newly built capacity to the existing capacity

        :param capacity_addition: pd.Series of newly built capacity of technology
        :param capex: pd.Series of capex of newly built capacity of technology
        :param base_time_steps: base time steps of current horizon step """
        system = self.optimization_setup.system
        # reduce lifetime of existing capacities and add new remaining lifetime
        self.lifetime_existing = (self.lifetime_existing - system["interval_between_years"]).clip(lower=0)
        # new capacity
        _time_step_years = self.energy_system.time_steps.encode_time_step(self.name, base_time_steps, "yearly", yearly=True)
        _new_capacity_addition = capacity_addition[_time_step_years].sum(axis=1)
        _capex = capex[_time_step_years].sum(axis=1)
        # if at least one value unequal to zero
        if not (_new_capacity_addition == 0).all():
            # add new index to set_technologies_existing
            index_new_technology = max(self.set_technologies_existing) + 1
            self.set_technologies_existing = np.append(self.set_technologies_existing, index_new_technology)
            # add new remaining lifetime
            _lifetime = self.lifetime_existing.unstack()
            _lifetime[index_new_technology] = self.lifetime
            self.lifetime_existing = _lifetime.stack()

            for type_capacity in list(set(_new_capacity_addition.index.get_level_values(0))):
                # if power
                if type_capacity == system["set_capacity_types"][0]:
                    _energy_string = ""
                # if energy
                else:
                    _energy_string = "_energy"
                _capacity_existing = getattr(self, "capacity_existing" + _energy_string)
                _capex_capacity_existing = getattr(self, "capex_capacity_existing" + _energy_string)
                # add new existing capacity
                _capacity_existing = _capacity_existing.unstack()
                _capacity_existing[index_new_technology] = _new_capacity_addition.loc[type_capacity]
                setattr(self, "capacity_existing" + _energy_string, _capacity_existing.stack())
                # calculate capex of existing capacity
                _capex_capacity_existing = _capex_capacity_existing.unstack()
                _capex_capacity_existing[index_new_technology] = _capex.loc[type_capacity]
                setattr(self, "capex_capacity_existing" + _energy_string, _capex_capacity_existing.stack())

    def add_new_capacity_investment(self, capacity_investment: pd.Series, step_horizon):
        """ adds the newly invested capacity to the list of invested capacity

        :param capacity_investment: pd.Series of newly built capacity of technology
        :param step_horizon: optimization time step """
        system = self.optimization_setup.system
        _new_capacity_investment = capacity_investment[step_horizon]
        _new_capacity_investment = _new_capacity_investment.fillna(0)
        if not (_new_capacity_investment == 0).all():
            for type_capacity in list(set(_new_capacity_investment.index.get_level_values(0))):
                # if power
                if type_capacity == system["set_capacity_types"][0]:
                    _energy_string = ""
                # if energy
                else:
                    _energy_string = "_energy"
                _capacity_investment_existing = getattr(self, "capacity_investment_existing" + _energy_string)
                # add new existing invested capacity
                _capacity_investment_existing = _capacity_investment_existing.unstack()
                _capacity_investment_existing[step_horizon] = _new_capacity_investment.loc[type_capacity]
                setattr(self, "capacity_investment_existing" + _energy_string, _capacity_investment_existing.stack())

    ### --- classmethods
    @classmethod
    def get_lifetime_range(cls, optimization_setup, tech, time, time_step_type: str = None):
        """ returns lifetime range of technology. If time_step_type, then converts the yearly time step 'time' to time_step_type

        :param optimization_setup: OptimizationSetup the technology is part of
        :param tech: name of the technology
        :param time: #TODO describe parameter/return
        :param time_step_type: #TODO describe parameter/return
        :return: lifetime range of technology
        """
        if time_step_type:
            time_step_year = optimization_setup.energy_system.time_steps.convert_time_step_operation2year(tech,time)
        else:
            time_step_year = time
        t_start, t_end = cls.get_start_end_time_of_period(optimization_setup, tech, time_step_year)

        return range(t_start, t_end + 1)

    @classmethod
    def get_available_existing_quantity(cls, optimization_setup, tech, capacity_type, loc, time, type_existing_quantity, time_step_type: str = None):
        """ returns existing quantity of 'tech', that is still available at invest time step 'time'.
        Either capacity or capex.

        :param optimization_setup: The OptimizationSetup the element is part of
        :param tech: name of technology
        :param capacity_type: type of capacity
        :param loc: location (node or edge) of existing capacity
        :param time: current time
        :param type_existing_quantity: capex or capacity
        :param time_step_type: type of time steps
        :return existing_quantity: existing capacity or capex of existing capacity
        """
        params = optimization_setup.parameters.dict_parameters
        system = optimization_setup.system
        discount_rate = optimization_setup.analysis["discount_rate"]
        if time_step_type:
            time_step_year = optimization_setup.energy_system.time_steps.convert_time_step_operation2year(tech,time)
        else:
            time_step_year = time

        sets = optimization_setup.sets
        existing_quantity = 0
        if type_existing_quantity == "capacity":
            existing_variable = params.capacity_existing
        elif type_existing_quantity == "cost_capex":
            existing_variable = params.capex_capacity_existing
        else:
            raise KeyError(f"Wrong type of existing quantity {type_existing_quantity}")

        for id_capacity_existing in sets["set_technologies_existing"][tech]:
            t_start = cls.get_start_end_time_of_period(optimization_setup, tech, time_step_year, id_capacity_existing=id_capacity_existing, loc=loc)
            # if still available at first base time step, add to list
            if t_start == sets["set_base_time_steps"][0] or t_start == time_step_year:
                existing_quantity += existing_variable[tech, capacity_type, loc, id_capacity_existing]
        return existing_quantity

    @classmethod
    def get_start_end_time_of_period(cls, optimization_setup, tech, time_step_year, period_type="lifetime", clip_to_first_time_step=True, id_capacity_existing=None, loc=None):
        """ counts back the period (either lifetime of construction_time) back to get the start invest time step and returns start_time_step_year

        :param energy_system: The Energy system to add everything
        :param tech: name of technology
        :param time_step_year: current investment time step
        :param period_type: "lifetime" if lifetime is counted backwards, "construction_time" if construction time is counted backwards
        :param clip_to_first_time_step: boolean to clip the time step to first time step if time step too far in the past
        :param id_capacity_existing: id of existing capacity
        :param loc: location (node or edge) of existing capacity
        :return beganInPast: boolean if the period began before the first optimization step
        :return start_time_step_year,end_time_step_year: start and end of period in invest time step domain"""

        # get model and system
        energy_system = optimization_setup.energy_system
        params = optimization_setup.parameters.dict_parameters
        sets = optimization_setup.sets
        system = optimization_setup.system
        # get which period to count backwards
        if period_type == "lifetime":
            period_time = params.lifetime
        elif period_type == "construction_time":
            period_time = params.construction_time
        else:
            raise NotImplemented(f"get_start_end_time_of_period not yet implemented for {period_type}")
        # get end_time_step_year
        if not isinstance(time_step_year, np.ndarray):
            end_time_step_year = time_step_year
        elif len(time_step_year) == 1:
            end_time_step_year = time_step_year[0]
        # if more than one investment time step
        else:
            end_time_step_year = time_step_year[-1]
            time_step_year = time_step_year[0]
        # convert period to interval of base time steps
        if id_capacity_existing is None:
            period_yearly = period_time[tech]
        else:
            delta_lifetime = params.lifetime_existing[tech, loc, id_capacity_existing] - period_time[tech]
            if delta_lifetime >= 0:
                if delta_lifetime <= (time_step_year - sets["set_time_steps_yearly"][0]) * system["interval_between_years"]:
                    return time_step_year
                else:
                    return -1
            period_yearly = params.lifetime_existing[tech, loc, id_capacity_existing]
        base_period = period_yearly / system["interval_between_years"] * system["unaggregated_time_steps_per_year"]
        base_period = round(base_period, optimization_setup.solver["rounding_decimal_points"])
        if int(base_period) != base_period:
            logging.warning(f"The period {period_type} of {tech} does not translate to an integer time interval in the base time domain ({base_period})")
        # decode to base time steps
        base_time_steps = energy_system.time_steps.decode_time_step(tech, time_step_year, time_step_type="yearly")
        if len(base_time_steps) == 0:
            return sets["set_base_time_steps"][0], sets["set_base_time_steps"][0] - 1
        base_time_step = base_time_steps[0]

        # if start_base_time_step is further in the past than first base time step, use first base time step
        if clip_to_first_time_step:
            start_base_time_step = int(max(sets["set_base_time_steps"][0], base_time_step - base_period + 1))
        else:
            start_base_time_step = int(base_time_step - base_period + 1)
        start_base_time_step = min(start_base_time_step, sets["set_base_time_steps"][-1])
        # if period of existing capacity, then only return the start base time step
        if id_capacity_existing is not None:
            return start_base_time_step
        start_time_step_year = energy_system.time_steps.encode_time_step(tech, start_base_time_step, time_step_type="yearly", yearly=True)[0]

        return start_time_step_year, end_time_step_year

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Technology --- ###
    @classmethod
    def construct_sets(cls, optimization_setup):
        """ constructs the pe.Sets of the class <Technology>

        :param optimization_setup: The OptimizationSetup the element is part of """
        # construct the pe.Sets of the class <Technology>
        energy_system = optimization_setup.energy_system

        # conversion technologies
        optimization_setup.sets.add_set(name="set_conversion_technologies", data=energy_system.set_conversion_technologies,
                                        doc="Set of conversion technologies. Subset: set_technologies")
        # transport technologies
        optimization_setup.sets.add_set(name="set_transport_technologies", data=energy_system.set_transport_technologies,
                                        doc="Set of transport technologies. Subset: set_technologies")
        # storage technologies
        optimization_setup.sets.add_set(name="set_storage_technologies", data=energy_system.set_storage_technologies,
                                        doc="Set of storage technologies. Subset: set_technologies")
        # existing installed technologies
        optimization_setup.sets.add_set(name="set_technologies_existing", data=optimization_setup.get_attribute_of_all_elements(cls, "set_technologies_existing"),
                                        doc="Set of existing technologies. Subset: set_technologies",
                                        index_set="set_technologies")
        # reference carriers
        optimization_setup.sets.add_set(name="set_reference_carriers", data=optimization_setup.get_attribute_of_all_elements(cls, "reference_carrier"),
                                        doc="set of all reference carriers correspondent to a technology. Dimensions: set_technologies",
                                        index_set="set_technologies")
        # add pe.Sets of the child classes
        for subclass in cls.__subclasses__():
            subclass.construct_sets(optimization_setup)

    @classmethod
    def construct_params(cls, optimization_setup):
        """ constructs the pe.Params of the class <Technology>

        :param optimization_setup: The OptimizationSetup the element is part of """
        # construct pe.Param of the class <Technology>

        # existing capacity
        optimization_setup.parameters.add_parameter(name="capacity_existing",
            data=optimization_setup.initialize_component(cls, "capacity_existing", index_names=["set_technologies", "set_capacity_types", "set_location", "set_technologies_existing"], capacity_types=True),
            doc='Parameter which specifies the existing technology size')
        # existing capacity
        optimization_setup.parameters.add_parameter(name="capacity_investment_existing",
            data=optimization_setup.initialize_component(cls, "capacity_investment_existing", index_names=["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly_entire_horizon"],
                                                   capacity_types=True), doc='Parameter which specifies the size of the previously invested capacities')
        # minimum capacity addition
        optimization_setup.parameters.add_parameter(name="capacity_addition_min",
            data=optimization_setup.initialize_component(cls, "capacity_addition_min", index_names=["set_technologies", "set_capacity_types"], capacity_types=True),
            doc='Parameter which specifies the minimum capacity addition that can be installed')
        # maximum capacity addition
        optimization_setup.parameters.add_parameter(name="capacity_addition_max",
            data=optimization_setup.initialize_component(cls, "capacity_addition_max", index_names=["set_technologies", "set_capacity_types"], capacity_types=True),
            doc='Parameter which specifies the maximum capacity addition that can be installed')
        # unbounded capacity addition
        optimization_setup.parameters.add_parameter(name="capacity_addition_unbounded",
            data=optimization_setup.initialize_component(cls, "capacity_addition_unbounded", index_names=["set_technologies"]),
            doc='Parameter which specifies the unbounded capacity addition that can be added each year (only for delayed technology deployment)')
        # lifetime existing technologies
        optimization_setup.parameters.add_parameter(name="lifetime_existing",
            data=optimization_setup.initialize_component(cls, "lifetime_existing", index_names=["set_technologies", "set_location", "set_technologies_existing"]),
            doc='Parameter which specifies the remaining lifetime of an existing technology')
        # lifetime existing technologies
        optimization_setup.parameters.add_parameter(name="capex_capacity_existing",
            data=optimization_setup.initialize_component(cls, "capex_capacity_existing", index_names=["set_technologies", "set_capacity_types", "set_location", "set_technologies_existing"],
                                                   capacity_types=True), doc='Parameter which specifies the total capex of an existing technology which still has to be paid')
        # variable specific opex
        optimization_setup.parameters.add_parameter(name="opex_specific_variable",
            data=optimization_setup.initialize_component(cls, "opex_specific_variable",index_names=["set_technologies","set_location","set_time_steps_operation"]),
            doc='Parameter which specifies the variable specific opex')
        # fixed specific opex
        optimization_setup.parameters.add_parameter(name="opex_specific_fixed",
            data=optimization_setup.initialize_component(cls, "opex_specific_fixed",index_names=["set_technologies", "set_capacity_types","set_location","set_time_steps_yearly"], capacity_types=True),
            doc='Parameter which specifies the fixed annual specific opex')
        # lifetime newly built technologies
        optimization_setup.parameters.add_parameter(name="lifetime", data=optimization_setup.initialize_component(cls, "lifetime", index_names=["set_technologies"]),
            doc='Parameter which specifies the lifetime of a newly built technology')
        # construction_time newly built technologies
        optimization_setup.parameters.add_parameter(name="construction_time", data=optimization_setup.initialize_component(cls, "construction_time", index_names=["set_technologies"]),
            doc='Parameter which specifies the construction time of a newly built technology')
        # maximum diffusion rate, i.e., increase in capacity
        optimization_setup.parameters.add_parameter(name="max_diffusion_rate", data=optimization_setup.initialize_component(cls, "max_diffusion_rate", index_names=["set_technologies", "set_time_steps_yearly"]),
            doc="Parameter which specifies the maximum diffusion rate which is the maximum increase in capacity between investment steps")
        # capacity_limit of technologies
        optimization_setup.parameters.add_parameter(name="capacity_limit",
            data=optimization_setup.initialize_component(cls, "capacity_limit", index_names=["set_technologies", "set_capacity_types", "set_location"], capacity_types=True),
            doc='Parameter which specifies the capacity limit of technologies')
        # capacity_limit of technologies
        optimization_setup.parameters.add_parameter(name="capacity_limit_super",
            data=optimization_setup.initialize_component(cls, "capacity_limit_super", index_names=["set_technologies", "set_capacity_types", "set_super_location"], capacity_types=True),
            doc='Parameter which specifies the capacity limit of technologies for the super locations')
        # minimum load relative to capacity
        optimization_setup.parameters.add_parameter(name="min_load",
            data=optimization_setup.initialize_component(cls, "min_load", index_names=["set_technologies", "set_capacity_types", "set_location", "set_time_steps_operation"], capacity_types=True),
            doc='Parameter which specifies the minimum load of technology relative to installed capacity')
        # maximum load relative to capacity
        optimization_setup.parameters.add_parameter(name="max_load",
            data=optimization_setup.initialize_component(cls, "max_load", index_names=["set_technologies", "set_capacity_types", "set_location", "set_time_steps_operation"], capacity_types=True),
            doc='Parameter which specifies the maximum load of technology relative to installed capacity')
        # carbon intensity
        optimization_setup.parameters.add_parameter(name="carbon_intensity_technology", data=optimization_setup.initialize_component(cls, "carbon_intensity_technology", index_names=["set_technologies", "set_location"]),
            doc='Parameter which specifies the carbon intensity of each technology')

        # Helper params
        t0 = time.perf_counter()
        optimization_setup.parameters.add_helper_parameter(name="existing_capacities", data=cls.get_existing_quantity(optimization_setup, type_existing_quantity="capacity"))
        optimization_setup.parameters.add_helper_parameter(name="existing_capex", data=cls.get_existing_quantity(optimization_setup, type_existing_quantity="cost_capex", time_step_type="yearly"))
        t1 = time.perf_counter()
        logging.debug(f"Helper Params took {t1 - t0:.4f} seconds")

        # add pe.Param of the child classes
        for subclass in cls.__subclasses__():
            subclass.construct_params(optimization_setup)

    @classmethod
    def construct_vars(cls, optimization_setup):
        """ constructs the pe.Vars of the class <Technology>

        :param optimization_setup: The OptimizationSetup the element is part of """

        model = optimization_setup.model
        variables = optimization_setup.variables
        sets = optimization_setup.sets

        def capacity_bounds(tech, capacity_type, loc, time):
            """ 
            # TODO: This could be vectorized
            return bounds of capacity for bigM expression
            :param tech: tech index
            :param capacity_type: either power or energy
            :param loc: location of capacity
            :param time: investment time step
            :return bounds: bounds of capacity"""
            # bounds only needed for Big-M formulation, thus if any technology is modeled with on-off behavior
            if tech in techs_on_off:
                system = optimization_setup.system
                params = optimization_setup.parameters.dict_parameters
                if capacity_type == system["set_capacity_types"][0]:
                    _energy_string = ""
                else:
                    _energy_string = "_energy"
                _capacity_existing = getattr(params, "capacity_existing" + _energy_string)
                _capacity_addition_max = getattr(params, "capacity_addition_max" + _energy_string)
                _capacity_limit = getattr(params, "capacity_limit" + _energy_string)
                capacities_existing = 0
                for id_technology_existing in sets["set_technologies_existing"][tech]:
                    if params.lifetime_existing[tech, loc, id_technology_existing] > params.lifetime[tech]:
                        if time > params.lifetime_existing[tech, loc, id_technology_existing] - params.lifetime[tech]:
                            capacities_existing += _capacity_existing[tech, capacity_type, loc, id_technology_existing]
                    elif time <= params.lifetime_existing[tech, loc, id_technology_existing] + 1:
                        capacities_existing += _capacity_existing[tech, capacity_type, loc, id_technology_existing]

                capacity_addition_max = len(sets["set_time_steps_yearly"]) * _capacity_addition_max[tech, capacity_type]
                max_capacity_limit = _capacity_limit[tech, capacity_type, loc]
                bound_capacity = min(capacity_addition_max + capacities_existing, max_capacity_limit + capacities_existing)
                return 0, bound_capacity
            else:
                return 0, np.inf

        # bounds only needed for Big-M formulation, thus if any technology is modeled with on-off behavior
        techs_on_off = cls.create_custom_set(["set_technologies", "set_on_off"], optimization_setup)[0]
        # construct pe.Vars of the class <Technology>
        # capacity technology
        variables.add_variable(model, name="capacity", index_sets=cls.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup),
            bounds=capacity_bounds, doc='size of installed technology at location l and time t')
        # built_capacity technology
        variables.add_variable(model, name="capacity_addition", index_sets=cls.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup),
            bounds=(0,np.inf), doc='size of built technology (invested capacity after construction) at location l and time t')
        # invested_capacity technology
        variables.add_variable(model, name="capacity_investment", index_sets=cls.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup),
            bounds=(0,np.inf), doc='size of invested technology at location l and time t')
        # capex of building capacity
        variables.add_variable(model, name="cost_capex", index_sets=cls.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup),
            bounds=(0,np.inf), doc='capex for building technology at location l and time t')
        # annual capex of having capacity
        variables.add_variable(model, name="capex_yearly", index_sets=cls.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup),
            bounds=(0,np.inf), doc='annual capex for having technology at location l')
        # total capex
        variables.add_variable(model, name="cost_capex_total", index_sets=sets["set_time_steps_yearly"],
            bounds=(0,np.inf), doc='total capex for installing all technologies in all locations at all times')
        # opex
        variables.add_variable(model, name="cost_opex", index_sets=cls.create_custom_set(["set_technologies", "set_location", "set_time_steps_operation"], optimization_setup),
            bounds=(0,np.inf), doc="opex for operating technology at location l and time t")
        # total opex
        variables.add_variable(model, name="cost_opex_total", index_sets=sets["set_time_steps_yearly"],
            bounds=(0,np.inf), doc="total opex all technologies and locations in year y")
        # yearly opex
        variables.add_variable(model, name="opex_yearly", index_sets=cls.create_custom_set(["set_technologies", "set_location", "set_time_steps_yearly"], optimization_setup),
            bounds=(0,np.inf), doc="yearly opex for operating technology at location l and year y")
        # carbon emissions
        variables.add_variable(model, name="carbon_emissions_technology", index_sets=cls.create_custom_set(["set_technologies", "set_location", "set_time_steps_operation"], optimization_setup),
            doc="carbon emissions for operating technology at location l and time t")
        # total carbon emissions technology
        variables.add_variable(model, name="carbon_emissions_technology_total", index_sets=sets["set_time_steps_yearly"],
            doc="total carbon emissions for operating technology at location l and time t")

        # install technology
        # Note: binary variables are written into the lp file by linopy even if they are not relevant for the optimization,
        # which makes all problems MIPs. Therefore, we only add binary variables, if really necessary. Gurobi can handle this
        # by noting that the binary variables are not part of the model, however, only if there are no binary variables at all,
        # it is possible to get the dual values of the constraints.
        mask = cls._technology_installation_mask(optimization_setup)
        if mask.any():
            variables.add_variable(model, name="technology_installation", index_sets=cls.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup),
                                   binary=True, doc='installment of a technology at location l and time t', mask=mask)

        # add pe.Vars of the child classes
        for subclass in cls.__subclasses__():
            subclass.construct_vars(optimization_setup)

    @classmethod
    def construct_constraints(cls, optimization_setup):
        """ constructs the pe.Constraints of the class <Technology>

        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        constraints = optimization_setup.constraints
        sets = optimization_setup.sets
        # construct pe.Constraints of the class <Technology>
        rules = TechnologyRules(optimization_setup)
        #  technology capacity_limit
        constraints.add_constraint_block(model, name="constraint_technology_capacity_limit",
                                         constraint=rules.constraint_technology_capacity_limit_block(),
                                         doc='limited capacity of  technology depending on loc and time')
        #  technology capacity_limit super
        constraints.add_constraint_block(model, name="constraint_technology_capacity_limit_super",
                                         constraint=rules.constraint_technology_capacity_limit_super_block(),
                                         doc='limited capacity of technology depending on super loc and time')

        # minimum capacity
        constraints.add_constraint_block(model, name="constraint_technology_min_capacity",
                                         constraint=rules.constraint_technology_min_capacity_block(),
                                         doc='min capacity of technology that can be installed')
        # maximum capacity
        constraints.add_constraint_block(model, name="constraint_technology_max_capacity",
                                         constraint=rules.constraint_technology_max_capacity_block(),
                                         doc='max capacity of technology that can be installed')
        # construction period
        constraints.add_constraint_block(model, name="constraint_technology_construction_time",
                                         constraint=rules.constraint_technology_construction_time_block(),
                                         doc='lead time in which invested technology is constructed')
        # lifetime
        constraints.add_constraint_block(model, name="constraint_technology_lifetime",
                                         constraint=rules.constraint_technology_lifetime_block(),
                                         doc='max capacity of  technology that can be installed')
        # limit diffusion rate
        constraints.add_constraint_block(model, name="constraint_technology_diffusion_limit",
                                         constraint=rules.constraint_technology_diffusion_limit_block(),
                                         doc="Limits the newly built capacity by the existing knowledge stock")
        # limit diffusion rate total
        constraints.add_constraint_block(model, name="constraint_technology_diffusion_limit_total",
                                         constraint=rules.constraint_technology_diffusion_limit_total_block(),
                                         doc="Limits the newly built capacity by the existing knowledge stock for the entire energy system")
        # limit max load by installed capacity
        constraints.add_constraint_block(model, name="constraint_capacity_factor",
                                         constraint=rules.constraint_capacity_factor_block(),
                                         doc='limit max load by installed capacity')
        # annual capex of having capacity
        constraints.add_constraint_block(model, name="constraint_capex_yearly",
                                         constraint=rules.constraint_capex_yearly_block(),
                                         doc='annual capex of having capacity of technology.')
        # total capex of all technologies
        constraints.add_constraint_rule(model, name="constraint_cost_capex_total", index_sets=sets["set_time_steps_yearly"], rule=rules.constraint_cost_capex_total_rule,
            doc='total capex of all technology that can be installed.')
        # calculate opex
        constraints.add_constraint_block(model, name="constraint_opex_technology",
                                         constraint=rules.constraint_opex_technology_block(),
                                         doc="opex for each technology at each location and time step")
        # yearly opex
        constraints.add_constraint_block(model, name="constraint_opex_yearly",
                                         constraint=rules.constraint_opex_yearly_block(),
                                         doc='total opex of all technology that are operated.')
        # total opex of all technologies
        constraints.add_constraint_rule(model, name="constraint_cost_opex_total", index_sets=sets["set_time_steps_yearly"], rule=rules.constraint_cost_opex_total_rule, doc='total opex of all technology that are operated.')
        # carbon emissions of technologies
        constraints.add_constraint_block(model, name="constraint_carbon_emissions_technology",
                                         constraint=rules.constraint_carbon_emissions_technology_block(),
                                         doc="carbon emissions for each technology at each location and time step")
        # total carbon emissions of technologies
        constraints.add_constraint_block(model, name="constraint_carbon_emissions_technology_total", constraint=rules.constraint_carbon_emissions_technology_total_block(),
                                         doc="total carbon emissions for each technology at each location and time step")

        # disjunct if technology is on
        # the disjunction variables
        variables = optimization_setup.variables
        index_vals, _ = cls.create_custom_set(["set_technologies", "set_on_off", "set_capacity_types", "set_location", "set_time_steps_operation"], optimization_setup)
        index_names = ["on_off_technologies", "on_off_capacity_types", "on_off_locations", "on_off_time_steps_operation"]
        variables.add_variable(model, name="tech_on_var",
                               index_sets=(index_vals, index_names),
                               doc="The tech on var", binary=True)
        variables.add_variable(model, name="tech_off_var",
                               index_sets=(index_vals, index_names),
                               doc="The tech off var", binary=True)
        model.add_constraints(model.variables["tech_on_var"] + model.variables["tech_off_var"] == 1, name="tech_on_off_cons")
        n_cons = model.constraints.ncons

        constraints.add_constraint_rule(model, name="disjunct_on_technology",
            index_sets=cls.create_custom_set(["set_technologies", "set_on_off", "set_capacity_types", "set_location", "set_time_steps_operation"], optimization_setup), rule=rules.disjunct_on_technology_rule,
            doc="disjunct to indicate that technology is on")
        # disjunct if technology is off
        constraints.add_constraint_rule(model, name="disjunct_off_technology",
            index_sets=cls.create_custom_set(["set_technologies", "set_on_off", "set_capacity_types", "set_location", "set_time_steps_operation"], optimization_setup), rule=rules.disjunct_off_technology_rule,
            doc="disjunct to indicate that technology is off")

        # if nothing was added we can remove the tech vars again
        if model.constraints.ncons == n_cons:
            model.constraints.remove("tech_on_off_cons")
            model.variables.remove("tech_on_var")
            model.variables.remove("tech_off_var")

        # add pe.Constraints of the child classes
        for subclass in cls.__subclasses__():
            logging.info(f"Construct pe.Constraints of {subclass.__name__}")
            subclass.construct_constraints(optimization_setup)

    @classmethod
    def _technology_installation_mask(cls, optimization_setup):
        """check if the binary variable is necessary

        :param optimization_setup: optimization setup object"""
        params = optimization_setup.parameters
        model = optimization_setup.model
        sets = optimization_setup.sets

        mask = xr.DataArray(False, coords=[model.variables.coords["set_time_steps_yearly"],
                                           model.variables.coords["set_technologies"],
                                           model.variables.coords["set_capacity_types"],
                                           model.variables.coords["set_location"], ])

        # used in transport technology
        techs = list(sets["set_transport_technologies"])
        techs = list(sets["set_transport_technologies"])
        if len(techs) > 0:
            edges = list(sets["set_edges"])
            sub_mask = (params.distance.loc[techs, edges] * params.capex_per_distance_transport.loc[techs, edges] != 0)
            sub_mask = sub_mask.rename({"set_transport_technologies": "set_technologies", "set_edges": "set_location"})
            mask.loc[:, techs, :, edges] |= sub_mask

        # used in constraint_technology_min_capacity
        mask = mask | (params.capacity_addition_min.notnull() & (params.capacity_addition_min != 0))

        # used in constraint_technology_max_capacity
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup)
        index = ZenIndex(index_values, index_names)
        sub_mask = (params.capacity_addition_max.notnull() & (params.capacity_addition_max != np.inf) & (params.capacity_addition_max != 0))
        for tech, capacity_type in index.get_unique([0, 1]):
            locs = index.get_values(locs=[tech, capacity_type], levels=2, unique=True)
            mask.loc[:, tech, capacity_type, locs] |= sub_mask.loc[tech, capacity_type]

        return mask

    @classmethod
    def get_existing_quantity(cls, optimization_setup, type_existing_quantity, time_step_type=None):
        """
        get existing capacities of all technologies
        :return: The existing capacities
        """

        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup)
        # get all the capacities
        index_arrs = IndexSet.tuple_to_arr(index_values, index_names)
        coords = [optimization_setup.sets.get_coord(data, name) for data, name in zip(index_arrs, index_names)]
        existing_quantities = xr.DataArray(np.nan, coords=coords, dims=index_names)
        values = np.zeros(len(index_values))
        for i, (tech, capacity_type, loc, time) in enumerate(index_values):
            values[i] = Technology.get_available_existing_quantity(optimization_setup, tech, capacity_type, loc, time,
                                                                   type_existing_quantity=type_existing_quantity,
                                                                   time_step_type=time_step_type)
        existing_quantities.loc[index_arrs] = values
        return existing_quantities


class TechnologyRules(GenericRule):
    """
    Rules for the Technology class
    """

    def __init__(self, optimization_setup):
        """
        Inits the rules
        :param optimization_setup: OptimizationSetup of the element
        """

        super().__init__(optimization_setup)

    # Disjunctive Constraints
    # -----------------------

    def disjunct_on_technology_rule(self, tech, capacity_type, loc, time):
        """definition of disjunct constraints if technology is On
        iterate through all subclasses to find corresponding implementation of disjunct constraints

        :param tech: #TODO describe parameter/return
        :param capacity_type: #TODO describe parameter/return
        :param loc: #TODO describe parameter/return
        :param time:  #TODO describe parameter/return
        """
        for subclass in Technology.__subclasses__():
            if tech in self.optimization_setup.get_all_names_of_elements(subclass):
                # extract the relevant binary variable (not scalar, .loc is necessary)
                binary_var = self.optimization_setup.model.variables["tech_on_var"].loc[tech, capacity_type, loc, time]
                subclass.disjunct_on_technology_rule(self.optimization_setup, tech, capacity_type, loc, time, binary_var)
                return None

    def disjunct_off_technology_rule(self, tech, capacity_type, loc, time):
        """definition of disjunct constraints if technology is off
        iterate through all subclasses to find corresponding implementation of disjunct constraints

        :param disjunct: #TODO describe parameter/return
        :param tech: #TODO describe parameter/return
        :param capacity_type: #TODO describe parameter/return
        :param loc: #TODO describe parameter/return
        :param time: #TODO describe parameter/return
        """
        for subclass in Technology.__subclasses__():
            if tech in self.optimization_setup.get_all_names_of_elements(subclass):
                # extract the relevant binary variable (not scalar, .loc is necessary)
                binary_var = self.optimization_setup.model.variables["tech_off_var"].loc[tech, capacity_type, loc, time]
                subclass.disjunct_off_technology_rule(self.optimization_setup, tech, capacity_type, loc, time, binary_var)
                return None

    # Rule-based constraints
    # -----------------------

    def constraint_cost_capex_total_rule(self, year):
        """ sums over all technologies to calculate total capex

        .. math::
            CAPEX_y = \\sum_{h\\in\mathcal{H}}\\sum_{p\\in\mathcal{P}}A_{h,p,y}+\\sum_{k\\in\mathcal{K}}\\sum_{n\\in\mathcal{N}}A^\mathrm{e}_{k,n,y}

        :param year: #TODO describe parameter/return
        :return: #TODO describe parameter/return
        """

        ### index sets
        # skipped because rule-based constraint

        ### masks
        # skipped because rule-based constraint

        ### index loop
        # skipped because rule-based constraint

        ### auxiliary calculations
        term_sum_yearly = self.variables["capex_yearly"].loc[..., year].sum()

        ### formulate constraint
        lhs = (self.variables["cost_capex_total"].loc[year]
               - term_sum_yearly)
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)

    def constraint_cost_opex_total_rule(self, year):
        """ sums over all technologies to calculate total opex

        .. math::
            OPEX_y = \sum_{h\in\mathcal{H}}\sum_{p\in\mathcal{P}} OPEX_{h,p,y}

        :param year: #TODO describe parameter/return
        :return: #TODO describe parameter/return
        """

        ### index sets
        # skipped because rule-based constraint

        ### masks
        # skipped because rule-based constraint

        ### index loop
        # skipped because rule-based constraint

        ### auxiliary calculations
        term_sum_yearly = self.variables["opex_yearly"].loc[..., year].sum()

        ### formulate constraint
        lhs = (self.variables["cost_opex_total"].loc[year]
               - term_sum_yearly)
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)

    # Block-based constraints
    # -----------------------

    def constraint_technology_capacity_limit_block(self):
        """limited capacity_limit of technology

        .. math::
            \mathrm{if\ existing\ capacities\ < capacity\ limit}\ s^\mathrm{max}_{h,p} \geq S_{h,p,y}
        .. math::
            \mathrm{else}\ \Delta S_{h,p,y} = 0

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_names = ["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"]
        index_values, index_names = Element.create_custom_set(index_names, self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # create the masks for the two cases, m1 for capacity_limit not reached, m2 for capacity_limit reached
        m1 = (self.parameters.capacity_limit != np.inf) & (self.parameters.existing_capacities < self.parameters.capacity_limit)
        m2 = (self.parameters.capacity_limit != np.inf) & ~(self.parameters.existing_capacities < self.parameters.capacity_limit)

        ### index loop
        # not necessary

        ### auxiliary calculations
        # not necessary

        ### formulate constraint
        # Note that this constraint has a different sign for the two cases
        lhs = self.variables["capacity"].where(m1) + self.variables["capacity_addition"].where(m2)
        rhs = self.parameters.capacity_limit.where(m1, 0.0)
        constraints = lhs <= rhs

        ### return
        return self.constraints.return_contraints(constraints, model=self.model, index_values=index.get_unique(index_names), index_names=index_names)

    def constraint_technology_capacity_limit_super_block(self):
        """limited capacity_limit of technology for super locations

        .. math::
            \mathrm{if\ existing\ capacities\ < capacity\ limit}\ s^\mathrm{max}_{h,sp} \geq \sum_{p \in \script{SP}_sp} S_{h,p,y}
        .. math::
            \mathrm{else}\ 0 \geq \sum_{p \in \script{SP}_sp} \Delta S_{h,p,y}

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_names = ["set_technologies", "set_capacity_types", "set_super_location", "set_time_steps_yearly"]
        index_values, index_names = Element.create_custom_set(index_names, self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # used within index loop

        ### index loop
        constraints = []
        capacity_limit_super = self.parameters.capacity_limit_super
        for super_loc in index.get_unique(levels=["set_super_location"]):
            ## auxiliary calculations
            # determine the total capacity at each super location
            if super_loc in self.sets["set_super_nodes"]:
                set_loc_in_super_loc = self.sets["set_nodes_in_super_nodes"][super_loc]
            elif super_loc in self.sets["set_super_edges"]:
                set_loc_in_super_loc = self.sets["set_edges_in_super_edges"][super_loc]
            if not set_loc_in_super_loc:
                continue
            existing_capacities = sum(self.parameters.existing_capacities.loc[:, :, loc, :] for loc in set_loc_in_super_loc)
            if not (isinstance(existing_capacities, float) or isinstance(existing_capacities, int)):
                existing_capacities = existing_capacities.where(existing_capacities == np.nan, 0.0)
            # masks to formulate constraints
            m1 = (capacity_limit_super.loc[...,super_loc] != np.inf) & (existing_capacities < capacity_limit_super.loc[...,super_loc])
            m2 = (capacity_limit_super.loc[...,super_loc] != np.inf) & ~(existing_capacities < capacity_limit_super.loc[...,super_loc])
            ### formulate constraint
            lhs = lp.merge(self.variables["capacity"].loc[:,:,set_loc_in_super_loc,:].where(m1).to_linexpr(),
                           self.variables["capacity_addition"].loc[:,:,set_loc_in_super_loc,:].where(m2).to_linexpr(),
                           compat="broadcast_equals")
            #sum(self.variables["capacity"].loc[:,:,loc,:].where(m1) + self.variables["capacity_addition"].loc[:,:,loc,:].where(m2) for loc in set_loc_in_super_loc)
            rhs = capacity_limit_super.loc[...,super_loc].where(m1, 0.0)
            constraints.append(lhs<=rhs)

        ### return
        return self.constraints.return_contraints(constraints, model=self.model, index_values=index.get_unique(["set_super_location"]), index_names=["set_super_location"])

    def constraint_technology_min_capacity_block(self):
        """ min capacity expansion of technology

        .. math::
            s^\mathrm{add, min}_{h} B_{i,p,y} \le \Delta S_{h,p,y}

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)
        tech_arr, capacity_type_arr = index.get_unique(["set_technologies", "set_capacity_types"], as_array=True)

        ### masks
        # we create a mask here only to avoid having constraints with binary variables when it's not necessary
        # passing constraints with binary variables to gurobi, even of the type 0 * binray_var, means that no
        # dual variables are returned
        mask = xr.zeros_like(self.parameters.capacity_addition_min, dtype=bool)
        mask.loc[tech_arr, capacity_type_arr] = True
        mask &= self.parameters.capacity_addition_min != 0

        ### index loop
        # not necessary

        ### auxiliary calculations
        # if the mask is empty, we don't need to do anything and abort here
        if not mask.any():
            return []

        ### formulate constraint
        lhs = mask * (self.parameters.capacity_addition_min * self.variables["technology_installation"]
                      - self.variables["capacity_addition"])
        rhs = 0
        constraints = lhs <= rhs

        ### return
        return self.constraints.return_contraints(constraints, mask=mask)

    def constraint_technology_max_capacity_block(self):
        """max capacity expansion of technology

        .. math::
            s^\mathrm{add, max}_{h} B_{i,p,y} \ge \Delta S_{h,p,y}

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        constraints = []
        for tech, capacity_type in index.get_unique(["set_technologies", "set_capacity_types"]):
            # not that the else here is just a dummy
            if self.parameters.capacity_addition_max.loc[tech, capacity_type] != np.inf:

                ### auxiliary calculations
                # we only want a constraints with a binary variable if the corresponding max_built_capacity is not zero
                if np.any(self.parameters.capacity_addition_max.loc[tech, capacity_type].notnull() & (self.parameters.capacity_addition_max.loc[tech, capacity_type] != 0)):
                    term_installation = self.parameters.capacity_addition_max.loc[tech, capacity_type].item() * self.variables["technology_installation"].loc[tech, capacity_type]
                else:
                    # dummy
                    term_installation = self.variables["capacity_addition"].loc[tech, capacity_type].where(False)

                ### formulate constraint
                lhs = (- self.variables["capacity_addition"].loc[tech, capacity_type]
                       + term_installation)
                rhs = 0
                constraints.append(lhs >= rhs)

            else:
                # a dummy
                constraints.append(np.nan*self.variables["capacity_addition"].loc[tech, capacity_type].where(False) == np.nan)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_technologies", "set_capacity_types"]),
                                                  index_names=["set_technologies", "set_capacity_types"])

    def constraint_technology_construction_time_block(self):
        """ construction time of technology, i.e., time that passes between investment and availability

        .. math::
            \mathrm{if\ start\ time\ step\ in\ set\ time\ steps\ yearly}\ \Delta S_{h,p,y} = S_{h,p,y}^\mathrm{invest}
        .. math::
            \mathrm{elif\ start\ time\ step\ in\ set\ time\ steps\ yearly\ entire\ horizon}\ \Delta S_{h,p,y} = s^\mathrm{invest, exist}_{h,p,y}
        .. math::
            \mathrm{else}\ \Delta S_{h,p,y} = 0

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over technologies and time steps, because the conditions depend on the time and the technology
        # we vectorize over capacity types and locations
        constraints = []
        for tech, time in index.get_unique(["set_technologies", "set_time_steps_yearly"]):

            ### auxiliary calculations
            start_time_step, _ = Technology.get_start_end_time_of_period(self.optimization_setup, tech, time, period_type="construction_time", clip_to_first_time_step=False)

            ### formulate constraint
            if start_time_step in self.sets["set_time_steps_yearly"]:
                lhs = (self.variables["capacity_addition"].loc[tech, :, :, time]
                       - self.variables["capacity_investment"].loc[tech, :, :, start_time_step])
                rhs = 0
            elif start_time_step in self.sets["set_time_steps_yearly_entire_horizon"]:
                lhs = self.variables["capacity_addition"].loc[tech, :, :, time]
                rhs = self.parameters.existing_invested_capacity.loc[tech, :, :, start_time_step]
            else:
                lhs = self.variables["capacity_addition"].loc[tech, :, :, time]
                rhs = 0
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])

    def constraint_technology_lifetime_block(self):
        """ limited lifetime of the technologies

        .. math::
            S_{h,p,y} = \\sum_{\\tilde{y}=\\max(y_0,y-\\lceil\\frac{l_h}{\\Delta^\mathrm{y}}\\rceil+1)}^y \\Delta S_{h,p,\\tilde{y}}
            + \\sum_{\\hat{y}=\\psi(\\min(y_0-1,y-\\lceil\\frac{l_h}{\\Delta^\mathrm{y}}\\rceil+1))}^{\\psi(y_0)} \\Delta s^\mathrm{ex}_{h,p,\\hat{y}}

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # this mask is just to make sure we only get constraints where we want them, no if-condition
        mask = self.variables["capacity"].mask

        ### index loop
        # we loop over technologies and time steps, because we need to cycle over the lifetime range of the technology
        # which requires the technology and the year, we vectorize over capacity types and locations
        constraints = []
        for tech, time in index.get_unique(["set_technologies", "set_time_steps_yearly"]):

            ### auxiliary calculations
            term_neg_previous_capacity_additions = []
            for previous_time in Technology.get_lifetime_range(self.optimization_setup, tech, time):
                term_neg_previous_capacity_additions.append(-1.0 * self.variables["capacity_addition"].loc[tech, :, :, previous_time])

            ### formulate constraint
            lhs = lp_sum([1.0 * self.variables["capacity"].loc[tech, :, :, time],
                          *term_neg_previous_capacity_additions])
            rhs = self.parameters.existing_capacities.loc[tech, :, :, time]
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  mask=mask,
                                                  index_values=index.get_unique(["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])

    def constraint_technology_diffusion_limit_block(self):
        """limited technology diffusion based on the existing capacity in the previous year

        .. math::
                \\Delta S_{j,e,y}\\leq ((1+\\vartheta_j)^{\\Delta^\mathrm{y}}-1)K_{j,e,y}
                +\\Delta^\mathrm{y}(\\xi\\sum_{\\tilde{j}\\in\\tilde{\mathcal{J}}}S_{\\tilde{j},e,y} + \\zeta_j)

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over technologies, capacity types and time steps, to accurately capture the conditions in the constraint
        # we vectorize over locations
        constraints = []
        for tech, time in index.get_unique(["set_technologies", "set_time_steps_yearly"]):

            ### auxiliary calculations
            # mask for the capacity types that are not considered
            capacity_types = index.get_values([tech, slice(None), slice(None), time], "set_capacity_types", unique=True)
            mask = xr.DataArray(np.nan, coords=[self.variables.coords["set_capacity_types"]], dims=["set_capacity_types"])
            mask.loc[capacity_types] = 1

            interval_between_years = self.system["interval_between_years"]
            knowledge_depreciation_rate = self.system["knowledge_depreciation_rate"]
            reference_carrier = self.sets["set_reference_carriers"][tech][0]
            if self.parameters.max_diffusion_rate.loc[tech, time] != np.inf:
                if tech in self.sets["set_transport_technologies"]:
                    set_locations = self.sets["set_edges"]
                    set_technology = self.sets["set_transport_technologies"]
                    knowledge_spillover_rate = 0
                else:
                    set_locations = self.sets["set_nodes"]
                    knowledge_spillover_rate = self.parameters.knowledge_spillover_rate
                    if tech in self.sets["set_conversion_technologies"]:
                        set_technology = self.sets["set_conversion_technologies"]
                    else:
                        set_technology = self.sets["set_storage_technologies"]

                    # add capacity addition of entire previous horizon
                end_time = time - 1

                # actual years between first invest time step and end_time
                delta_time = interval_between_years * (end_time - self.sets["set_time_steps_yearly"][0])
                existing_time = self.sets["set_technologies_existing"][tech]
                # Note: instead of summing over all but one location, we sum over all and then subtract one
                total_capacity_knowledge_existing = ((self.parameters.capacity_existing.loc[tech, :, set_locations, existing_time]  # add spillover from other regions
                                                      + knowledge_spillover_rate * (self.parameters.capacity_existing.loc[tech, :, set_locations, existing_time].sum("set_location") - self.parameters.capacity_existing.loc[tech, :, set_locations, existing_time]))
                                                     * (1 - knowledge_depreciation_rate) ** (delta_time + self.parameters.lifetime.loc[tech].item() - self.parameters.lifetime_existing.loc[tech, set_locations, existing_time])).sum("set_technologies_existing")
                rounding_value = 10 ** (-self.optimization_setup.solver["rounding_decimal_points"])
                total_capacity_knowledge_existing = total_capacity_knowledge_existing.where(total_capacity_knowledge_existing > rounding_value, 0)

                horizon_time = self.variables.coords["set_time_steps_yearly"][self.sets["set_time_steps_yearly"][0]: end_time + 1]
                if len(horizon_time) > 1:
                    term_total_capacity_knowledge_addition = ((self.variables["capacity_addition"].loc[tech, :, set_locations, horizon_time]  # add spillover from other regions
                                                          + knowledge_spillover_rate * (self.variables["capacity_addition"].loc[tech, :, set_locations, horizon_time].sum("set_location") - self.variables["capacity_addition"].loc[tech, :, set_locations, horizon_time]))
                                                         * (1 - knowledge_depreciation_rate) ** (interval_between_years * (end_time - horizon_time))).sum("set_time_steps_yearly")
                else:
                    # dummy term
                    term_total_capacity_knowledge_addition = self.variables["capacity_investment"].loc[tech, :, set_locations, time].where(False)

                # total capacity in previous year; if time is first time step of interval, use existing capacities of present year
                other_techs = [other_tech for other_tech in set_technology if self.sets["set_reference_carriers"][other_tech][0] == reference_carrier]
                if time != self.optimization_setup.energy_system.set_time_steps_yearly[0]:
                    term_total_capacity_all_techs_var = self.variables["capacity"].loc[other_techs, :, set_locations, time-1].sum("set_technologies")
                    term_total_capacity_all_techs_param = self.parameters.existing_capacities.loc[tech,:,set_locations,time].where(False)
                else:
                    term_total_capacity_all_techs_param = self.parameters.existing_capacities.loc[other_techs,:,set_locations,time].sum("set_technologies")
                    term_total_capacity_all_techs_var = self.variables["capacity"].loc[tech, :, set_locations, time].where(False)

                ### formulate constraint
                # build the lhs
                lhs = (self.variables["capacity_investment"].loc[tech, :, set_locations, time]
                       + self.variables["capacity_investment"].loc[tech, :, :, time].where(False)
                       - ((1 + self.parameters.max_diffusion_rate.loc[tech, time].item()) ** interval_between_years - 1) * term_total_capacity_knowledge_addition
                       - self.parameters.market_share_unbounded * term_total_capacity_all_techs_var)
                lhs *= mask

                # build the rhs
                rhs = xr.zeros_like(self.parameters.existing_capacities).loc[tech, :,:, time]
                rhs.loc[:, set_locations] += ((1 + self.parameters.max_diffusion_rate.loc[tech, time].item()) ** interval_between_years - 1) * total_capacity_knowledge_existing # add initial market share until which the diffusion rate is unbounded
                rhs.loc[:, set_locations] += self.parameters.market_share_unbounded * term_total_capacity_all_techs_param + self.parameters.capacity_addition_unbounded.loc[tech]
                rhs *= mask

                # combine
                constraints.append(lhs <= rhs)

        ### return
        return self.constraints.return_contraints(constraints, model=self.model, stack_dim_name="diffusion_limit_dim")

    def constraint_technology_diffusion_limit_total_block(self):
        """limited technology diffusion based on the existing capacity in the previous year for the entire energy system

        .. math:: #TODO
                \\Delta S_{j,e,y}\\leq ((1+\\vartheta_j)^{\\Delta^\mathrm{y}}-1)K_{j,e,y}
                +\\Delta^\mathrm{y}(\\xi\\sum_{\\tilde{j}\\in\\tilde{\mathcal{J}}}S_{\\tilde{j},e,y} + \\zeta_j)

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over technologies, capacity types and time steps, to accurately capture the conditions in the constraint
        # we vectorize over locations
        constraints = []
        for tech, time in index.get_unique(["set_technologies", "set_time_steps_yearly"]):

            ### auxiliary calculations
            # mask for the capacity types that are not considered
            capacity_types = index.get_values([tech, slice(None), slice(None), time], "set_capacity_types", unique=True)
            mask = xr.DataArray(np.nan, coords=[self.variables.coords["set_capacity_types"]], dims=["set_capacity_types"])
            mask.loc[capacity_types] = 1

            interval_between_years = self.system["interval_between_years"]
            knowledge_depreciation_rate = self.system["knowledge_depreciation_rate"]

            reference_carrier = self.sets["set_reference_carriers"][tech][0]
            if self.parameters.max_diffusion_rate.loc[tech, time] != np.inf:
                if tech in self.sets["set_transport_technologies"]:
                    set_locations = self.sets["set_edges"]
                    set_technology = self.sets["set_transport_technologies"]
                else:
                    set_locations = self.sets["set_nodes"]
                    if tech in self.sets["set_conversion_technologies"]:
                        set_technology = self.sets["set_conversion_technologies"]
                    else:
                        set_technology = self.sets["set_storage_technologies"]

                # add capacity addition of entire previous horizon
                end_time = time - 1

                # actual years between first invest time step and end_time
                delta_time = interval_between_years * (end_time - self.sets["set_time_steps_yearly"][0])
                existing_time = self.sets["set_technologies_existing"][tech]
                term_total_capacity_knowledge_existing = (self.parameters.capacity_existing.loc[tech, :, set_locations, existing_time]
                                                     * (1 - knowledge_depreciation_rate) ** (delta_time + self.parameters.lifetime.loc[tech].item() - self.parameters.lifetime_existing.loc[tech, set_locations, existing_time])).sum(["set_technologies_existing","set_location"])
                _rounding_value = 10 ** (-self.optimization_setup.solver["rounding_decimal_points"])
                term_total_capacity_knowledge_existing = term_total_capacity_knowledge_existing.where(term_total_capacity_knowledge_existing > _rounding_value, 0)

                horizon_time = self.variables.coords["set_time_steps_yearly"][self.sets["set_time_steps_yearly"][0]: end_time + 1]
                if len(horizon_time) > 1:
                    term_total_capacity_knowledge_addition = (self.variables["capacity_addition"].loc[tech, :, set_locations, horizon_time]
                        * (1 - knowledge_depreciation_rate) ** (interval_between_years * (end_time - horizon_time))).sum(["set_time_steps_yearly","set_location"])
                else:
                    # dummy term
                    term_total_capacity_knowledge_addition = self.variables["capacity_investment"].loc[tech, :, set_locations, time].where(False).sum("set_location")

                # total capacity in previous year; if time is first time step of interval, use existing capacities of present year
                other_techs = [other_tech for other_tech in set_technology if self.sets["set_reference_carriers"][other_tech][0] == reference_carrier]
                if time != self.optimization_setup.energy_system.set_time_steps_yearly[0]:
                    term_total_capacity_all_techs_var = self.variables["capacity"].loc[other_techs, :,set_locations, time - 1].sum(["set_technologies","set_location"])
                    term_total_capacity_all_techs_param = self.parameters.existing_capacities.loc[tech, :,set_locations, time].where(False).sum("set_location")
                else:
                    term_total_capacity_all_techs_param = self.parameters.existing_capacities.loc[other_techs, :, set_locations, time].sum(["set_technologies","set_location"])
                    term_total_capacity_all_techs_var = self.variables["capacity"].loc[tech, :, set_locations,time].where(False).sum("set_location")
                # total_capacity_all_techs_param = sum(self.parameters.existing_capacities.loc[other_tech, :, set_locations, time]
                #                                      for other_tech in set_technology if self.sets["set_reference_carriers"][other_tech][0] == reference_carrier)
                # lifetime_range = Technology.get_lifetime_range(self.optimization_setup, tech, end_time)
                # if len(lifetime_range) > 0:
                #     previous_times = [previous_time for previous_time in lifetime_range]
                #     other_techs = [other_tech for other_tech in set_technology if self.sets["set_reference_carriers"][other_tech][0] == reference_carrier]
                #     term_total_capacity_all_techs_var = self.variables["capacity_addition"].loc[:, :, set_locations, previous_times].sum("set_time_steps_yearly")
                #     term_total_capacity_all_techs_var = term_total_capacity_all_techs_var[other_techs, :, :].sum("set_technologies")
                # else:
                #     # dummy term
                #     term_total_capacity_all_techs_var = self.variables["capacity_investment"].loc[tech, :, set_locations, time].where(False)
                # total_capacity_all_techs_param = sum(self.parameters.existing_capacities.loc[other_tech, :, set_locations, time]
                #                                      for other_tech in set_technology if self.sets["set_reference_carriers"][other_tech][0] == reference_carrier).sum("set_location")
                # lifetime_range = Technology.get_lifetime_range(self.optimization_setup, tech, end_time)
                # if len(lifetime_range) > 0:
                #     previous_times = [previous_time for previous_time in lifetime_range]
                #     other_techs = [other_tech for other_tech in set_technology if self.sets["set_reference_carriers"][other_tech][0] == reference_carrier]
                #     term_total_capacity_all_techs_var = self.variables["capacity_addition"].loc[:, :, set_locations, previous_times].sum("set_time_steps_yearly")
                #     term_total_capacity_all_techs_var = term_total_capacity_all_techs_var.loc[other_techs].sum(["set_technologies", "set_location"])
                # else:
                #     # dummy term
                #     term_total_capacity_all_techs_var = self.variables["capacity_investment"].loc[tech, :, set_locations, time].where(False).sum("set_location")

                ### formulate constraint
                # build the lhs
                lhs = (self.variables["capacity_investment"].loc[tech, :, set_locations, time].sum("set_location")
                       - ((1 + self.parameters.max_diffusion_rate.loc[tech, time].item()) ** interval_between_years - 1) * term_total_capacity_knowledge_addition
                       - self.parameters.market_share_unbounded * term_total_capacity_all_techs_var)
                lhs *= mask

                # build the rhs
                rhs = (((1 + self.parameters.max_diffusion_rate.loc[tech, time].item()) ** interval_between_years - 1) * term_total_capacity_knowledge_existing
                       # add initial market share until which the diffusion rate is unbounded
                       + self.parameters.market_share_unbounded * term_total_capacity_all_techs_param + self.parameters.capacity_addition_unbounded.loc[tech]*len(set_locations))
                rhs *= mask

                constraints.append(lhs <= rhs)

        ### return
        # reording takes too much memory!
        return self.constraints.return_contraints(constraints, model=self.model, stack_dim_name="diffusion_limit_total_dim")

    def constraint_capex_yearly_block(self):
        """ aggregates the capex of built capacity and of existing capacity

        .. math::
            A_{h,p,y} = f_h (\\sum_{\\tilde{y} = \\max(y_0,y-\\lceil\\frac{l_h}{\\Delta^\mathrm{y}}\\rceil+1)}^y \\alpha_{h,y}\\Delta S_{h,p,\\tilde{y}}
            + \\sum_{\\hat{y}=\\psi(\\min(y_0-1,y-\\lceil\\frac{l_h}{\\Delta^\mathrm{y}}\\rceil+1))}^{\\psi(y_0)} \\alpha_{h,y_0}\\Delta s^\mathrm{ex}_{h,p,\\hat{y}})

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over all technologies and yearly time steps because we need to calculate the lifetime range
        # we vectorize over capacities and locations
        constraints = []
        for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):

            ### auxiliary calculations
            discount_rate = self.analysis["discount_rate"]
            lifetime = self.parameters.lifetime.loc[tech].item()
            annuity = ((1+discount_rate)**lifetime * discount_rate)/((1+discount_rate)**lifetime - 1)
            term_neg_annuity_cost_capex_previous = []
            for previous_year in Technology.get_lifetime_range(self.optimization_setup, tech, year, time_step_type="yearly"):
                term_neg_annuity_cost_capex_previous.append(-annuity * self.variables["cost_capex"].loc[tech, :, :, previous_year])

            ### formulate constraint
            lhs = lp_sum([1.0 * self.variables["capex_yearly"].loc[tech, :, :, year],
                          *term_neg_annuity_cost_capex_previous])
            rhs = annuity * self.parameters.existing_capex.loc[tech, :, :, year]
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])

    def constraint_opex_technology_block(self):
        """ calculate opex of each technology

        .. math::
            \mathrm{if\ tech\ is\ conversion\ tech}\ OPEX_{h,p,t}^\mathrm{cost} = \\beta_{h,p,t} G_{i,n,t,y}^\mathrm{r}
        .. math::
            \mathrm{if\ tech\ is\ transport\ tech}\ OPEX_{h,p,t}^\mathrm{cost} = \\beta_{h,p,t} F_{j,e,t}
        .. math::
            \mathrm{if\ tech\ is\ storage\ tech}\ OPEX_{h,p,t}^\mathrm{cost} = \\beta_{h,p,t} (\\underline{H}_{k,n,t} + \\overline{H}_{k,n,t})

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_location", "set_time_steps_operation"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over all technologies because of the reference carrier and flow which depend on the technology
        # we vectorize over locations and time steps
        constraints = []
        for tech in index.get_unique(["set_technologies"]):
            locs = index.get_values([tech], "set_location", unique=True)
            reference_carrier = self.sets["set_reference_carriers"][tech][0]
            if tech in self.sets["set_conversion_technologies"]:
                if reference_carrier in self.sets["set_input_carriers"][tech]:
                    reference_flow = self.variables["flow_conversion_input"].loc[tech, reference_carrier, locs].to_linexpr()
                    reference_flow = reference_flow.rename({"set_nodes": "set_location"})
                else:
                    reference_flow = self.variables["flow_conversion_output"].loc[tech, reference_carrier, locs].to_linexpr()
                    reference_flow = reference_flow.rename({"set_nodes": "set_location"})
            elif tech in self.sets["set_transport_technologies"]:
                reference_flow = self.variables["flow_transport"].loc[tech, locs].to_linexpr()
                reference_flow = reference_flow.rename({"set_edges": "set_location"})
            else:
                reference_flow = self.variables["flow_storage_charge"].loc[tech, locs] + self.variables["flow_storage_discharge"].loc[tech, locs]
                reference_flow = reference_flow.rename({"set_nodes": "set_location"})

            term_reference_flow = - self.parameters.opex_specific_variable.loc[tech, locs] * reference_flow

            ### formulate constraint
            # the first term is just to ensure full shape
            lhs = lp.merge(self.variables["cost_opex"].loc[tech].where(False).to_linexpr(),
                           self.variables["cost_opex"].loc[tech, locs].to_linexpr(),
                           term_reference_flow,
                           compat="broadcast_equals")
            rhs = 0
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_technologies"]),
                                                  index_names=["set_technologies"])

    def constraint_opex_yearly_block(self):
        """ yearly opex for a technology at a location in each year

        .. math::
            OPEX_{h,p,y} = \sum_{t\in\mathcal{T}}\tau_t OPEX_{h,p,t}^\mathrm{cost}
            + \gamma_{h,y} S_{h,p,y}
            #TODO complete constraint (second summation symbol)

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_location", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over all technologies and yearly time steps because some auxillary calculations depend on the technology
        # we vectorize over locations
        constraints = []
        for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):

            ### auxiliary calculations
            times = self.time_steps.get_time_steps_year2operation(tech, year)

            term_neg_summed_cost_opex = - (self.variables["cost_opex"].loc[tech, :, times] * self.parameters.time_steps_operation_duration.loc[tech, times]).sum(["set_time_steps_operation"])
            term_neg_summed_capacities = - lp_sum([self.parameters.opex_specific_fixed.loc[tech, capacity_type, :, year]*self.variables["capacity"].loc[tech, capacity_type, :, year]
                                                   for capacity_type in self.system["set_capacity_types"] if tech in self.sets["set_storage_technologies"] or capacity_type == self.system["set_capacity_types"][0]])

            ### formulate constraint
            lhs = lp_sum([1.0 * self.variables["opex_yearly"].loc[tech, :, year],
                          term_neg_summed_cost_opex,
                          term_neg_summed_capacities])
            rhs = 0
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])

    def constraint_carbon_emissions_technology_block(self):
        """ calculate carbon emissions of each technology

        .. math::
            \mathrm{if\ tech\ is\ conversion\ tech}\ E_{h,p,t} = \\epsilon_h G_{i,n,t,y}^\mathrm{r}
        .. math::
            \mathrm{if\ tech\ is\ transport\ tech}\ E_{h,p,t} = \\epsilon_h F_{j,e,t}
        .. math::
            \mathrm{if\ tech\ is\ storage\ tech}\ E_{h,p,t} = \\epsilon_h (\\underline{H}_{k,n,t} + \\overline{H}_{k,n,t})

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_location", "set_time_steps_operation"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over all technologies because of the reference carrier and flow which depend on the technology
        # we vectorize over locations and time steps
        constraints = []
        for tech in index.get_unique(["set_technologies"]):

            ### auxiliary calculations
            locs = index.get_values([tech], 1, unique=True)
            reference_carrier = self.sets["set_reference_carriers"][tech][0]
            if tech in self.sets["set_conversion_technologies"]:
                if reference_carrier in self.sets["set_input_carriers"][tech]:
                    reference_flow = self.variables["flow_conversion_input"].loc[tech, reference_carrier, locs].to_linexpr()
                    reference_flow = reference_flow.rename({"set_nodes": "set_location"})
                else:
                    reference_flow = self.variables["flow_conversion_output"].loc[tech, reference_carrier, locs].to_linexpr()
                    reference_flow = reference_flow.rename({"set_nodes": "set_location"})
            elif tech in self.sets["set_transport_technologies"]:
                reference_flow = self.variables["flow_transport"].loc[tech, locs].to_linexpr()
                reference_flow = reference_flow.rename({"set_edges": "set_location"})
            else:
                reference_flow = self.variables["flow_storage_charge"].loc[tech, locs] + self.variables["flow_storage_discharge"].loc[tech, locs]
                reference_flow = reference_flow.rename({"set_nodes": "set_location"})

            term_reference_flow = - self.parameters.carbon_intensity_technology.loc[tech, locs] * reference_flow

            ### formulate constraint
            # the first term is just to ensure full shape
            lhs = lp.merge(self.variables["carbon_emissions_technology"].loc[tech].where(False).to_linexpr(),
                           self.variables["carbon_emissions_technology"].loc[tech, locs].to_linexpr(),
                           term_reference_flow,
                           compat="broadcast_equals")
            rhs = 0
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_technologies"]),
                                                  index_names=["set_technologies"])

    def constraint_carbon_emissions_technology_total_block(self):
        """ calculate total carbon emissions of each technology

        .. math::
            E_y^{\mathcal{H}} = \sum_{t\in\mathcal{T}}\sum_{h\in\mathcal{H}} E_{h,p,t} \\tau_{t}

        :return: #TODO describe parameter/return
        """

        ### index sets
        years = self.sets["set_time_steps_yearly"]
        # this index is just for the sums in the auxiliary calculations
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_location"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we cycle over the years, because the sum of the operational time steps depends on the year
        constraints = []
        for year in years:

            ### auxiliary calculations
            term_summed_carbon_emissions_technology = []
            for tech in index.get_unique(["set_technologies"]):
                locs = index.get_values([tech], "set_location", unique=True)
                times = self.time_steps.get_time_steps_year2operation(tech, year)
                term_summed_carbon_emissions_technology.append((self.variables["carbon_emissions_technology"].loc[tech, locs, times] * self.parameters.time_steps_operation_duration.loc[tech, times]).sum())
            term_summed_carbon_emissions_technology = lp_sum(term_summed_carbon_emissions_technology)

            ### formulate constraint
            lhs = self.variables["carbon_emissions_technology_total"].loc[year] - term_summed_carbon_emissions_technology
            rhs = 0
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=years,
                                                  index_names=["set_time_steps_yearly"])

    def constraint_capacity_factor_block(self):
        """ Load is limited by the installed capacity and the maximum load factor

        .. math::
            \mathrm{if\ tech\ is\ conversion\ tech}\ G_{i,n,t,y}^\mathrm{r} \\leq m_{i,n,t,y}S_{i,n,y}
        .. math::
            \mathrm{if\ tech\ is\ transport\ tech}\ F_{j,e,t,y}^\mathrm{r} \\leq m_{j,e,t,y}S_{j,e,y}
        .. math::
            \mathrm{if\ tech\ is\ storage\ tech}\ \\underline{H}_{k,n,t,y}+\\overline{H}_{k,n,t,y}\\leq m_{k,n,t,y}S_{k,n,y}

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_operation"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we oop over all technologies for the conditions and vectorize over the rest
        constraints = []
        for tech in index.get_unique(["set_technologies"]):

            ### auxiliary calculations
            capacity_types, locs, times = index.get_values([tech], [1, 2, 3], unique=True)
            # to actual coords to avoid renaming
            capacity_types = self.variables.coords["set_capacity_types"].loc[capacity_types]
            locs = self.variables.coords["set_location"].loc[locs]
            times = self.variables.coords["set_time_steps_operation"].loc[times]
            # the reference carrier
            reference_carrier = self.sets["set_reference_carriers"][tech][0]
            # get invest time step
            time_step_year = xr.DataArray([self.optimization_setup.energy_system.time_steps.convert_time_step_operation2year(tech, t) for t in times.data], coords=[times])
            # we create the capacity term (the dimension reassignment does not change the variables, just the broadcasting)
            term_capacity = self.parameters.max_load.loc[tech, capacity_types, locs, times] * self.variables["capacity"].loc[tech, capacity_types, locs, time_step_year].to_linexpr()

            # this term is just to ensure full shape
            full_shape_term = self.variables["capacity"].loc[tech, ..., time_step_year].where(False).to_linexpr()

            # conversion technology
            if tech in self.sets["set_conversion_technologies"]:
                if reference_carrier in self.sets["set_input_carriers"][tech]:
                    term_flow = -1.0 * self.variables["flow_conversion_input"].loc[tech, reference_carrier, locs, times]
                else:
                    term_flow = -1.0 * self.variables["flow_conversion_output"].loc[tech, reference_carrier, locs, times]
            # transport technology
            elif tech in self.sets["set_transport_technologies"]:
                term_flow = -1.0 * self.variables["flow_transport"].loc[tech, locs, times]
            # storage technology
            elif tech in self.sets["set_storage_technologies"]:
                system = self.optimization_setup.system
                # if limit power
                mask = (capacity_types == system["set_capacity_types"][0]).astype(float)
                # where true
                term_flow = mask*(-1.0 * self.variables["flow_storage_charge"].loc[tech, locs, times] - 1.0 * self.variables["flow_storage_discharge"].loc[tech, locs, times])

                # TODO integrate level storage here as well

            ### formulate constraint
            lhs = lp.merge(lp.merge(term_capacity, term_flow), full_shape_term)
            rhs = 0
            constraints.append(lhs >= rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_technologies"]),
                                                  index_names=["set_technologies"])
