"""
:Title: ZEN-GARDEN
:Created: October-2021
:Authors:   Alissa Ganter (aganter@ethz.ch), Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Class defining the parameters, variables, and constraints of the retrofitting technologies.
The class takes the abstract optimization model as an input and adds parameters, variables, and
constraints of the retrofitting technologies.
"""
import itertools
import logging

import numpy as np
import pandas as pd
import xarray as xr

from zen_garden.utils import linexpr_from_tuple_np, InputDataChecks, align_like
from .conversion_technology import ConversionTechnology
from ..component import ZenIndex
from ..element import GenericRule


class RetrofittingTechnology(ConversionTechnology):
    """
    Class defining conversion technologies
    """
    # set label
    label = "set_retrofitting_technologies"
    location_type = "set_nodes"

    def __init__(self, tech, optimization_setup):
        """
        init conversion technology object

        :param tech: name of added technology
        :param optimization_setup: The OptimizationSetup the element is part of
        """
        super().__init__(tech, optimization_setup)

    def store_carriers(self):
        """ retrieves and stores information on reference, input and output carriers """
        # get reference carrier from class <Technology>
        super().store_carriers()

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().store_input_data()
        # get retrofit base technology
        self.retrofit_base_technology = self.data_input.extract_retrofit_base_technology()
        # get flow_coupling factor and capex
        self.retrofit_flow_coupling_factor = self.data_input.extract_input_data("retrofit_flow_coupling_factor", index_sets=["set_nodes", "set_time_steps"], unit_category={})

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to ConversionTechnology --- ###
    @classmethod
    def construct_sets(cls, optimization_setup):
        """ constructs the pe.Sets of the class <RetrofittingTechnology>

        :param optimization_setup: The OptimizationSetup the element is part of """

        # get base technologies
        retrofit_base_technology = optimization_setup.get_attribute_of_all_elements(cls, "retrofit_base_technology")

        # retrofitting base technologies
        optimization_setup.sets.add_set(name="set_retrofitting_base_technologies", data=retrofit_base_technology,
                                        doc="set of base technologies for a specific retrofitting technology. Indexed by set_retrofitting_technologies",
                                        index_set="set_retrofitting_technologies")

    @classmethod
    def construct_params(cls, optimization_setup):
        """ constructs the pe.Params of the class <RetrofittingTechnology>

        :param optimization_setup: The OptimizationSetup the element is part of """

        # slope of linearly modeled capex
        optimization_setup.parameters.add_parameter(name="retrofit_flow_coupling_factor", index_names=["set_retrofitting_technologies", "set_nodes", "set_time_steps_operation"], capacity_types=False, doc="Parameter which specifies the flow coupling between the retrofitting technologies and its base technology", calling_class=cls)

    @classmethod
    def construct_constraints(cls, optimization_setup):
        """ constructs the Constraints of the class <RetrofittingTechnology>

        :param optimization_setup: The OptimizationSetup the element is part of """
        # add pwa constraints
        rules = RetrofittingTechnologyRules(optimization_setup)

        # flow coupling of retrofitting technology and its base technology
        rules.constraint_retrofit_flow_coupling()

class RetrofittingTechnologyRules(GenericRule):
    """
    Rules for the RetrofittingTechnology class
    """

    def __init__(self, optimization_setup):
        """
        Inits the rules for a given EnergySystem
        :param optimization_setup: The OptimizationSetup the element is part of
        """

        super().__init__(optimization_setup)

    def constraint_retrofit_flow_coupling(self):
        """ couples reference flow variables based on modeling technique

        .. math::
            \mathrm{if\ reference\ carrier\ in\ input\ carriers}\ \\underline{G}_{i,n,t}^\mathrm{r} = G^\mathrm{d,approximation}_{i,n,t}
        .. math::
            \mathrm{if\ reference\ carrier\ in\ output\ carriers}\ \\overline{G}_{i,n,t}^\mathrm{r} = G^\mathrm{d,approximation}_{i,n,t}

        """
        flow_conversion_input = self.variables["flow_conversion_input"]
        flow_conversion_output = self.variables["flow_conversion_output"]
        rc_in = pd.Series(
            {(t, c): True if c in self.sets["set_reference_carriers"][t] else False for t, c in
             itertools.product(self.sets["set_conversion_technologies"],
                               self.sets["set_input_carriers"].superset)})
        rc_out = pd.Series(
            {(t, c): True if c in self.sets["set_reference_carriers"][t] else False for t, c in
             itertools.product(self.sets["set_conversion_technologies"],
                               self.sets["set_output_carriers"].superset)})
        rc_in.index.names = ["set_conversion_technologies", "set_input_carriers"]
        rc_out.index.names = ["set_conversion_technologies", "set_output_carriers"]
        rc_in = align_like(rc_in.to_xarray(), flow_conversion_input)
        rc_out = align_like(rc_out.to_xarray(), flow_conversion_output)
        term_flow_reference = (
                flow_conversion_input.where(rc_in).sum("set_input_carriers")
                + flow_conversion_output.where(rc_out).sum("set_output_carriers"))
        retrofit_base_technologies = pd.Series(
            {t: rt for t in self.sets["set_conversion_technologies"] if
             t in self.sets["set_retrofitting_base_technologies"] for rt in
             self.sets["set_retrofitting_base_technologies"][t]},
            name="set_conversion_technologies")
        retrofit_base_technologies.index.name = "set_conversion_technologies"
        term_flow_retrofit = self.map_and_expand(term_flow_reference, retrofit_base_technologies)
        term_flow_base = term_flow_reference.sel({"set_conversion_technologies": self.sets["set_retrofitting_technologies"]})
        lhs = term_flow_base - self.parameters.retrofit_flow_coupling_factor * term_flow_retrofit
        rhs = 0
        constraints = lhs <= rhs

        self.constraints.add_constraint("name",constraints)
