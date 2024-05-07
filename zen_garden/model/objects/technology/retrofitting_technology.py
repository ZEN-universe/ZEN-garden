"""
:Title: ZEN-GARDEN
:Created: October-2021
:Authors:   Alissa Ganter (aganter@ethz.ch), Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Class defining the parameters, variables, and constraints of the retrofitting technologies.
The class takes the abstract optimization model as an input and adds parameters, variables, and
constraints of the retrofitting technologies.
"""
import logging

import numpy as np
import pandas as pd
import xarray as xr

from zen_garden.utils import linexpr_from_tuple_np, InputDataChecks
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
                                        doc="set of base technologies for a specific retrofitting technology. Dimensions: set_retrofitting_technologies",
                                        index_set="set_retrofitting_technologies")

    @classmethod
    def construct_params(cls, optimization_setup):
        """ constructs the pe.Params of the class <RetrofittingTechnology>

        :param optimization_setup: The OptimizationSetup the element is part of """

        # slope of linearly modeled capex
        optimization_setup.parameters.add_parameter(name="retrofit_flow_coupling_factor", index_names=["set_retrofitting_technologies", "set_nodes", "set_time_steps_operation"], capacity_types=False, doc="Parameter which specifies the flow coupling between the retrofitting technologies and its base technology", calling_class=cls)

    @classmethod
    def construct_constraints(cls, optimization_setup):
        """ constructs the pe.Constraints of the class <RetrofittingTechnology>

        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        constraints = optimization_setup.constraints
        # add pwa constraints
        rules = RetrofittingTechnologyRules(optimization_setup)

        # flow coupling of retrofitting technology and its base technology
        constraints.add_constraint_block(model, name="constraint_retrofit_flow_coupling",
                                         constraint=rules.constraint_retrofit_flow_coupling_block(
                                             *cls.create_custom_set(["set_retrofitting_technologies", "set_nodes", "set_time_steps_operation"], optimization_setup)),
                                         doc="couples the reference flows of the retrofitting technology and its base technology")

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


    # Rule-based constraints
    # -----------------------

    # Block-based constraints
    # -----------------------

    def constraint_retrofit_flow_coupling_block(self, index_values, index_names):
        """ couples reference flow variables based on modeling technique

        .. math::
            \mathrm{if\ reference\ carrier\ in\ input\ carriers}\ \\underline{G}_{i,n,t}^\mathrm{r} = G^\mathrm{d,approximation}_{i,n,t}
        .. math::
            \mathrm{if\ reference\ carrier\ in\ output\ carriers}\ \\overline{G}_{i,n,t}^\mathrm{r} = G^\mathrm{d,approximation}_{i,n,t}

        :param index_values: index values
        :param index_names: index names
        :return: linopy constraints
        """

        ### index sets
        # check if we even have something
        if len(index_values) == 0:
            return []
        index = ZenIndex(index_values, index_names)

        ### masks
        # note necessary

        ### index loop
        # we loop over all technologies and dependent carriers, mostly to avoid renaming of dimension
        constraints = []
        for retrofit_tech in index.get_unique([0]):

            ### auxiliary calculations
            # get coords
            coords = [self.variables.coords["set_nodes"], self.variables.coords["set_time_steps_operation"]]
            nodes = index.get_values([retrofit_tech], 1, dtype=list, unique=True)
            times = index.get_values([retrofit_tech], 2, dtype=list, unique=True)
            # get flow term retrofit technology and base technology
            term_flow_retrofit_tech = ConversionTechnology.get_flow_term_reference_carrier(self.optimization_setup, tech=retrofit_tech)
            base_tech = self.sets["set_retrofitting_base_technologies"][retrofit_tech][0]
            term_flow_base_tech = ConversionTechnology.get_flow_term_reference_carrier(self.optimization_setup, tech=base_tech)

            ### formulate constraint
            lhs = linexpr_from_tuple_np(
                [(1.0, term_flow_retrofit_tech),
                 (-self.parameters.retrofit_flow_coupling_factor.loc[retrofit_tech, nodes, times],term_flow_base_tech)],
                coords=coords, model=self.model)
            rhs = 0
            constraints.append(lhs <= rhs)

        ### return
        return self.constraints.return_contraints(constraints, model=self.model, stack_dim_name="constraint_retrofit_flow_coupling_dim")


