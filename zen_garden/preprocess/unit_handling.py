"""
:Title:          ZEN-GARDEN
:Created:        April-2022
:Authors:        Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Class containing the unit handling procedure.
"""
import cProfile
import logging
import numpy as np
import pandas as pd
import scipy as sp
import warnings
import json
import os
import linopy as lp
import re
from pint import UnitRegistry
from pint.util import column_echelon_form
from pathlib import Path
from zen_garden.model.objects.technology.technology import Technology
from zen_garden.model.objects.carrier.carrier import Carrier

import time


# enable Deprecation Warnings
warnings.simplefilter('always', DeprecationWarning)


class UnitHandling:
    """
    Class containing the unit handling procedure
    """

    def __init__(self, folder_path, rounding_decimal_points_units):
        """ initialization of the unit_handling instance

        :param folder_path: The path to the folder containing the system specifications
        :param round_decimal_points: rounding tolerance
        """
        self.folder_path = folder_path
        self.rounding_decimal_points_units = rounding_decimal_points_units
        self.get_base_units()
        # dict of element attribute values
        self.dict_attribute_values = {}
        self.carrier_energy_quantities = {}

    def get_base_units(self):
        """ gets base units of energy system """
        _list_base_unit = self.extract_base_units()
        self.ureg = UnitRegistry()

        # disable pint logger
        logging.getLogger("pint").setLevel(logging.CRITICAL)
        # redefine standard units
        self.redefine_standard_units()
        # load additional units
        self.ureg.load_definitions(self.folder_path / "unit_definitions.txt")

        # empty base units and dimensionality matrix
        self.base_units = {}
        self.dim_matrix = pd.DataFrame(index=_list_base_unit).astype(int)
        for base_unit in _list_base_unit:
            dim_unit = self.ureg.get_dimensionality(self.ureg(base_unit))
            self.base_units[base_unit] = self.ureg(base_unit).dimensionality
            self.dim_matrix.loc[base_unit, list(dim_unit.keys())] = list(dim_unit.values())
        self.dim_matrix = self.dim_matrix.fillna(0).astype(int).T

        # check if unit defined twice or more
        duplicate_units = self.dim_matrix.T.duplicated()
        if duplicate_units.any():
            dim_matrix_duplicate = self.dim_matrix.loc[:, duplicate_units]
            for duplicate in dim_matrix_duplicate:
                # if same unit twice (same order of magnitude and same dimensionality)
                if len(self.dim_matrix[duplicate].shape) > 1:
                    logging.warning(f"The base unit <{duplicate}> was defined more than once. Duplicates are dropped.")
                    _duplicateDim = self.dim_matrix[duplicate].T.drop_duplicates().T
                    self.dim_matrix = self.dim_matrix.drop(duplicate, axis=1)
                    self.dim_matrix[duplicate] = _duplicateDim
                else:
                    raise KeyError(f"More than one base unit defined for dimensionality {self.base_units[duplicate]} (e.g., {duplicate})")
        # get linearly dependent units
        M, I, pivot = column_echelon_form(np.array(self.dim_matrix), ntype=float)
        M = np.array(M).squeeze()
        I = np.array(I).squeeze()
        pivot = np.array(pivot).squeeze()
        # index of linearly dependent units in M and I
        idx_lin_dep = np.squeeze(np.argwhere(np.all(M == 0, axis=1)))
        # index of linearly dependent units in dimensionality matrix
        _idx_pivot = range(len(self.base_units))
        idx_lin_dep_dim_matrix = list(set(_idx_pivot).difference(pivot))
        self.dim_analysis = {}
        self.dim_analysis["dependent_units"] = self.dim_matrix.columns[idx_lin_dep_dim_matrix]
        dependent_dims = I[idx_lin_dep, :]
        # if only one dependent unit
        if len(self.dim_analysis["dependent_units"]) == 1:
            dependent_dims = dependent_dims.reshape(1, dependent_dims.size)
        # reorder dependent dims to match dependent units
        dim_of_dependent_units = dependent_dims[:, idx_lin_dep_dim_matrix]
        # if not already in correct order (ones on the diagonal of dependent_dims)
        if not np.all(np.diag(dim_of_dependent_units) == 1):
            # get position of ones in dim_of_dependent_units
            pos_ones = np.argwhere(dim_of_dependent_units == 1)
            assert np.size(pos_ones, axis=0) == len(self.dim_analysis["dependent_units"]), f"Cannot determine order of dependent base units {self.dim_analysis['dependent_units']}, " \
                                                                                           f"because diagonal of dimensions of the dependent units cannot be determined."
            # pivot dependent dims
            dependent_dims = dependent_dims[pos_ones[:, 1], :]
        self.dim_analysis["dependent_dims"] = dependent_dims
        # check that no base unit can be directly constructed from the others (e.g., GJ from GW and hour)
        assert ~UnitHandling.check_pos_neg_boolean(dependent_dims, axis=1), f"At least one of the base units {list(self.base_units.keys())} can be directly constructed from the others"

    def extract_base_units(self):
        """ extracts base units of energy system

        :return list_base_units: list of base units """
        list_base_units = pd.read_csv(self.folder_path / "base_units.csv").squeeze().values.tolist()
        return list_base_units

    def calculate_combined_unit(self, input_unit, return_combination=False):
        """ calculates the combined unit for converting an input_unit to the base units

        :param input_unit: string of input unit
        :param return_combination: If True, return the combination of units
        :return combined_unit: multiplication factor """
        # check if "h" and thus "planck_constant" in unit
        self.check_if_invalid_hourstring(input_unit)
        # create dimensionality vector for input_unit
        dim_input = self.ureg.get_dimensionality(self.ureg(input_unit))
        dim_vector = pd.Series(index=self.dim_matrix.index, data=0)
        missing_dim = set(dim_input.keys()).difference(dim_vector.keys())
        assert len(missing_dim) == 0, f"No base unit defined for dimensionalities <{missing_dim}>"
        if len(dim_input) > 0:  # check for content of dim_input to avoid Warning
            dim_vector[list(dim_input.keys())] = list(dim_input.values())
        # calculate dimensionless combined unit (e.g., tons and kilotons)
        combined_unit = self.ureg(input_unit).units
        # if unit (with a different multiplier) is already in base units
        if self.dim_matrix.isin(dim_vector).all(axis=0).any():
            base_combination = self.dim_matrix.isin(dim_vector).all(axis=0).astype(int)
            base_unit = self.ureg(self.dim_matrix.columns[self.dim_matrix.isin(dim_vector).all(axis=0)][0])
            combined_unit *= base_unit ** (-1)
        # if inverse of unit (with a different multiplier) is already in base units (e.g. 1/km and km)
        elif (self.dim_matrix * -1).isin(dim_vector).all(axis=0).any():
            base_combination = (self.dim_matrix * -1).isin(dim_vector).all(axis=0).astype(int) * (-1)
            base_unit = self.ureg(self.dim_matrix.columns[(self.dim_matrix * -1).isin(dim_vector).all(axis=0)][0])
            combined_unit *= base_unit
        else:
            # drop dependent units
            dim_matrix_reduced = self.dim_matrix.drop(self.dim_analysis["dependent_units"], axis=1)
            # solve system of linear equations
            combination_solution = np.linalg.solve(dim_matrix_reduced, dim_vector)
            # check if only -1, 0, 1
            if UnitHandling.check_pos_neg_boolean(combination_solution):
                base_combination = pd.Series(index=self.dim_matrix.columns, data=0)
                base_combination[dim_matrix_reduced.columns] = combination_solution
                # compose relevant units to dimensionless combined unit
                for unit, power in zip(dim_matrix_reduced.columns, combination_solution):
                    combined_unit *= self.ureg(unit) ** (-1 * power)
            else:
                base_combination,combined_unit = self._get_combined_unit_of_different_matrix(
                    dim_matrix_reduced= dim_matrix_reduced,
                    dim_vector=dim_vector,
                    input_unit=input_unit
                )
        if return_combination:
            return combined_unit, base_combination
        else:
            return combined_unit

    def _get_combined_unit_of_different_matrix(self,dim_matrix_reduced,dim_vector,input_unit):
        """ calculates the combined unit for a different dimensionality matrix.
        We substitute base units by the dependent units and try again.
        If the matrix is singular we solve the overdetermined problem
        :param dim_matrix_reduced: dimensionality matrix without dependent units
        :param dim_vector: dimensionality vector of input unit
        :param input_unit: input unit
        :return base_combination: base combination of input unit
        :return combined_unit: input unit expressed in base units
        """
        calculated_multiplier = False
        combined_unit = self.ureg(input_unit).units
        base_combination = pd.Series(index=self.dim_matrix.columns, data=0)
        # try to substitute unit by a dependent unit
        for unit in dim_matrix_reduced.columns:
            if not calculated_multiplier:
                # iterate through dependent units
                for dependent_unit, dependent_dim in zip(self.dim_analysis["dependent_units"],
                                                         self.dim_analysis["dependent_dims"]):
                    # substitute current unit with dependent unit
                    dim_matrix_reduced_temp = dim_matrix_reduced.drop(unit, axis=1)
                    dim_matrix_reduced_temp[dependent_unit] = self.dim_matrix[dependent_unit]
                    # if full rank
                    if np.linalg.matrix_rank == np.size(dim_matrix_reduced_temp, 1):
                        combination_solution_temp = np.linalg.solve(dim_matrix_reduced_temp, dim_vector)
                    # if singular, check if zero row in matrix corresponds to zero row in unit dimensionality
                    else:
                        zero_row = dim_matrix_reduced_temp.index[~dim_matrix_reduced_temp.any(axis=1)]
                        if (dim_vector[zero_row] == 0).all():
                            # remove zero row
                            dim_matrix_reduced_temp_reduced = dim_matrix_reduced_temp.drop(zero_row, axis=0)
                            dim_vector_reduced = dim_vector.drop(zero_row, axis=0)
                            # formulate as optimization problem with 1,-1 bounds
                            # to determine solution of overdetermined matrix
                            ub = np.array([1] * len(dim_matrix_reduced_temp_reduced.columns))
                            lb = np.array([-1] * len(dim_matrix_reduced_temp_reduced.columns))
                            res = sp.optimize.lsq_linear(
                                dim_matrix_reduced_temp_reduced, dim_vector_reduced,
                                bounds=(lb, ub))
                            # if an exact solution is found (after rounding)
                            if np.round(res.cost, 4) == 0:
                                combination_solution_temp = np.round(res.x, 4)
                            # if not solution is found
                            else:
                                continue
                        # definitely not a solution because zero row corresponds to nonzero dimensionality
                        else:
                            continue
                    if UnitHandling.check_pos_neg_boolean(combination_solution_temp):
                        # compose relevant units to dimensionless combined unit
                        base_combination[dim_matrix_reduced_temp.columns] = combination_solution_temp
                        for unit_temp, power_temp in zip(dim_matrix_reduced_temp.columns, combination_solution_temp):
                            combined_unit *= self.ureg(unit_temp) ** (-1 * power_temp)
                        calculated_multiplier = True
                        break
        assert calculated_multiplier, f"Cannot establish base unit conversion for {input_unit} from base units {self.base_units.keys()}"
        return base_combination,combined_unit

    def get_unit_multiplier(self, input_unit, attribute_name, path=None, combined_unit=None):
        """ calculates the multiplier for converting an input_unit to the base units

        :param input_unit: string of input unit
        :param attribute_name: name of attribute
        :param path: path of element
        :return multiplier: multiplication factor """
        # if input unit is already in base units --> the input unit is base unit, multiplier = 1
        if input_unit in self.base_units:
            return 1
        # if input unit is nan --> dimensionless old definition
        elif type(input_unit) != str and np.isnan(input_unit):
            warnings.warn(f"Parameter {attribute_name} of {Path(path).name} has no unit (assign unit '1' to unitless parameters)",DeprecationWarning)
            return 1
        else:
            # convert to string
            input_unit = str(input_unit)
            # if input unit is 1 --> dimensionless new definition
            if input_unit == "1":
                return 1
            if not combined_unit:
                combined_unit = self.calculate_combined_unit(input_unit)
            assert combined_unit.to_base_units().unitless, f"The unit conversion of unit {input_unit} did not resolve to a dimensionless conversion factor. Something went wrong."
            # magnitude of combined unit is multiplier
            multiplier = combined_unit.to_base_units().magnitude
            # check that multiplier is larger than rounding tolerance
            assert multiplier >= 10 ** (-self.rounding_decimal_points_units), f"Multiplier {multiplier} of unit {input_unit} in parameter {attribute_name} is smaller than rounding tolerance {10 ** (-self.rounding_decimal_points_units)}"
            # round to decimal points
            return round(multiplier, self.rounding_decimal_points_units)

    def convert_unit_into_base_units(self, input_unit, get_multiplier=False, attribute_name=None, path=None):
        """Converts the input_unit into base units and returns the multiplier such that the combined unit mustn't be computed twice

        :param input_unit: unit read from input csv files
        :param attribute_name: name of the attribute the input_unit corresponds to
        :param path: path of the attribute's csv file
        :param get_multiplier: bool whether multiplier should be returned or not
        :return: multiplier to convert input_unit to base  units, pint Quantity of input_unit converted to base units
        """
        # convert attribute unit into unit combination of base units
        combined_unit = None
        attribute_unit_in_base_units = self.ureg("")
        if input_unit != "1" and not pd.isna(input_unit):
            combined_unit, base_combination = self.calculate_combined_unit(input_unit, return_combination=True)
            for unit, power in zip(base_combination.index, base_combination):
                attribute_unit_in_base_units *= self.ureg(unit) ** power
        # calculate the multiplier to convert the attribute unit into base units
        if get_multiplier:
            multiplier = self.get_unit_multiplier(input_unit, attribute_name, path, combined_unit=combined_unit)
            return multiplier, attribute_unit_in_base_units
        else:
            return attribute_unit_in_base_units

    def consistency_checks_input_units(self, optimization_setup):
        """Checks if the units of the parameters specified in the input csv files are consistent

        :param optimization_setup: OptimizationSetup object
        """
        if not optimization_setup.solver["check_unit_consistency"]:
            return
        elements = optimization_setup.dict_elements["Element"]
        items = elements + [optimization_setup.energy_system]
        conversion_factor_units = {}
        retrofit_flow_coupling_factors = {}
        for item in items:
            energy_quantity_units = {}
            unit_dict = item.units
            # since technology elements have a lot of parameters related to their reference carrier, their unit consistency must be checked together (second if for retrofit techs)
            if isinstance(item, Technology):
                reference_carrier = optimization_setup.get_element(cls=Carrier,name=item.reference_carrier[0])
                unit_dict.update(reference_carrier.units)
            # add units of conversion factors/flow coupling factors to carrier units to perform consistency checks (works only since carriers are located at end of optimization_setup.dict_elements)
            if isinstance(item, Carrier):
                for tech_name, cf_dict in conversion_factor_units.items():
                    for dependent_carrier, unit_pair in cf_dict.items():
                        units_to_check = [unit for key, unit in unit_pair.items() if key == item.name]
                        if len(units_to_check) != 0:
                            unit_in_base_units = self.convert_unit_into_base_units(units_to_check[0])
                            energy_quantity_units.update({tech_name+"_conversion_factor_"+dependent_carrier: unit_in_base_units})
                for tech_name, fcf_dict in retrofit_flow_coupling_factors.items():
                    for dependent_carrier, unit_pair in fcf_dict.items():
                        units_to_check = [unit for key, unit in unit_pair.items() if key == item.name]
                        if len(units_to_check) != 0:
                            unit_in_base_units = self.convert_unit_into_base_units(units_to_check[0])
                            energy_quantity_units.update({tech_name+"_retrofit_flow_coupling_factor_"+dependent_carrier: unit_in_base_units})
            # conduct consistency checks
            for attribute_name, unit_specs in unit_dict.items():
                if attribute_name == "conversion_factor":
                    conversion_factor_units[item.name] = self._get_conversion_factor_units(item, unit_specs, reference_carrier, elements)
                elif attribute_name == "retrofit_flow_coupling_factor":
                    # reference_carrier = optimization_setup.get_element(cls=Carrier,name=item.retrofit_reference_carrier[0])
                    base_technology = optimization_setup.get_element(cls=Technology,name=item.retrofit_base_technology[0])
                    reference_carrier = optimization_setup.get_element(cls=Carrier,name=base_technology.reference_carrier[0])
                    retrofit_flow_coupling_factors[item.name] = self._get_conversion_factor_units(item, unit_specs, reference_carrier, elements)
                elif unit_specs["unit_category"] == {}:
                    assert unit_specs["unit_in_base_units"] == self.ureg("dimensionless"), f"The attribute {attribute_name} of {item.__class__.__name__} {item.name} is per definition dimensionless. However, its unit was defined as {unit_specs['unit_in_base_units']}."
                # check if nonlinear capex file exists for conversion technology since the units defined there overwrite the attributes file units
                elif attribute_name == "capex_specific" and hasattr(item, "units_nonlinear_capex_files"):
                    for key, value in item.units_nonlinear_capex_files.items():
                        if "capex" in value:
                            capex_specific_unit = value["capex"].values[0]
                            unit_specs["unit_in_base_units"] = self.convert_unit_into_base_units(capex_specific_unit)
                            energy_quantity_units.update(self._remove_non_energy_units(unit_specs, "capex_"+key))
                        capacity_unit = value["capacity"].values[0]
                        unit_specs["unit_category"] = [value["unit_category"] for key, value in unit_dict.items() if key == "capacity_limit"][0]
                        unit_specs["unit_in_base_units"] = self.convert_unit_into_base_units(capacity_unit)
                        energy_quantity_units.update(self._remove_non_energy_units(unit_specs, "capacity_"+key))
                # units of input/output/reference carrier not of interest for consistency
                elif attribute_name not in ["input_carrier", "output_carrier", "reference_carrier"]:
                    energy_quantity_units.update(self._remove_non_energy_units(unit_specs, attribute_name))
            # remove attributes whose units became dimensionless since they don't have an energy quantity
            energy_quantity_units = {key: value for key, value in energy_quantity_units.items() if value != self.ureg("dimensionless")}
            # check if conversion factor units are consistent
            self._check_for_power_power_conversion_factor(energy_quantity_units)
            # check if units are consistent
            self.assert_unit_consistency(elements, energy_quantity_units, item,optimization_setup, reference_carrier.name, unit_dict)
        logging.info(f"Parameter unit consistency is fulfilled!")
        self.save_carrier_energy_quantities(optimization_setup)

    def _check_for_power_power_conversion_factor(self, energy_quantity_units):
        """
        if unit consistency is not fulfilled because of conversion factor, try to change "wrong" conversion factor units from power/power to energy/energy (since both is allowed)
        :param energy_quantity_units:
        :return:
        """
        if self._is_inconsistent(energy_quantity_units) and not self._is_inconsistent(energy_quantity_units,
                                                                                      exclude_string="conversion_factor"):
            non_cf_energy_quantity_unit = \
            [value for key, value in energy_quantity_units.items() if "conversion_factor" not in key][0]
            cf_energy_quantity_units = {key: value for key, value in energy_quantity_units.items() if
                                        "conversion_factor" in key}
            time_base_unit = [key for key, value in self.base_units.items() if value == "[time]"][0]
            for key, value in cf_energy_quantity_units.items():
                # if conversion factor unit is in not in energy units, try to convert it to energy units by multiplying with time base unit
                if value != non_cf_energy_quantity_unit:
                    energy_quantity_units[key] = value * self.ureg(time_base_unit)

    def assert_unit_consistency(self, elements, energy_quantity_units, item,optimization_setup, reference_carrier_name, unit_dict):
        """Asserts that the units of the attributes of an element are consistent
        :param elements: list of all elements
        :param energy_quantity_units: dict containing attribute names and their energy quantity terms
        :param item: element or energy system
        :param optimization_setup: OptimizationSetup object
        :param reference_carrier_name: name of reference carrier if item is a conversion technology
        :param unit_dict: dict containing attribute names along with their units in base units
        """
        attributes_with_lowest_appearance = self._get_attributes_with_least_often_appearing_unit(energy_quantity_units)
        # assert unit consistency
        if item in elements and self._is_inconsistent(energy_quantity_units):
            # check if there is a conversion factor with wrong units
            wrong_cf_atts = {att: unit for att, unit in attributes_with_lowest_appearance.items() if
                             "conversion_factor" in att}
            name_pairs_cf = []
            if wrong_cf_atts:
                for wrong_cf_att in wrong_cf_atts:
                    names = wrong_cf_att.split("_conversion_factor_")
                    name_pairs_cf.append(names[1] + " of " + names[0])
                self._write_inconsistent_units_file(energy_quantity_units, item.name,
                                                    analysis=optimization_setup.analysis)
                raise AssertionError(
                    f"Unit inconsistency! Most probably, the {item.name} unit(s) of the conversion factor(s) with dependent carrier {name_pairs_cf} are wrong.")
            # check if there is a retrofit flow coupling factor with wrong units
            wrong_rf_atts = {att: unit for att, unit in attributes_with_lowest_appearance.items() if
                             "retrofit_flow_coupling_factor" in att}
            name_pairs_rf = []
            if wrong_rf_atts:
                for wrong_rf_att in wrong_rf_atts:
                    names = wrong_rf_att.split("_retrofit_flow_coupling_factor_")
                    name_pairs_rf.append(names[1] + " of " + names[0])
                self._write_inconsistent_units_file(energy_quantity_units, item.name,
                                                    analysis=optimization_setup.analysis)
                raise AssertionError(
                    f"Unit inconsistency! Most probably, the {item.name} unit(s) of the retrofit flow coupling factor(s) with dependent carrier {name_pairs_rf} are wrong.")
            if item.__class__ is Carrier:
                self._write_inconsistent_units_file(energy_quantity_units, item.name,
                                                    analysis=optimization_setup.analysis)
                raise AssertionError(
                    f"The attribute units of the {item.__class__.__name__} {item.name} are not consistent! Most probably, the unit(s) of the attribute(s) {self._get_units_of_wrong_attributes(wrong_atts=attributes_with_lowest_appearance, unit_dict=unit_dict)} are wrong.")
            else:
                self._write_inconsistent_units_file(energy_quantity_units, item.name,
                                                    analysis=optimization_setup.analysis,
                                                    reference_carrier_name=reference_carrier_name)
                raise AssertionError(
                    f"The attribute units of the {item.__class__.__name__} {item.name} and its reference carrier {reference_carrier_name} are not consistent! Most propably, the unit(s) of the attribute(s) {self._get_units_of_wrong_attributes(wrong_atts=attributes_with_lowest_appearance, unit_dict=unit_dict)} are wrong.")
        # since energy system doesn't have any attributes with energy dimension, its dict must be empty
        elif item not in elements and len(energy_quantity_units) != 0:
            self._write_inconsistent_units_file(energy_quantity_units, item.name, analysis=optimization_setup.analysis)
            raise AssertionError(
                f"The attribute units defined in the energy_system are not consistent! Most probably, the unit(s) of the attribute(s) {self._get_units_of_wrong_attributes(wrong_atts=energy_quantity_units, unit_dict=unit_dict)} are wrong.")

    def _is_inconsistent(self, energy_quantity_units,exclude_string=None):
        """
        Checks if the units of the attributes of an element are inconsistent
        :param energy_quantity_units: dict containing attribute names and their energy quantity terms
        :param exclude_string: string for which consistency is not checked
        :return: bool whether the units are consistent or not
        """
        # exclude attributes which are not of interest for consistency
        if exclude_string:
            energy_quantity_units = {key: value for key, value in energy_quantity_units.items() if exclude_string not in key}
        # check if all energy quantity units are the same
        if len(set(energy_quantity_units.values())) > 1:
            return True
        else:
            return False

    def _get_units_of_wrong_attributes(self, wrong_atts, unit_dict):
        """Gets units of attributes showing wrong units

        :param wrong_atts: dict containing attribute names along with their energy_quantity part of attributes which have inconsistent units
        :param unit_dict: dict containing attribute names along with their units in base units
        :return: dict containing attribute names along with their unit in base unit of attributes which have inconsistent units
        """
        wrong_atts_with_units = {}
        for att in wrong_atts:
            wrong_atts_with_units[att] = [str(unit_specs["unit_in_base_units"].units) for key, unit_specs in unit_dict.items() if key == att][0]
        return wrong_atts_with_units

    def _write_inconsistent_units_file(self, inconsistent_attributes, item_name, analysis, reference_carrier_name=None):
        """Writes file of attributes and their units which cause unit inconsistency

        :param inconsistent_attributes: attributes which are not consistent
        :param item_name: element name or energy system name which shows inconsistent units
        :param analysis:  dictionary defining the analysis settings
        :param reference_carrier_name: name of reference carrier if item is a conversion technology
        """
        inconsistent_attributes_dict = {"element_name": item_name, "reference_carrier": reference_carrier_name, "attribute_names": str(inconsistent_attributes.keys())}
        directory = os.path.join(analysis["folder_output"], os.path.basename(analysis["dataset"]))
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, "inconsistent_units.json")
        with open(path, 'w') as json_file:
            json.dump(inconsistent_attributes_dict, json_file)

    def _get_attributes_with_least_often_appearing_unit(self, energy_quantity_units):
        """Finds all attributes which have the least often appearing unit

        :param energy_quantity_units: dict containing attribute names and their energy quantity terms
        :return: attribute names and energy quantity terms which appear the least often in energy_quantity_units
        """
        min_unit_count = np.inf
        attributes_with_lowest_appearance = {}
        # count for all unique units how many times they appear to get an estimate which unit most likely is the wrong one
        for distinct_unit in set(energy_quantity_units.values()):
            unit_count = list(energy_quantity_units.values()).count(distinct_unit)
            if unit_count <= min_unit_count and unit_count < len(energy_quantity_units)/2:
                min_unit_count = unit_count
                wrong_value = distinct_unit
                attributes_with_lowest_appearance.update({key: value for key, value in energy_quantity_units.items() if value == wrong_value})
        return attributes_with_lowest_appearance

    def get_most_often_appearing_energy_unit(self, energy_units):
        """finds a carriers most likely correct energy unit

        :param energy_units: all the energy_quantity terms of a carriers attributes
        :return: most frequently appearing energy quantity
        """
        max_unit_count = 0
        correct_value = None
        # count for all unique units how many times they appear to get an estimate which unit most likely is the correct one
        for distinct_unit in set(energy_units.values()):
            unit_count = list(energy_units.values()).count(distinct_unit)
            if unit_count > max_unit_count:
                max_unit_count = unit_count
                correct_value = distinct_unit
        return correct_value

    def _get_conversion_factor_units(self, conversion_element, unit_specs, reference_carrier, elements):
        """Splits conversion factor units into dependent carrier and reference carrier part

        :param conversion_element: Conversion technology element the conversion factor belongs to
        :param unit_specs: dict containing unit category and unit as pint Quantity in base units
        :param reference_carrier: Carrier object of conversion_element's reference carrier
        :param elements: list containing all existing elements
        :return: dict of conversion_element's conversion factors' units separated by dependent carrier and reference carrier
        """
        conversion_factor_units = {}
        for dependent_carrier_name, cf_unit_specs in unit_specs.items():
            assert cf_unit_specs["unit"] != "1", f"Since there doesn't exist a conversion_factor file for the technology {conversion_element.name}, the attribute conversion_factor_default must be defined with units to ensure unit consistency"
            units = cf_unit_specs["unit"].split("/")
            # check that no asterisk in unit strings without parentheses
            correct_unit_string = [("*" in u and u[0] == "(" and u[1] == ")") or ("*" not in u) for u in units]
            assert all(correct_unit_string), f"The conversion factor string(s) {[u for u,s in zip(units,correct_unit_string) if not s]} of technology {conversion_element.name} must not contain an asterisk '*' unless it is enclosed in parentheses '()'"

            #problem: we don't know which parts of cf unit belong to which carrier for units of format different from "unit/unit" (e.g. kg/h/kW)
            #method: compare number of division signs of conversion factor unit with number  of division signs of corresponding carrier element energy/power quantity
            dependent_carrier = [carrier for carrier in elements if carrier.name == dependent_carrier_name][0]

            div_signs_dependent_carrier_energy = self._get_number_of_division_signs_energy_quantity(dependent_carrier.units)
            div_signs_ref_carrier_energy = self._get_number_of_division_signs_energy_quantity(reference_carrier.units)
            number_of_division_signs_energy = div_signs_dependent_carrier_energy + div_signs_ref_carrier_energy

            div_signs_dependent_carrier_power = self._get_number_of_division_signs_energy_quantity(dependent_carrier.units, power=True)
            div_signs_ref_carrier_power = self._get_number_of_division_signs_energy_quantity(reference_carrier.units, power=True)
            number_of_division_signs_power = div_signs_ref_carrier_power + div_signs_dependent_carrier_power

            #conversion factor unit must be defined as energy/energy or power/power in the corresponding carrier energy quantity units
            #Check if the conversion factor is defined as energy/energy
            factor_units = {}
            if len(units) - 2 == number_of_division_signs_energy:
                #assign the unit parts to the corresponding carriers
                factor_units[dependent_carrier_name] = units[0:div_signs_dependent_carrier_energy + 1]
                factor_units[reference_carrier.name] = units[div_signs_dependent_carrier_energy + 1:]
            #check if the conversion factor is defined as power/power
            elif len(units) - 2 == number_of_division_signs_power:
                #assign the unit parts to the corresponding carriers
                factor_units[dependent_carrier_name] = units[0:div_signs_dependent_carrier_power + 1]
                factor_units[reference_carrier.name] = units[div_signs_dependent_carrier_power + 1:]
            else:
                raise AssertionError(f"The conversion factor units of technology {conversion_element.name} must be defined as power/power or energy/energy of input/output carrier divided by reference carrier, e.g. MW/MW, MW/kg/s or GWh/GWh, kg/MWh etc.")
            #recombine the separated units carrier-wise to the initial fraction
            for key, value in factor_units.items():
                factor_units[key] = "/".join(value)
            conversion_factor_units[dependent_carrier_name] = factor_units
        return conversion_factor_units

    def _get_number_of_division_signs_energy_quantity(self, carrier_units, power=False):
        """Finds the most common energy quantity of a carrier and counts its number of division signs (or the number of division signs of the resulting power unit)

        :param carrier_units: unit attribute of the underlying carrier element
        :param power: bool to get the number of division signs of the most common power quantity (energy quantity divided by time)
        :return: number of division signs of the carriers most common energy/power unit
        """
        energy_units = {}
        time_base_unit = [key for key, value in self.base_units.items() if value == "[time]"][0]
        for attribute_name, unit_specs in carrier_units.items():
            energy_unit = self._remove_non_energy_units(unit_specs, attribute_name)
            if power:
                energy_unit[attribute_name] = energy_unit[attribute_name] / self.ureg(time_base_unit)
            energy_units.update(energy_unit)
        energy_unit_ref_carrier = self.get_most_often_appearing_energy_unit(energy_units)
        return len(str(energy_unit_ref_carrier.units).split("/")) - 1

    def _remove_non_energy_units(self, unit_specs, attribute_name):
        """Removes all non-energy dimensions from unit by multiplication/division

        :param unit_specs: dict containing unit category and unit as pint Quantity in base units
        :param attribute_name: name of attribute whose unit is reduced to energy unit
        :return: dict with attribute name and reduced unit
        """
        # dictionary which assigns unit dimensions to corresponding base unit namings
        distinct_dims = {"money": "[currency]", "distance": "[length]", "time": "[time]", "emissions": "[mass]"}
        unit = unit_specs["unit_in_base_units"]
        unit_category = unit_specs["unit_category"]
        for dim, dim_name in distinct_dims.items():
            if dim in unit_category:
                dim_unit = [key for key, value in self.base_units.items() if value == dim_name][0]
                if dim == "time" and "energy_quantity" in unit_category:
                    unit = unit / self.ureg(dim_unit) ** (-1 * unit_category["energy_quantity"])
                else:
                    unit = unit / self.ureg(dim_unit) ** unit_category[dim]
        if "energy_quantity" in unit_category:
            unit = unit ** unit_category["energy_quantity"]
        return {attribute_name: unit}

    def save_carrier_energy_quantities(self, optimization_setup):
        """
        saves energy_quantity units of carriers after consistency checks in order to assign units to the variables later on

        :param optimization_setup: optimization setup object
        :return: dict of carrier units
        """
        for carrier in optimization_setup.dict_elements["Carrier"]:
            self.carrier_energy_quantities[carrier.name] = self._remove_non_energy_units(carrier.units["demand"], attribute_name=None)[None]

    def set_base_unit_combination(self, input_unit, attribute):
        """ converts the input unit to the corresponding base unit

        :param input_unit: unit of input
        :param attribute: name of attribute
        """
        # TODO combine overlap with get_unit_multiplier
        # if input unit is already in base units --> the input unit is base unit
        if input_unit in self.base_units:
            _, base_unit_combination = self.calculate_combined_unit(input_unit, return_combination=True)
        # if input unit is nan --> dimensionless old definition
        elif type(input_unit) != str and np.isnan(input_unit):
            base_unit_combination = pd.Series(index=self.dim_matrix.columns, data=0)
        else:
            # convert to string
            input_unit = str(input_unit)
            # if input unit is 1 --> dimensionless new definition
            if input_unit == "1":
                return 1
            _, base_unit_combination = self.calculate_combined_unit(input_unit, return_combination=True)
        if (base_unit_combination != 0).any():
            self.dict_attribute_values[attribute] = {"base_combination": base_unit_combination, "values": None}

    def set_attribute_values(self, df_output, attribute):
        """ saves the attributes values of an attribute

        :param df_output: output dataframe
        :param attribute: attribute name
        """
        if attribute in self.dict_attribute_values.keys():
            self.dict_attribute_values[attribute]["values"] = df_output

    def recommend_base_units(self, immutable_unit, unit_exps):
        """ gets the best base units based on the input parameter values

        :param immutable_unit: base units which must not be changed to recommend a better set of base units
        :param unit_exps: exponent range inbetween which the base units can be scaled by 10^exponent
        """
        logging.info(f"Check for best base unit combination between 10^{unit_exps['min']} and 10^{unit_exps['max']}")
        dict_values = {}
        dict_units = {}
        base_units = self.dim_matrix.columns.copy()
        for item in self.dict_attribute_values:
            if self.dict_attribute_values[item]["values"] is not None:
                _df_values_temp = self.dict_attribute_values[item]["values"].reset_index(drop=True)
                _df_units_temp = pd.DataFrame(index=_df_values_temp.index, columns=base_units)
                _df_units_temp.loc[_df_values_temp.index, :] = self.dict_attribute_values[item]["base_combination"][base_units].values
                dict_values[item] = _df_values_temp
                dict_units[item] = _df_units_temp
        df_values = pd.concat(dict_values, ignore_index=True).abs()
        df_units = pd.concat(dict_units, ignore_index=True)
        mutable_unit = self.dim_matrix.columns[self.dim_matrix.columns.isin(base_units.difference(immutable_unit))]
        df_units = df_units.loc[:, mutable_unit].values

        # remove rows of df_units which contain only zeros since they cannot be scaled anyway and may influence minimization convergence
        zero_rows_mask = np.all(df_units == 0, axis=1)
        A = df_units[~zero_rows_mask]
        b = df_values[~zero_rows_mask]

        def fun_LSE(x):
            """
            function to compute the least square error of the individual coefficients compared to their mean value

            :param x: array of exponents the coefficients get scaled with (b_tilde = b * 10^(A*x))
            :return: square error evaluated at x
            """
            b_tilde_log = np.log10(b) - np.dot(A, x)
            b_avg = b.sum() / b.size
            return ((b_tilde_log - np.log10(b_avg)) ** 2).sum()

        x0 = np.ones(A.shape[1])
        result = sp.optimize.minimize(fun_LSE, x0, method='L-BFGS-B', bounds=[(unit_exps["min"], unit_exps["max"]) for i in range(df_units.shape[1])])

        if not result.success:
            logging.info(f"Minimization for better base units was not successful, initial base units will therefore be used.")

        #cast solution array to integers since base units should be scaled by factors of 10, 100, etc.
        x_int = result.x.astype(int)

        lse_initial_base_units = fun_LSE(np.zeros(df_units.shape[1]))
        lse = fun_LSE(x_int)
        if lse >= lse_initial_base_units:
            logging.info("The current base unit setting is the best in the given search interval")
        else:
            list_units = []
            for exp, unit in zip(x_int, mutable_unit):
                if exp != 0:
                    list_units.append(str(self.ureg(f"{10.0 ** exp} {unit}").to_compact()))
            logging.info(f"A better base unit combination is {', '.join(list_units)}. This reduces the square error of the coefficients compared to their mean by {'{:e}'.format(lse_initial_base_units-lse)}")

    def check_if_invalid_hourstring(self, input_unit):
        """
        checks if "h" and thus "planck_constant" in input_unit

        :param input_unit: string of input_unit
        """
        _tuple_units = self.ureg(input_unit).to_tuple()[1]
        _list_units = [_item[0] for _item in _tuple_units]
        assert "planck_constant" not in _list_units, f"Error in input unit '{input_unit}'. Did you want to define hour? Use 'hour' instead of 'h' ('h' is interpreted as the planck constant)"

    def define_ton_as_metric(self):
        """ redefines the "ton" as a metric ton """
        self.ureg.define("ton = metric_ton")

    def redefine_standard_units(self):
        """ defines the standard units always required in ZEN and removes the rounding error for leap years."""
        self.ureg.define("Euro = [currency] = EURO = Eur = €")
        self.ureg.define("year = 365 * day = a = yr = julian_year")
        self.ureg.define("ton = metric_ton")

    @staticmethod
    def check_pos_neg_boolean(array, axis=None):
        """ checks if the array has only positive or negative booleans (-1,0,1)

        :param array: numeric numpy array
        :param axis: axis of dataframe
        :return is_pos_neg_boolean """
        if axis:
            is_pos_neg_boolean = np.apply_along_axis(lambda row: np.array_equal(np.abs(row), np.abs(row).astype(bool)), 1, array).any()
        else:
            is_pos_neg_boolean = np.array_equal(np.abs(array), np.abs(array).astype(bool))
        return is_pos_neg_boolean


#ToDo get rid of A matrix dependency -> for big models slowest part; can we use the data structure of linopy directly to determine column and row scaling factors
#ToDo slight numerical errors after rescaling -> dependent on solver -> for gurobi very accurate
class Scaling:
    """
    This class scales the optimization model before solving it and rescales the solution
    """
    def __init__(self, model, algorithm=["geom"], include_rhs = True):
        #optimization model to perform scaling on
        self.model = model
        self.algorithm = algorithm
        self.include_rhs = include_rhs

    def initiate_A_matrix(self):
        self.A_matrix = self.model.constraints.to_matrix(filter_missings=False)
        self.A_matrix_copy = self.A_matrix.copy() #necessary for printing of numerics
        self.D_r_inv = np.ones(self.A_matrix.get_shape()[0])
        self.D_c_inv = np.ones(self.A_matrix.get_shape()[1])
        self.rhs = []
        for name in self.model.constraints:
            constraint = self.model.constraints[name]
            labels = constraint.labels.data
            mask = np.atleast_1d(labels != -2).nonzero()
            try:
                self.rhs += constraint.rhs.data[mask].tolist()
            except:
                self.rhs += [constraint.rhs.data]
        self.rhs = np.array(self.rhs) #np.abs(np.array(self.rhs)) -> could get rid of all the other np.ads in iter_sclaing() etc. but then print numerics only includes absolute values
        self.rhs[self.rhs == np.inf] = 0
        self.rhs_copy = self.rhs.copy() #necessary for printing of numerics

    def re_scale(self):
        model = self.model
        for name_var in model.variables:
            var = model.variables[name_var]
            mask = np.where(var.labels.data != -1)
            var.solution.data[mask] = var.solution.data[mask] * (self.D_c_inv[var.labels.data[mask]])

    def analyze_numerics(self):
        #print numerics if no scaling is activated
        self.initiate_A_matrix()
        self.print_numerics(0,True)

    def run_scaling(self):
        #cp = cProfile.Profile()
        #cp.enable()
        logging.info(f"\n--- Start Scaling ---\n")
        t0 = time.perf_counter()
        self.initiate_A_matrix()
        self.iter_scaling()
        self.overwrite_problem()
        t1 = time.perf_counter()
        logging.info(f"\nTime to Scale Problem: {t1 - t0:0.1f} seconds\n")
        #cp.disable()
        #cp.print_stats("cumtime")

    def replace_data(self, name):
        constraint = self.model.constraints[name]
        #Get data
        lhs = constraint.coeffs.data
        mask_skip_constraints = constraint.labels.data
        mask_variables = constraint.vars.data
        rhs = constraint.rhs.data
        # Find the indices where constraint_mask is not equal to -1
        indices = np.atleast_1d(mask_skip_constraints != -1).nonzero()
        if indices[0].size > 0:
            # Update rhs
            try:
                rhs[indices] = rhs[indices] * self.D_r_inv[mask_skip_constraints[indices]]
            except IndexError:
                constraint.rhs.data = rhs * self.D_r_inv[mask_skip_constraints]
            # Update lhs
            non_nan_mask = ~np.isnan(lhs)
            entries_to_overwrite = np.where(non_nan_mask & (mask_variables != -1))
            lhs[entries_to_overwrite] *= (self.D_r_inv[mask_skip_constraints[entries_to_overwrite[:-1]]] *
                                        self.D_c_inv[mask_variables[entries_to_overwrite]])

    def adjust_upper_lower_bounds_variables(self):
        vars = self.model.variables
        for var in vars:
            mask = np.where(vars[var].labels.data != -1)
            scaling_factors = self.D_c_inv[vars[var].labels.data[mask]]
            vars[var].upper.data[mask] = vars[var].upper.data[mask] * scaling_factors**(-1)
            vars[var].lower.data[mask] = vars[var].lower.data[mask] * scaling_factors**(-1)

    def adjust_scaling_factors_of_skipped_rows(self, name):
        constraint = self.model.constraints[name]
        #rows -> unnecessary to adjust scaling factor of rows with binary and integer variables as skipped anyways
        #cols
        mask_variables = constraint.vars.data
        indices = np.where(mask_variables != -1)
        self.D_c_inv[mask_variables[indices]] = 1

    def adjust_int_variables(self):
        vars = self.model.variables
        for var in vars:
            if vars[var].attrs['binary'] or vars[var].attrs['integer']:
                mask = np.where(vars[var].labels.data != -1)
                self.D_c_inv[vars[var].labels.data[mask]] = 1

    def overwrite_problem(self):
        #pre-check variables -> skip binary and integer variables
        self.adjust_int_variables()
        #adjust scaling factors that have inf or nan values -> not really necessary anymore but might be a good security check
        self.D_c_inv[self.D_c_inv == np.inf] = 1
        self.D_r_inv[self.D_r_inv == np.inf] = 1
        self.D_c_inv = np.nan_to_num(self.D_c_inv, nan=1)
        self.D_r_inv = np.nan_to_num(self.D_r_inv, nan=1)
        #pre-check rows -> otherwise inconsistency in scaling
        for name_con in self.model.constraints:
            if self.model.constraints[name_con].coeffs.dtype == int:
                self.adjust_scaling_factors_of_skipped_rows(name_con)
        self.print_numerics_of_last_iteration()
        #Include adjust upper/lower bounds of variables that are scaled
        self.adjust_upper_lower_bounds_variables()
        #overwrite constraints
        for name_con in self.model.constraints:
            #overwrite data
            #check if only integers are allowed in scaling: if yes skip and overwrite scaling vector
            if self.model.constraints[name_con].coeffs.dtype == int:
                continue
            else:
                self.replace_data(name_con)
        #overwrite objective
        vars = self.model.objective.vars.data
        scale_factors = self.D_c_inv[vars]
        self.model.objective.coeffs.data = self.model.objective.coeffs.data * scale_factors

    def get_min(self,A_matrix):
        d = A_matrix.data
        try:
            mins_values = np.minimum.reduceat(np.abs(d), A_matrix.indptr[:-1])
        except: #necessary if multiple columns and rows at the end of the matrix without entries -> if not only last entry of indptr is len(data) and therefore out of range
            last_empty_entries = A_matrix.indptr[A_matrix.indptr == len(d)]
            non_empty_entries = A_matrix.indptr[A_matrix.indptr < len(d)]
            mins_values = np.minimum.reduceat(np.abs(d), non_empty_entries)
            mins_values = np.hstack((mins_values,np.ones((len(last_empty_entries)-1,))))
        return mins_values

    def get_full_geom(self,A_matrix,axis): #Very slow and less effective than simplified geom norm
        d = A_matrix.data
        geom = np.ones(len(A_matrix.indptr)-1)
        nonzero_entries = np.unique(list(A_matrix.nonzero()[axis]))
        idx_unique = np.unique(A_matrix.indptr[:-1])
        d_slices = np.split(d,idx_unique[1:])
        geom[nonzero_entries] = list(map(lambda x: sp.stats.gmean(np.abs(x)),d_slices))
        return geom

    def update_A(self, vector, axis):
        if axis == 1:
            self.A_matrix = sp.sparse.diags(vector, 0, format='csr').dot(self.A_matrix)
            self.D_r_inv = self.D_r_inv * vector
            self.rhs = self.rhs * vector
        elif axis == 0:
            self.A_matrix = self.A_matrix.dot(sp.sparse.diags(vector, 0, format='csr'))
            self.D_c_inv = self.D_c_inv * vector

    def print_numerics_of_last_iteration(self):
        self.A_matrix =  sp.sparse.diags(self.D_r_inv, 0, format='csr').dot(self.A_matrix_copy).dot(sp.sparse.diags(self.D_c_inv, 0, format='csr'))
        self.rhs = self.rhs_copy * self.D_r_inv
        self.print_numerics(len(self.algorithm))

    def generate_numerics_string(self,label,index=None,A_matrix=None,var=None, is_rhs=False):
        if is_rhs:
            cons_str = self.model.constraints.get_label_position(label)
            cons_str = cons_str[0] + str(list(cons_str[1].values()))
            return f"{self.rhs[label]} in {cons_str}"
        else:
            cons_str = self.model.constraints.get_label_position(label)
            cons_str = cons_str[0] + str(list(cons_str[1].values()))
            var_str = self.model.variables.get_label_position(var)
            var_str = var_str[0] + str(list(var_str[1].values()))
            return f"{A_matrix[index]} {var_str} in {cons_str}"

    def print_numerics(self,i,no_scaling = False):
        data_coo = self.A_matrix.tocoo()
        A_abs = np.abs(data_coo.data)
        index_max = np.argmax(A_abs)
        index_min = np.argmin(A_abs)
        row_max = data_coo.row[index_max]
        col_max = data_coo.col[index_max]
        row_min = data_coo.row[index_min]
        col_min = data_coo.col[index_min]
        rhs_max_index = np.where(np.abs(self.rhs) == np.max(np.abs(self.rhs)[self.rhs != np.inf]))[0][0]
        rhs_min_index = np.where(np.abs(self.rhs) == np.min(np.abs(self.rhs)[np.abs(self.rhs) > 0]))[0][0]
        #Max Matrix String
        cons_str_max = self.generate_numerics_string(row_max, index=index_max,A_matrix=data_coo.data,var=col_max)
        #Min Matrix String
        cons_str_min = self.generate_numerics_string(row_min, index=index_min,A_matrix=data_coo.data,var=col_min)
        #RHS values
        cons_rhs_max = self.generate_numerics_string(rhs_max_index, is_rhs=True)
        cons_rhs_min = self.generate_numerics_string(rhs_min_index, is_rhs=True)
        #Prints
        if no_scaling:
            logging.info(f"\n--- Analyze Numerics ---\n")
        else:
            logging.info(f"\n--- Numerics at iteration {i} ---\n")
        print("Max value of A matrix: " + cons_str_max)
        print("Min value of A matrix: " + cons_str_min)
        print("Max value of RHS: " + cons_rhs_max)
        print("Min value of RHS: " + cons_rhs_min)
        print("Numerical Range:")
        print("LHS : {}".format([format(A_abs[index_min],".1e"),format(A_abs[index_max],".1e")]))
        print("RHS : {}".format([format(np.abs(self.rhs[rhs_min_index]),".1e"),format(np.abs(self.rhs[rhs_max_index]),".1e")]))



    def iter_scaling(self):
        #transform A matrix to csr matrix for better computational properties
        self.A_matrix.eliminate_zeros()
        self.A_matrix = sp.sparse.csr_matrix(self.A_matrix)
        #initiate iteration counter
        i = 0
        self.print_numerics(i)
        for algo in self.algorithm:
            i+=1
            #update row scaling vector
            if algo == "infnorm":
                #update row scaling vector
                max_rows = sp.sparse.linalg.norm(self.A_matrix, ord=np.inf, axis=1)
                if self.include_rhs:
                    max_rows = np.maximum(max_rows, np.abs(self.rhs), out=max_rows, where=self.rhs != np.inf)
                max_rows[max_rows == 0] = 1 #to avoid warning outputs
                r_vector = 1 / max_rows
                r_vector = np.power(2, np.round(np.emath.logn(2, r_vector)))
                #update A and row scaling matrix
                self.update_A(r_vector,1)
                #update column scaling vector
                max_cols = sp.sparse.linalg.norm(self.A_matrix, ord=np.inf, axis=0)
                max_cols[max_cols == 0] = 1 #to avoid warning outputs
                c_vector = 1/max_cols
                c_vector = np.power(2, np.round(np.emath.logn(2, c_vector)))
                #update A and column scaling matrix
                self.update_A(c_vector,0)
                # Print Numerics
                if i < len(self.algorithm):
                    self.print_numerics(i)

            #ToDo add rhs to row scaling
            elif algo == "full_geom":
                #update row scaling vector
                geom = self.get_full_geom(self.A_matrix, 0)
                r_vector = 1 / geom
                r_vector = np.power(2, np.round(np.emath.logn(2, r_vector)))
                #update A and row scaling matrix
                self.update_A(r_vector,1)
                #update column scaling vector
                geom = self.get_full_geom(self.A_matrix.tocsc(), 1)
                c_vector = 1 / geom
                c_vector = np.power(2, np.round(np.emath.logn(2, c_vector)))
                #update A and column scaling matrix
                self.update_A(c_vector,0)
                # Print Numerics
                if i < len(self.algorithm):
                    self.print_numerics(i)

            elif algo == "geom":
                # update row scaling vector
                max_rows = sp.sparse.linalg.norm(self.A_matrix, ord=np.inf, axis=1)
                min_rows = self.get_min(self.A_matrix)
                if self.include_rhs:
                    max_rows = np.maximum(max_rows, np.abs(self.rhs), out=max_rows, where=self.rhs != np.inf)
                    min_rows = np.minimum(min_rows,np.abs(self.rhs),out =min_rows, where=np.abs(self.rhs)>0)
                geom = (max_rows * min_rows) ** 0.5
                geom [geom == 0] = 1 #to avoid warning outputs
                r_vector = 1 / geom
                r_vector = np.power(2, np.round(np.emath.logn(2, r_vector)))
                # update A and row scaling matrix
                self.update_A(r_vector,1)
                # update column scaling vector
                max_cols = sp.sparse.linalg.norm(self.A_matrix, ord=np.inf, axis=0)
                min_cols = self.get_min(self.A_matrix.tocsc())
                geom = (max_cols * min_cols) ** 0.5
                geom[geom == 0] = 1 #to avoid warning outputs
                c_vector = 1 / geom
                c_vector = np.power(2, np.round(np.emath.logn(2, c_vector)))
                # update A and column scaling matrix
                self.update_A(c_vector,0)
                #Print Numerics
                if i < len(self.algorithm):
                    self.print_numerics(i)

            elif algo == "arithm":
                #update row scaling vector
                mean_rows = sp.sparse.linalg.norm(self.A_matrix, ord=1, axis=1)/(np.diff(self.A_matrix.indptr)+np.ones(self.A_matrix.get_shape()[0]))
                if self.include_rhs:
                    mean_rows = mean_rows + np.abs(self.rhs)/(np.diff(self.A_matrix.indptr)+np.ones(self.A_matrix.get_shape()[0]))
                c_vector = 1/mean_rows
                c_vector = np.power(2, np.round(np.emath.logn(2, c_vector)))
                #update A and row scaling matrix
                self.update_A(c_vector,1)
                #update column scaling vector
                mean_cols = sp.sparse.linalg.norm(self.A_matrix, ord=1, axis=0)/np.diff(self.A_matrix.tocsc().indptr)
                r_vector = 1/mean_cols
                r_vector = np.power(2, np.round(np.emath.logn(2, r_vector)))
                #update A and column scaling matrix
                self.update_A(r_vector,0)
                # Print Numerics
                if i < len(self.algorithm):
                    self.print_numerics(i)




