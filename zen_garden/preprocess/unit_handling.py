"""File which contains the unit handling and scaling class."""

import itertools
import json
import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy as sp
from pint import UnitRegistry
from pint.util import column_echelon_form

from zen_garden.elements.carrier import Carrier
from zen_garden.elements.technology import Technology
from zen_garden.model.config import Config

if TYPE_CHECKING:
    from zen_garden.elements.energy_system import EnergySystem
    from zen_garden.services.element_registry import ElementRegistry

logger = logging.getLogger(__name__)


class UnitHandling:
    """A class for managing and converting units in an energy system model.

    This class facilitates unit consistency checks, dimensionality analysis, and
    unit conversions in energy systems models, particularly those that involve
    energy carriers, technologies, and conversion processes. It helps in
    defining and converting units across various parameters and ensures that
    unit definitions are consistent across the entire system.

    Key functionalities:
        - Loading and extracting base units for the system.
        - Converting input units into a unified system of base units.
        - Checking for dimensional consistency between input units and
          base units.
        - Redefining and verifying the dimensional matrix of the system.
        - Ensuring that unit conversions and combinations are performed
          accurately.
    """

    def __init__(self, folder_path, rounding_decimal_points_units):
        """Initializes an instance of the UnitHandling class.

        This constructor processes and stores the system's base unit definitions
        and other configurations. It also defines the rounding tolerance for
        unit conversions.

        Args:
            folder_path (str or Path): The path to the folder containing system
                specifications (e.g., "unit_definitions.txt", "base_units.csv").
            rounding_decimal_points_units (int): The number of decimal points to
                which units should be rounded during conversion and consistency
                checks.
        """
        self.folder_path = folder_path
        self.rounding_decimal_points_units = rounding_decimal_points_units
        self.get_base_units()
        # dict of element attribute values
        self.dict_attribute_values = {}
        self.carrier_energy_quantities = {}

    def get_base_units(self):
        """Extracts and initializes the base units of the energy system.

        This method loads unit definitions, processes them to extract base
        units, and constructs the dimensionality matrix for the system.
        It also checks for duplicates and verifies that no unit can be
        constructed from other base units. Additionally, it ensures that all
        base units have a valid dimensionality and that no linear dependencies
        exist between them.

        Raises:
            KeyError: If there are multiple base units defined for the same
                dimensionality.
            AssertionError: If there are linear dependencies between base units
                that can't be resolved.
        """
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
            self.dim_matrix.loc[base_unit, list(dim_unit.keys())] = list(
                dim_unit.values()
            )
        self.dim_matrix = self.dim_matrix.fillna(0).astype(int).T

        # check if unit defined twice or more
        duplicate_units = self.dim_matrix.T.duplicated()
        if duplicate_units.any():
            dim_matrix_duplicate = self.dim_matrix.loc[:, duplicate_units]
            for duplicate in dim_matrix_duplicate:
                # if same unit twice (same order of magnitude and same dimensionality)
                if len(self.dim_matrix[duplicate].shape) > 1:
                    logger.warning(
                        f"The base unit <{duplicate}> was defined more than once. "
                        "Duplicates are dropped."
                    )
                    _duplicateDim = self.dim_matrix[duplicate].T.drop_duplicates().T
                    self.dim_matrix = self.dim_matrix.drop(duplicate, axis=1)
                    self.dim_matrix[duplicate] = _duplicateDim
                else:
                    raise KeyError(
                        f"More than one base unit defined for dimensionality "
                        f"{self.base_units[duplicate]} (e.g., {duplicate})"
                    )
        # get linearly dependent units
        M, I_Mat, pivot = column_echelon_form(np.array(self.dim_matrix), ntype=float)
        M = np.array(M).squeeze()
        I_Mat = np.array(I_Mat).squeeze()
        pivot = np.array(pivot).squeeze()
        # index of linearly dependent units in M and I
        idx_lin_dep = np.squeeze(np.argwhere(np.all(M == 0, axis=1)))
        # index of linearly dependent units in dimensionality matrix
        _idx_pivot = range(len(self.base_units))
        idx_lin_dep_dim_matrix = list(set(_idx_pivot).difference(pivot))
        self.dim_analysis = {}
        self.dim_analysis["dependent_units"] = self.dim_matrix.columns[
            idx_lin_dep_dim_matrix
        ]
        dependent_dims = I_Mat[idx_lin_dep, :]
        # if only one dependent unit
        if len(self.dim_analysis["dependent_units"]) == 1:
            dependent_dims = dependent_dims.reshape(1, dependent_dims.size)
        # reorder dependent dims to match dependent units
        dim_of_dependent_units = dependent_dims[:, idx_lin_dep_dim_matrix]
        # if not already in correct order (ones on the diagonal of dependent_dims)
        if not np.all(np.diag(dim_of_dependent_units) == 1):
            # get position of ones in dim_of_dependent_units
            pos_ones = np.argwhere(dim_of_dependent_units == 1)
            assert np.size(pos_ones, axis=0) == len(
                self.dim_analysis["dependent_units"]
            ), (
                f"Cannot determine order of dependent base units "
                f"{self.dim_analysis['dependent_units']}, "
                f"because diagonal of dimensions of the dependent units cannot "
                "be determined."
            )
            # pivot dependent dims
            dependent_dims = dependent_dims[pos_ones[:, 1], :]
        self.dim_analysis["dependent_dims"] = dependent_dims
        # check that no base unit can be directly constructed from the others
        # (e.g., GJ from GW and hour)
        assert not UnitHandling.check_pos_neg_boolean(dependent_dims, axis=1), (
            f"At least one of the base units {list(self.base_units.keys())} "
            "can be directly constructed from the others"
        )

    def extract_base_units(self):
        """Extracts the base units from either a CSV or JSON file.

        If the CSV file (``base_units.csv``) is not found, the method will
        fall back on a JSON file (``base_units.json``) to load the base units.
        If ``hour`` is not found in the list of base units, a warning will
        be raised. This method provides the list of all base units that will be
        used for further calculations and unit consistency checks.

        Returns:
            list:
                A list of base units defined in the system.

        Raises:
            UserWarning: If the hour unit is not found in the base unit
            definitions.
        """
        if os.path.exists(os.path.join(self.folder_path / "base_units.csv")):
            list_base_units = (
                pd.read_csv(self.folder_path / "base_units.csv")
                .squeeze()
                .values.tolist()
            )
            logger.warning(
                "DeprecationWarning: Specifying the base units in .csv file "
                "format is deprecated. Use the .json file format instead."
            )
        else:
            with open(os.path.join(self.folder_path, "base_units.json"), "r") as f:
                data = json.load(f)
            list_base_units = data["unit"]
        if "hour" not in list_base_units:
            warnings.warn(
                "The base unit for time is intended to be `hour` but is not "
                "found in the base_units file."
                "If this is intentional, make sure that your settings and "
                "input data are aligned with this change.",
                UserWarning,
                stacklevel=2,
            )
        return list_base_units

    def calculate_combined_unit(self, input_unit):
        """Represents the input unit as a combination of base units.

        This method constructs a combined unit by converting an input unit into
        a set of base units. It first checks the dimensionality of the input
        unit and constructs the appropriate combined unit through dimensional
        analysis. It then checks for unit consistency with the base units
        and returns the combined unit.

        Args:
            input_unit (str): The input unit to be converted (e.g.,
                ``kg``, ``m/s``).
            return_combination (bool): If True, also returns the base unit
                combination, in addition to the combined unit.

        Returns:
            pint.Quantity or tuple:
                The combined unit represented as a ``pint.Quantity``. If
                ``return_combination=True``, returns a tuple containing the
                combined unit and the base unit combination.

        Raises:
            AssertionError: If the dimensionality of the input unit cannot be
                matched with base units.
        """
        # check if "h" and thus "planck_constant" in unit
        self.check_if_invalid_hourstring(input_unit)
        # create dimensionality vector for input_unit
        dim_input = self.ureg.get_dimensionality(self.ureg(input_unit))
        dim_vector = pd.Series(index=self.dim_matrix.index, data=0)
        missing_dim = set(dim_input.keys()).difference(dim_vector.keys())
        assert (
            len(missing_dim) == 0
        ), f"No base unit defined for dimensionalities <{missing_dim}>"
        if len(dim_input) > 0:  # check for content of dim_input to avoid Warning
            dim_vector[list(dim_input.keys())] = list(dim_input.values())
        # calculate dimensionless combined unit (e.g., tons and kilotons)
        combined_unit = self.ureg(input_unit).units
        # if unit (with a different multiplier) is already in base units
        if self.dim_matrix.isin(dim_vector).all(axis=0).any():
            base_combination = self.dim_matrix.isin(dim_vector).all(axis=0).astype(int)
            base_unit = self.ureg(
                self.dim_matrix.columns[self.dim_matrix.isin(dim_vector).all(axis=0)][0]
            )
            combined_unit *= base_unit ** (-1)
        # if inverse of unit (with a different multiplier) is already in base
        # units (e.g. 1/km and km)
        elif (self.dim_matrix * -1).isin(dim_vector).all(axis=0).any():
            base_combination = (self.dim_matrix * -1).isin(dim_vector).all(
                axis=0
            ).astype(int) * (-1)
            base_unit = self.ureg(
                self.dim_matrix.columns[
                    (self.dim_matrix * -1).isin(dim_vector).all(axis=0)
                ][0]
            )
            combined_unit *= base_unit
        else:
            # drop dependent units
            dim_matrix_reduced = self.dim_matrix.drop(
                self.dim_analysis["dependent_units"], axis=1
            )
            # solve system of linear equations
            combination_solution = np.linalg.solve(dim_matrix_reduced, dim_vector)
            # check if only -1, 0, 1
            if UnitHandling.check_pos_neg_boolean(combination_solution):
                base_combination = pd.Series(index=self.dim_matrix.columns, data=0)
                base_combination[dim_matrix_reduced.columns] = combination_solution
                # compose relevant units to dimensionless combined unit
                for unit, power in zip(
                    dim_matrix_reduced.columns, combination_solution, strict=False
                ):
                    combined_unit *= self.ureg(unit) ** (-1 * power)
            else:
                base_combination, combined_unit = (
                    self._get_combined_unit_of_different_matrix(
                        dim_matrix_reduced=dim_matrix_reduced,
                        dim_vector=dim_vector,
                        input_unit=input_unit,
                    )
                )
        return combined_unit, base_combination

    def _get_combined_unit_of_different_matrix(
        self, dim_matrix_reduced, dim_vector, input_unit
    ):
        """Calculates the combined unit for a different dimensionality matrix.
        We substitute base units by the dependent units and try again.
        If the matrix is singular we solve the overdetermined problem.

        :param dim_matrix_reduced: dimensionality matrix without dependent units
        :param dim_vector: dimensionality vector of input unit
        :param input_unit: input unit
        :return: base_combination: base combination of input unit
        :return: combined_unit: input unit expressed in base units
        """
        calculated_multiplier = False
        combined_unit = self.ureg(input_unit).units
        base_combination = pd.Series(index=self.dim_matrix.columns, data=0)
        # try to substitute unit by a dependent unit
        for unit_combination in itertools.combinations(
            self.dim_matrix.columns, len(self.dim_matrix.index)
        ):
            if (
                not calculated_multiplier
                and len(
                    set(unit_combination).difference(set(dim_matrix_reduced.columns))
                )
                != 0
            ):
                # use reduced matrix based on the unit_combination
                dim_matrix_reduced_temp = self.dim_matrix.loc[:, unit_combination]
                # if full rank
                if np.linalg.matrix_rank(dim_matrix_reduced_temp) == np.size(
                    dim_matrix_reduced_temp, 1
                ):
                    combination_solution_temp = np.linalg.solve(
                        dim_matrix_reduced_temp, dim_vector
                    )
                # if singular, check if zero row in matrix corresponds to zero row in
                # unit dimensionality
                else:
                    zero_row = dim_matrix_reduced_temp.index[
                        ~dim_matrix_reduced_temp.any(axis=1)
                    ]
                    if (dim_vector[zero_row] == 0).all():
                        # remove zero row
                        dim_matrix_reduced_temp_reduced = dim_matrix_reduced_temp.drop(
                            zero_row, axis=0
                        )
                        dim_vector_reduced = dim_vector.drop(zero_row, axis=0)
                        # formulate as optimization problem with 1,-1 bounds
                        # to determine solution of overdetermined matrix
                        ub = np.array(
                            [1] * len(dim_matrix_reduced_temp_reduced.columns)
                        )
                        lb = np.array(
                            [-1] * len(dim_matrix_reduced_temp_reduced.columns)
                        )
                        res = sp.optimize.lsq_linear(
                            dim_matrix_reduced_temp_reduced,
                            dim_vector_reduced,
                            bounds=(lb, ub),
                        )
                        # if an exact solution is found (after rounding)
                        if np.round(res.cost, 4) == 0:
                            combination_solution_temp = np.round(res.x, 4)
                        # if not solution is found
                        else:
                            continue
                    # definitely not a solution because zero row corresponds to nonzero
                    # dimensionality
                    else:
                        continue
                if UnitHandling.check_pos_neg_boolean(combination_solution_temp):
                    # compose relevant units to dimensionless combined unit
                    base_combination[dim_matrix_reduced_temp.columns] = (
                        combination_solution_temp
                    )
                    for unit_temp, power_temp in zip(
                        dim_matrix_reduced_temp.columns,
                        combination_solution_temp,
                        strict=False,
                    ):
                        combined_unit *= self.ureg(unit_temp) ** (-1 * power_temp)
                    calculated_multiplier = True
                    break
        assert calculated_multiplier, (
            f"Cannot establish base unit conversion for {input_unit} from base "
            f"units {self.base_units.keys()}"
        )
        return base_combination, combined_unit

    # ToDo: check if combined_unit is described correctly in the header
    def get_unit_multiplier(
        self, input_unit, attribute_name, path=None, combined_unit=None
    ):
        """Calculates the multiplier for converting an input unit into the base
        units.

        This method computes the scaling factor (multiplier) needed to convert
        the given `input_unit` into a base unit. If the `input_unit` is already
        a base unit, the multiplier is 1. If the `input_unit` is not in base
        units, it computes the conversion using dimensional analysis and ensures
        that the resulting multiplier meets the rounding tolerance.

        Args:
            input_unit (str): The unit to be converted (e.g., "kg", "m/s").
            attribute_name (str): The name of the attribute that this unit
                corresponds to.
            path (str, optional): The file path associated with the unit
                (for logging purposes).
            combined_unit (pint.Quantity, optional): The combined unit in
                base units. If provided, skips recomputing the combined unit.

        Returns:
            float:
                The multiplier that scales the `input_unit` into the
                base units.

        Raises:
            AssertionError: If the multiplier is smaller than the rounding
                tolerance.
        """
        # if input unit is already in base units --> the input unit is base
        # unit, multiplier = 1
        if input_unit in self.base_units:
            return 1
        # if input unit is nan --> dimensionless old definition
        elif type(input_unit) is not str and np.isnan(input_unit):
            logger.warning(
                f"DeprecationWarning: Parameter {attribute_name} of "
                f"{Path(path).name} has no unit (assign unit '1' to unitless "
                "parameters)"
            )
            return 1
        else:
            # convert to string
            input_unit = str(input_unit)
            # if input unit is 1 --> dimensionless new definition
            if input_unit == "1":
                return 1
            if not combined_unit:
                combined_unit, _ = self.calculate_combined_unit(input_unit)
            assert combined_unit.to_base_units().unitless, (
                f"The unit conversion of unit {input_unit} did not "
                "resolve to a dimensionless conversion factor. "
                "Something went wrong."
            )
            # magnitude of combined unit is multiplier
            multiplier = combined_unit.to_base_units().magnitude
            # check that multiplier is larger than rounding tolerance
            assert multiplier >= 10 ** (-self.rounding_decimal_points_units), (
                f"Multiplier {multiplier} of unit {input_unit} in parameter "
                f"{attribute_name} is smaller than rounding tolerance "
                f"{10 ** (-self.rounding_decimal_points_units)}"
            )
            # round to decimal points
            return round(multiplier, self.rounding_decimal_points_units)

    def convert_unit_into_base_units(
        self, input_unit, get_multiplier=False, attribute_name=None, path=None
    ):
        """Converts an input unit into base units.

        This method converts an input unit into the equivalent base units,
        following the dimensional analysis process to express the `input_unit`
        as a combination of base units. Additionally, it can return the
        multiplier that scales the input unit into the base units, depending on
        the value of `get_multiplier`.

        Args:
            input_unit (str): The unit to be converted (e.g., "kg", "m/s").
            attribute_name (str, optional): The name of the attribute
                corresponding to the unit.
            path (str, optional): The file path of the attribute for
                logging purposes.
            get_multiplier (bool, optional): Whether to return the multiplier
                for the conversion. If False, returns the base unit combination.

        Returns:
            pint.Quantity or tuple:
                If `get_multiplier` is False, returns the
                `input_unit` converted to base units as a `pint.Quantity`.
                If `get_multiplier` is True, returns the multiplier as a float
                and the base units as a `pint.Quantity`.
        """
        # convert attribute unit into unit combination of base units
        combined_unit = None
        attribute_unit_in_base_units = self.ureg("")
        if input_unit != "1" and not pd.isna(input_unit):
            combined_unit, base_combination = self.calculate_combined_unit(input_unit)
            for unit, power in zip(
                base_combination.index, base_combination, strict=False
            ):
                attribute_unit_in_base_units *= self.ureg(unit) ** power
        # calculate the multiplier to convert the attribute unit into base units
        if get_multiplier:
            multiplier = self.get_unit_multiplier(
                input_unit, attribute_name, path, combined_unit=combined_unit
            )
            return multiplier, attribute_unit_in_base_units
        else:
            return attribute_unit_in_base_units

    def consistency_checks_input_units(
        self,
        config: Config,
        energy_system: "EnergySystem",
        element_registry: "ElementRegistry",
    ):
        """Performs unit consistency checks on the input data.

        This method checks whether the units of the parameters defined in the
        input CSV files are consistent with the system's dimensional framework.
        It compares units across elements and technologies and ensures that the
        units match the expected dimensional definitions. The check also
        includes units for conversion factors, retrofit flow coupling factors,
        and other related parameters.

        Args:
            config (Config): The configuration object containing settings for
                the optimization, including unit consistency checks.
            energy_system (EnergySystem): The energy system object containing
                information about the overall system, including carriers and
                technologies.

        Raises:
            AssertionError: If unit inconsistencies are found in the input
                files or optimization setup.
        """
        if not config.solver.check_unit_consistency:
            return
        elements = element_registry.all_elements()
        items = elements + [energy_system]
        conversion_factor_units = {}
        retrofit_flow_coupling_factors = {}
        for item in items:
            energy_quantity_units = {}
            unit_dict = item.units
            # since technology elements have a lot of parameters related to
            # their reference carrier, their unit consistency must be checked
            # together (second if for retrofit techs)
            if isinstance(item, Technology):
                reference_carrier = element_registry.get_element(
                    Carrier, item.reference_carrier[0]
                )
                assert reference_carrier is not None
                unit_dict.update(reference_carrier.units)
            # add units of conversion factors/flow coupling factors to carrier
            # units to perform consistency checks (works only since carriers
            # are located at end of ELEMENT_TYPE_CLASSES)
            if isinstance(item, Carrier):
                for tech_name, cf_dict in conversion_factor_units.items():
                    for dependent_carrier, unit_pair in cf_dict.items():
                        units_to_check = [
                            unit for key, unit in unit_pair.items() if key == item.name
                        ]
                        if len(units_to_check) != 0:
                            unit_in_base_units = self.convert_unit_into_base_units(
                                units_to_check[0]
                            )
                            energy_quantity_units.update(
                                {
                                    tech_name
                                    + "_conversion_factor_"
                                    + dependent_carrier: unit_in_base_units
                                }
                            )
                for tech_name, fcf_dict in retrofit_flow_coupling_factors.items():
                    for dependent_carrier, unit_pair in fcf_dict.items():
                        units_to_check = [
                            unit for key, unit in unit_pair.items() if key == item.name
                        ]
                        if len(units_to_check) != 0:
                            unit_in_base_units = self.convert_unit_into_base_units(
                                units_to_check[0]
                            )
                            energy_quantity_units.update(
                                {
                                    tech_name
                                    + "_retrofit_flow_coupling_factor_"
                                    + dependent_carrier: unit_in_base_units
                                }
                            )
            # conduct consistency checks
            for attribute_name, unit_specs in unit_dict.items():
                if attribute_name == "conversion_factor":
                    conversion_factor_units[item.name] = (
                        self._get_conversion_factor_units(
                            item, unit_specs, reference_carrier, elements
                        )
                    )
                elif attribute_name == "retrofit_flow_coupling_factor":
                    base_technology = element_registry.get_element(
                        Technology, item.retrofit_base_technology[0]
                    )
                    reference_carrier = element_registry.get_element(
                        Carrier, base_technology.reference_carrier[0]
                    )
                    retrofit_flow_coupling_factors[item.name] = (
                        self._get_conversion_factor_units(
                            item, unit_specs, reference_carrier, elements
                        )
                    )
                elif unit_specs["unit_category"] == {}:
                    assert unit_specs["unit_in_base_units"] == self.ureg(
                        "dimensionless"
                    ), (
                        f"The attribute {attribute_name} of "
                        f"{item.__class__.__name__} {item.name} is per "
                        "definition dimensionless. However, its unit was "
                        f"defined as {unit_specs['unit_in_base_units']}."
                    )
                # check if nonlinear capex file exists for conversion technology
                # since the units defined there overwrite the attributes file
                # units
                elif attribute_name == "capex_specific_conversion" and hasattr(
                    item, "units_nonlinear_capex_files"
                ):
                    for key, value in item.units_nonlinear_capex_files.items():
                        if "capex" in value:
                            capex_specific_unit = value["capex"].values[0]
                            unit_specs["unit_in_base_units"] = (
                                self.convert_unit_into_base_units(capex_specific_unit)
                            )
                            energy_quantity_units.update(
                                self._remove_non_energy_units(
                                    unit_specs, "capex_" + key
                                )
                            )
                        capacity_unit = value["capacity"].values[0]
                        unit_specs["unit_category"] = [
                            value["unit_category"]
                            for key, value in unit_dict.items()
                            if key == "capacity_limit"
                        ][0]
                        unit_specs["unit_in_base_units"] = (
                            self.convert_unit_into_base_units(capacity_unit)
                        )
                        energy_quantity_units.update(
                            self._remove_non_energy_units(unit_specs, "capacity_" + key)
                        )
                # units of input/output/reference carrier not of interest for
                # consistency
                elif attribute_name not in [
                    "input_carrier",
                    "output_carrier",
                    "reference_carrier",
                ]:
                    energy_quantity_units.update(
                        self._remove_non_energy_units(unit_specs, attribute_name)
                    )
            # remove attributes whose units became dimensionless since they don't
            # have an energy quantity
            energy_quantity_units_check = {
                key: value.to_base_units().units
                for key, value in energy_quantity_units.items()
                if value.to_base_units().units != self.ureg("dimensionless")
            }
            energy_quantity_units = {
                key: value
                for key, value in energy_quantity_units.items()
                if value != self.ureg("dimensionless")
            }
            # check if conversion factor units are consistent
            self._check_for_power_power(
                energy_quantity_units, energy_quantity_units_check
            )
            # check if units are consistent
            self.assert_unit_consistency(
                elements,
                energy_quantity_units,
                energy_quantity_units_check,
                item,
                config,
                reference_carrier.name,
                unit_dict,
            )
        logger.info("Parameter unit consistency is fulfilled!")
        self.save_carrier_energy_quantities(element_registry)

    def _check_for_power_power(
        self, energy_quantity_units, energy_quantity_units_check
    ):
        """Adjusts conversion factors or retrofit flow coupling factor units from
        power/power to energy/energy if needed.

        This helper method tries to resolve unit inconsistencies that might
        arise due to the units of conversion factors or retrofit flow coupling
        factors. If units are inconsistent and involve power terms, the method
        attempts to change the units from "power/power" to "energy/energy" to
        resolve the inconsistency. This is done since both types of units are
        allowed as inputs.

        Args:
            energy_quantity_units (dict): Dictionary containing the energy
                quantity units for each attribute.
            energy_quantity_units_check (dict): Dictionary of energy quantities
                in base units to check for consistency.
        """
        exclude_strings = ["conversion_factor", "retrofit_flow_coupling_factor"]
        if self._is_inconsistent(
            energy_quantity_units_check
        ) and not self._is_inconsistent(
            energy_quantity_units_check, exclude_strings=exclude_strings
        ):
            non_cf_energy_quantity_unit = [
                value
                for key, value in energy_quantity_units.items()
                if all(es not in key for es in exclude_strings)
            ][0]
            cf_energy_quantity_units = {
                key: value
                for key, value in energy_quantity_units.items()
                if any(es in key for es in exclude_strings)
            }
            time_base_unit = [
                key for key, value in self.base_units.items() if value == "[time]"
            ][0]
            for key, value in cf_energy_quantity_units.items():
                # if conversion factor unit is in not in energy units, try to
                # convert it to energy units by multiplying with time base unit
                if value != non_cf_energy_quantity_unit:
                    energy_quantity_units[key] = value * self.ureg(time_base_unit)
                    energy_quantity_units_check[key] = (
                        energy_quantity_units_check[key]
                        * self.ureg(time_base_unit).to_base_units().units
                    )

    def assert_unit_consistency(
        self,
        elements,
        energy_quantity_units,
        energy_quantity_units_check,
        item,
        config: "Config",
        reference_carrier_name,
        unit_dict,
    ):
        """Asserts that the units of the attributes of an element are consistent
        with the system's dimensional framework.

        This method checks if the units of attributes defined in the input
        files (or the optimization setup) are consistent with each other and
        with the base units. It verifies that all the parameters' units
        conform to dimensional analysis and resolves any inconsistencies.

        Args:
            elements (list): List of all elements in the system.
            energy_quantity_units (dict): Dictionary of attribute names and
                their corresponding energy quantity units.
            energy_quantity_units_check (dict): Dictionary of energy quantity
                units in base units for consistency checking.
            item: The specific element or energy system being checked.
            config (Config): The configuration object containing settings for
                the optimization, including unit consistency checks.
            reference_carrier_name (str): The name of the reference carrier
                associated with the element (if applicable).
            unit_dict (dict): Dictionary of unit specifications for attributes.

        Raises:
            AssertionError: If inconsistencies are found in the units of the
                attributes.
        """
        attributes_with_lowest_appearance = (
            self._get_attributes_with_least_often_appearing_unit(energy_quantity_units)
        )
        # assert unit consistency
        if item in elements and self._is_inconsistent(energy_quantity_units_check):
            # check if there is a conversion factor with wrong units
            wrong_cf_atts = {
                att: unit
                for att, unit in attributes_with_lowest_appearance.items()
                if "conversion_factor" in att
            }
            name_pairs_cf = []
            if wrong_cf_atts:
                for wrong_cf_att in wrong_cf_atts:
                    names = wrong_cf_att.split("_conversion_factor_")
                    name_pairs_cf.append(names[1] + " of " + names[0])
                self._write_inconsistent_units_file(
                    energy_quantity_units,
                    item.name,
                    analysis=config.analysis,
                )
                raise AssertionError(
                    f"Unit inconsistency! Most probably, the {item.name} "
                    "unit(s) of the conversion factor(s) with dependent "
                    f"carrier {name_pairs_cf} are wrong."
                )
            # check if there is a retrofit flow coupling factor with wrong units
            wrong_rf_atts = {
                att: unit
                for att, unit in attributes_with_lowest_appearance.items()
                if "retrofit_flow_coupling_factor" in att
            }
            name_pairs_rf = []
            if wrong_rf_atts:
                for wrong_rf_att in wrong_rf_atts:
                    names = wrong_rf_att.split("_retrofit_flow_coupling_factor_")
                    name_pairs_rf.append(names[1] + " of " + names[0])
                self._write_inconsistent_units_file(
                    energy_quantity_units,
                    item.name,
                    analysis=config.analysis,
                )
                raise AssertionError(
                    f"Unit inconsistency! Most probably, the {item.name} "
                    f"unit(s) of the retrofit flow coupling factor(s) with "
                    f"dependent carrier {name_pairs_rf} are wrong."
                )
            if item.__class__ is Carrier:
                self._write_inconsistent_units_file(
                    energy_quantity_units,
                    item.name,
                    analysis=config.analysis,
                )
                units_of_wrong_attributes = self._get_units_of_wrong_attributes(
                    wrong_atts=attributes_with_lowest_appearance, unit_dict=unit_dict
                )
                raise AssertionError(
                    f"The attribute units of the {item.__class__.__name__} "
                    f"{item.name} are not consistent! Most probably, the "
                    f"unit(s) of the attribute(s) "
                    f"{units_of_wrong_attributes} are wrong."
                )
            else:
                self._write_inconsistent_units_file(
                    energy_quantity_units,
                    item.name,
                    analysis=config.analysis,
                    reference_carrier_name=reference_carrier_name,
                )
                units_of_wrong_attributes = self._get_units_of_wrong_attributes(
                    wrong_atts=attributes_with_lowest_appearance, unit_dict=unit_dict
                )
                raise AssertionError(
                    f"The attribute units of the {item.__class__.__name__} "
                    f"{item.name} and its reference carrier "
                    f"{reference_carrier_name} are not consistent! Most "
                    f"probably, the unit(s) of the attribute(s) "
                    f"{units_of_wrong_attributes} are wrong."
                )
        # since energy system doesn't have any attributes with energy dimension,
        # its dict must be empty
        elif item not in elements and len(energy_quantity_units_check) != 0:
            self._write_inconsistent_units_file(
                energy_quantity_units, item.name, analysis=config.analysis
            )
            units_of_wrong_attributes = self._get_units_of_wrong_attributes(
                wrong_atts=energy_quantity_units, unit_dict=unit_dict
            )
            raise AssertionError(
                f"The attribute units defined in the energy_system are not "
                f"consistent! Most probably, the unit(s) of the attribute(s) "
                f"{units_of_wrong_attributes} are wrong."
            )

    def _is_inconsistent(self, energy_quantity_units, exclude_strings=None):
        """Checks if the units of the attributes of an element are inconsistent.

        This method identifies inconsistencies in the units of attributes
        by comparing the energy  quantity terms across all attributes. It allows
        for the exclusion of certain attributes from the consistency check
        based on the `exclude_strings` parameter.

        Args:
            energy_quantity_units (dict): Dictionary containing attribute
                names and their corresponding energy quantity terms (e.g.,
                "kg/s", "m^2").
            exclude_strings (list, optional): List of strings for which c
                consistency is not checked (e.g., ["conversion_factor",
                "retrofit_flow_coupling_factor"]).

        Returns:
            bool:
                Returns `True` if there are inconsistencies (i.e., if the
                energy quantity units differ), otherwise returns `False`.
        """
        # exclude attributes which are not of interest for consistency
        if exclude_strings:
            energy_quantity_units = {
                key: value
                for key, value in energy_quantity_units.items()
                if all(es not in key for es in exclude_strings)
            }
        # check if all energy quantity units are the sames
        if len(set(energy_quantity_units.values())) > 1:
            return True
        else:
            return False

    def _get_units_of_wrong_attributes(self, wrong_atts, unit_dict):
        """Gets units of attributes showing wrong units.

        This method retrieves the units in base units for attributes that have
        inconsistent energy quantities based on the provided `wrong_atts`.

        Args:
            wrong_atts (dict): Dictionary containing attribute names with
                inconsistent units.
            unit_dict (dict): Dictionary of attribute names and their unit
                specifications in base units.

        Returns:
            dict:
                A dictionary where keys are attribute names, and values
                are their corresponding units in base units.
        """
        wrong_atts_with_units = {}
        for att in wrong_atts:
            wrong_atts_with_units[att] = [
                str(unit_specs["unit_in_base_units"].units)
                for key, unit_specs in unit_dict.items()
                if key == att
            ][0]
        return wrong_atts_with_units

    def _write_inconsistent_units_file(
        self, inconsistent_attributes, item_name, analysis, reference_carrier_name=None
    ):
        """Writes a file documenting attributes and their units that cause unit
        inconsistency.

        This method writes a JSON file that contains a record of the
        inconsistent attributes and their units for a given element or energy
        system. This helps with identifying and resolving unit issues in the
        system.

        Args:
            inconsistent_attributes (dict): Attributes that are inconsistent in
                terms of their units.
            item_name (str): The name of the element or energy system that has
                inconsistent units.
            analysis (dict): Dictionary containing analysis settings, including
                output folder.
            reference_carrier_name (str, optional): The name of the reference
                carrier, if the item is a conversion technology.
        """
        inconsistent_attributes_dict = {
            "element_name": item_name,
            "reference_carrier": reference_carrier_name,
            "attribute_names": str(inconsistent_attributes.keys()),
        }
        directory = os.path.join(
            analysis.folder_output, os.path.basename(analysis.dataset)
        )
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, "inconsistent_units.json")
        with open(path, "w") as json_file:
            json.dump(inconsistent_attributes_dict, json_file)

    def _get_attributes_with_least_often_appearing_unit(self, energy_quantity_units):
        """Finds attributes that have the least commonly appearing unit.

        This method identifies the attributes with the least frequent unit occurrence.
        The assumption is that the least frequent unit is most likely the incorrect one.

        Args:
            energy_quantity_units (dict): Dictionary containing attribute names
                and their corresponding energy quantity terms.

        Returns:
            dict:
                A dictionary of attributes that have the least frequently
                appearing units, along with their energy quantity terms.
        """
        min_unit_count = np.inf
        attributes_with_lowest_appearance = {}
        # count for all unique units how many times they appear to get an estimate
        # which unit most likely is the wrong one
        for distinct_unit in set(energy_quantity_units.values()):
            unit_count = list(energy_quantity_units.values()).count(distinct_unit)
            if (
                unit_count <= min_unit_count
                and unit_count < len(energy_quantity_units) / 2
            ):
                min_unit_count = unit_count
                wrong_value = distinct_unit
                attributes_with_lowest_appearance.update(
                    {
                        key: value
                        for key, value in energy_quantity_units.items()
                        if value == wrong_value
                    }
                )
        return attributes_with_lowest_appearance

    def get_most_often_appearing_energy_unit(self, energy_units):
        """Finds the most commonly appearing energy unit for a carrier's attributes.

        This method identifies the most frequently used energy unit across the
        attributes of a given carrier, which is assumed to be the correct one.

        Args:
            energy_units (dict): Dictionary containing attribute names and their
                energy quantity terms.

        Returns:
            str:
                The energy unit that appears most frequently across the
                attributes of the carrier.
        """
        max_unit_count = 0
        correct_value = None
        # count for all unique units how many times they appear to get an estimate
        # which unit most likely is the correct one
        for distinct_unit in set(energy_units.values()):
            unit_count = list(energy_units.values()).count(distinct_unit)
            if unit_count > max_unit_count:
                max_unit_count = unit_count
                correct_value = distinct_unit
        return correct_value

    def _get_conversion_factor_units(
        self, conversion_element, unit_specs, reference_carrier, elements
    ):
        """Splits conversion factor units into dependent and reference carrier units.

        This method takes a conversion factor and splits its units into two parts:
        one for the dependent carrier and one for the reference carrier. This is
        necessary when dealing with complex unit formats like "MW/MW".

        Args:
            conversion_element (object): The conversion technology element the
                conversion factor belongs to.
            unit_specs (dict): Dictionary containing unit category and unit as
                pint Quantity in base units.
            reference_carrier (Carrier): The reference carrier object for the
                conversion technology.
            elements (list): List of all elements in the system, used to find
                dependent carriers.

        Returns:
            dict:
                A dictionary of conversion factor units separated by
                dependent carrier and reference carrier.
        """
        conversion_factor_units = {}
        for dependent_carrier_name, cf_unit_specs in unit_specs.items():
            assert cf_unit_specs["unit"] != "1", (
                f"Since there doesn't exist a conversion_factor file for the "
                f"technology {conversion_element.name}, the attribute "
                f"conversion_factor_default must be defined with units to "
                f"ensure unit consistency"
            )
            units = cf_unit_specs["unit"].split("/")
            # check that no asterisk in unit strings without parentheses
            correct_unit_string = [
                ("*" in u and u[0] == "(" and u[1] == ")") or ("*" not in u)
                for u in units
            ]
            conversion_factors = [
                u for u, s in zip(units, correct_unit_string, strict=False) if not s
            ]
            assert all(correct_unit_string), (
                f"The conversion factor string(s)"
                f"{conversion_factors} of technology {conversion_element.name} "
                f"must not contain an asterisk '*' unless it is enclosed "
                "in parentheses '()'"
            )

            # problem: we don't know which parts of cf unit belong to which
            # carrier for units of format different from "unit/unit" (e.g.
            # kg/h/kW)
            # method: compare number of division signs of conversion factor
            # unit with number of division signs of corresponding carrier
            # element energy/power quantity
            dependent_carrier = [
                carrier
                for carrier in elements
                if carrier.name == dependent_carrier_name
            ][0]

            div_signs_dependent_carrier_energy = (
                self._get_number_of_division_signs_energy_quantity(
                    dependent_carrier.units
                )
            )
            div_signs_ref_carrier_energy = (
                self._get_number_of_division_signs_energy_quantity(
                    reference_carrier.units
                )
            )
            number_of_division_signs_energy = (
                div_signs_dependent_carrier_energy + div_signs_ref_carrier_energy
            )

            div_signs_dependent_carrier_power = (
                self._get_number_of_division_signs_energy_quantity(
                    dependent_carrier.units, power=True
                )
            )
            div_signs_ref_carrier_power = (
                self._get_number_of_division_signs_energy_quantity(
                    reference_carrier.units, power=True
                )
            )
            number_of_division_signs_power = (
                div_signs_ref_carrier_power + div_signs_dependent_carrier_power
            )

            # conversion factor unit must be defined as energy/energy or
            # power/power in the corresponding carrier energy quantity units
            # Check if the conversion factor is defined as energy/energy
            factor_units = {}
            if len(units) - 2 == number_of_division_signs_energy:
                # assign the unit parts to the corresponding carriers
                factor_units[dependent_carrier_name] = units[
                    0 : div_signs_dependent_carrier_energy + 1
                ]
                factor_units[reference_carrier.name] = units[
                    div_signs_dependent_carrier_energy + 1 :
                ]
            # check if the conversion factor is defined as power/power
            elif len(units) - 2 == number_of_division_signs_power:
                # assign the unit parts to the corresponding carriers
                factor_units[dependent_carrier_name] = units[
                    0 : div_signs_dependent_carrier_power + 1
                ]
                factor_units[reference_carrier.name] = units[
                    div_signs_dependent_carrier_power + 1 :
                ]
            else:
                raise AssertionError(
                    f"The conversion factor units of technology "
                    f"{conversion_element.name} must be defined as power/power "
                    f"or energy/energy of input/output carrier divided by "
                    f"reference carrier, e.g. MW/MW, MW/kg/s or GWh/GWh, "
                    f"kg/MWh etc."
                )
            # recombine the separated units carrier-wise to the initial fraction
            for key, value in factor_units.items():
                factor_units[key] = "/".join(value)
            conversion_factor_units[dependent_carrier_name] = factor_units
        return conversion_factor_units

    def _get_number_of_division_signs_energy_quantity(self, carrier_units, power=False):
        """Counts the number of division signs in a carrier's energy or power unit.

        This method counts the number of division signs ("/") in the most
        common energy or power unit of a carrier's attributes. It helps
        determine how energy or power is distributed across different units
        in the system.

        Args:
            carrier_units (dict): The units of the carrier element.
            power (bool, optional): If `True`, it counts the number of division
                signs in the power unit (energy divided by time). Defaults to
                `False`.

        Returns:
            int:
                The number of division signs in the most common energy or
                power unit.
        """
        energy_units = {}
        time_base_unit = [
            key for key, value in self.base_units.items() if value == "[time]"
        ][0]
        for attribute_name, unit_specs in carrier_units.items():
            energy_unit = self._remove_non_energy_units(unit_specs, attribute_name)
            if power:
                energy_unit[attribute_name] = energy_unit[attribute_name] / self.ureg(
                    time_base_unit
                )
            energy_units.update(energy_unit)
        energy_unit_ref_carrier = self.get_most_often_appearing_energy_unit(
            energy_units
        )
        return len(str(energy_unit_ref_carrier.units).split("/")) - 1

    def _remove_non_energy_units(self, unit_specs, attribute_name):
        """Removes all non-energy dimensions from a unit by multiplication/division.

        This method strips non-energy units (e.g., mass, time, etc.) from the
        specified unit and leaves only the energy quantity part. This is used
        for comparing energy quantities across different attributes.

        Args:
            unit_specs (dict): The unit specifications for the attribute.
            attribute_name (str): The name of the attribute to be processed.

        Returns:
            dict:
                A dictionary containing the attribute name and the
                energy-only unit.
        """
        # dictionary which assigns unit dimensions to corresponding base unit namings
        distinct_dims = {
            "money": "[currency]",
            "distance": "[length]",
            "time": "[time]",
            "emissions": "[mass]",
        }
        unit = unit_specs["unit_in_base_units"]
        unit_category = unit_specs["unit_category"]
        for dim, dim_name in distinct_dims.items():
            if dim in unit_category:
                dim_unit = [
                    key for key, value in self.base_units.items() if value == dim_name
                ][0]
                if dim == "time" and "energy_quantity" in unit_category:
                    unit = unit / self.ureg(dim_unit) ** (
                        -1 * unit_category["energy_quantity"]
                    )
                else:
                    unit = unit / self.ureg(dim_unit) ** unit_category[dim]
        if "energy_quantity" in unit_category:
            unit = unit ** unit_category["energy_quantity"]
        return {attribute_name: unit}

    def save_carrier_energy_quantities(self, element_registry: "ElementRegistry"):
        """Saves energy quantity units of carriers after consistency checks.

        This method stores the energy quantities of the carriers after they
        have been verified for unit consistency. It ensures that the units of
        the carrier's attributes are properly assigned to variables for later
        use in calculations.

        Args:
            element_registry (ElementRegistry): The registry containing all elements in
                the system, used to retrieve carrier elements.

        Returns:
            dict:
                A dictionary containing the carrier units.
        """
        for carrier in element_registry.all_elements_of_type(Carrier):
            self.carrier_energy_quantities[carrier.name] = (
                self._remove_non_energy_units(
                    carrier.units["demand"], attribute_name=None
                )[None]
            )

    def set_base_unit_combination(self, input_unit, attribute):
        """Converts the input unit to the corresponding base unit.

        This method takes an input unit and converts it to its base unit
        equivalent, which can be used for further unit analysis. It also handles
        special cases where the input unit is `NaN` or dimensionless.

        Args:
            input_unit (str or Quantity): The unit to be converted to base units.
            attribute (str): The name of the attribute that uses the unit.
        """
        # TODO combine overlap with get_unit_multiplier
        # if input unit is already in base units --> the input unit is base unit
        if input_unit in self.base_units:
            _, base_unit_combination = self.calculate_combined_unit(input_unit)
        # if input unit is nan --> dimensionless old definition
        elif type(input_unit) is not str and np.isnan(input_unit):
            base_unit_combination = pd.Series(index=self.dim_matrix.columns, data=0)
        else:
            # convert to string
            input_unit = str(input_unit)
            # if input unit is 1 --> dimensionless new definition
            if input_unit == "1":
                return 1
            _, base_unit_combination = self.calculate_combined_unit(input_unit)
        if (base_unit_combination != 0).any():
            self.dict_attribute_values[attribute] = {
                "base_combination": base_unit_combination,
                "values": None,
            }

    def set_attribute_values(self, df_output, attribute):
        """Saves the values of an attribute from a dataframe output.

        This method stores the values of a given attribute from a dataframe into
        the class' internal dictionary for future use.

        Args:
            df_output (DataFrame): The dataframe containing the output values
                for attributes.
            attribute (str): The name of the attribute whose values are being
                saved.
        """
        if attribute in self.dict_attribute_values.keys():
            self.dict_attribute_values[attribute]["values"] = df_output

    def check_if_invalid_hourstring(self, input_unit):
        """Checks if "h" in the input unit refers to the Planck constant.

        This method ensures that the string "h" is not mistaken for the
        Planck constant when specifying time units in the system. It will
        raise an error if "h" is used incorrectly.

        Args:
            input_unit (str): The unit string to be checked.
        """
        _tuple_units = self.ureg(input_unit).to_tuple()[1]
        _list_units = [_item[0] for _item in _tuple_units]
        assert "planck_constant" not in _list_units, (
            f"Error in input unit '{input_unit}'. Did you want to "
            "define hour? Use 'hour' instead of 'h' ('h' is interpreted "
            "as the planck constant)"
        )

    def define_ton_as_metric(self):
        """Redefines the "ton" as a metric ton.

        This method redefines the unit "ton" to represent the metric ton,
        ensuring consistency across the system when dealing with mass units.
        """
        self.ureg.define("ton = metric_ton")

    def redefine_standard_units(self):
        """Redefines standard units required in the system.

        This method sets up standard units such as "Euro", "year", and "ton",
        and ensures that the system handles leap years correctly.
        """
        self.ureg.define("Euro = [currency] = EURO = Eur = €")
        self.ureg.define("year = 365 * day = a = yr = julian_year")
        self.ureg.define("ton = metric_ton")

    @staticmethod
    def check_pos_neg_boolean(array, axis=None):
        """Checks if array contains only positive or negative booleans (-1, 0, 1).

        This method verifies if the input array contains values that are either
        positive or negative booleans, which is often used to check binary
        states in the optimization.

        Args:
            array (numpy.ndarray): The numeric array to be checked.
            axis (int, optional): The axis of the dataframe along which the
                check is applied.

        Returns:
            bool:
                Returns `True` if the array contains only positive or
                negative booleans, otherwise returns `False`.
        """
        if axis:
            is_pos_neg_boolean = np.apply_along_axis(
                lambda row: np.array_equal(np.abs(row), np.abs(row).astype(bool)),
                1,
                array,
            ).any()
        else:
            is_pos_neg_boolean = np.array_equal(
                np.abs(array), np.abs(array).astype(bool)
            )
        return is_pos_neg_boolean
