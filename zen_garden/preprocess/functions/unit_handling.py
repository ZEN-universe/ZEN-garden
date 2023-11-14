"""
:Title:          ZEN-GARDEN
:Created:        April-2022
:Authors:        Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Class containing the unit handling procedure.
"""
import logging
import numpy as np
import pandas as pd
import scipy as sp
import warnings
from pint import UnitRegistry
from pint.util import column_echelon_form

# enable Deprecation Warnings
warnings.simplefilter('always', DeprecationWarning)

class UnitHandling:
    """
    Class containing the unit handling procedure
    """

    def __init__(self, folder_path, round_decimal_points, define_ton_as_metric_ton=True):
        """ initialization of the unit_handling instance

        :param folder_path: The path to the folder containing the system specifications
        :param round_decimal_points: rounding tolerance
        :param define_ton_as_metric_ton: bool to use another definition for tons
        """
        self.folder_path = folder_path
        self.rounding_decimal_points = round_decimal_points
        self.get_base_units(define_ton_as_metric_ton)
        # dict of element attribute values
        self.dict_attribute_values = {}

    def get_base_units(self, define_ton_as_metric_ton=True):
        """ gets base units of energy system

        :param define_ton_as_metric_ton: bool to use another definition for tons
        """
        _list_base_unit = self.extract_base_units()
        self.ureg = UnitRegistry()

        if define_ton_as_metric_ton:
            self.define_ton_as_metric()
        # load additional units
        self.ureg.load_definitions(self.folder_path + "/unit_definitions.txt")

        # empty base units and dimensionality matrix
        self.base_units = {}
        self.dim_matrix = pd.DataFrame(index=_list_base_unit).astype(int)
        for _base_unit in _list_base_unit:
            dim_unit = self.ureg.get_dimensionality(self.ureg(_base_unit))
            self.base_units[_base_unit] = self.ureg(_base_unit).dimensionality
            self.dim_matrix.loc[_base_unit, list(dim_unit.keys())] = list(dim_unit.values())
        self.dim_matrix = self.dim_matrix.fillna(0).astype(int).T

        # check if unit defined twice or more
        _duplicate_units = self.dim_matrix.T.duplicated()
        if _duplicate_units.any():
            _dim_matrix_duplicate = self.dim_matrix.loc[:, _duplicate_units]
            for _duplicate in _dim_matrix_duplicate:
                # if same unit twice (same order of magnitude and same dimensionality)
                if len(self.dim_matrix[_duplicate].shape) > 1:
                    logging.warning(f"The base unit <{_duplicate}> was defined more than once. Duplicates are dropped.")
                    _duplicateDim = self.dim_matrix[_duplicate].T.drop_duplicates().T
                    self.dim_matrix = self.dim_matrix.drop(_duplicate, axis=1)
                    self.dim_matrix[_duplicate] = _duplicateDim
                else:
                    raise KeyError(f"More than one base unit defined for dimensionality {self.base_units[_duplicate]} (e.g., {_duplicate})")
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
        list_base_units = pd.read_csv(self.folder_path + "/base_units.csv").squeeze().values.tolist()
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
        _missing_dim = set(dim_input.keys()).difference(dim_vector.keys())
        assert len(_missing_dim) == 0, f"No base unit defined for dimensionalities <{_missing_dim}>"
        dim_vector[list(dim_input.keys())] = list(dim_input.values())
        # calculate dimensionless combined unit (e.g., tons and kilotons)
        combined_unit = self.ureg(input_unit).units
        # if unit (with a different multiplier) is already in base units
        if self.dim_matrix.isin(dim_vector).all(axis=0).any():
            base_combination = self.dim_matrix.isin(dim_vector).all(axis=0).astype(int)
            _base_unit = self.ureg(self.dim_matrix.columns[self.dim_matrix.isin(dim_vector).all(axis=0)][0])
            combined_unit *= _base_unit ** (-1)
        # if inverse of unit (with a different multiplier) is already in base units (e.g. 1/km and km)
        elif (self.dim_matrix * -1).isin(dim_vector).all(axis=0).any():
            base_combination = (self.dim_matrix * -1).isin(dim_vector).all(axis=0).astype(int) * (-1)
            _base_unit = self.ureg(self.dim_matrix.columns[(self.dim_matrix * -1).isin(dim_vector).all(axis=0)][0])
            combined_unit *= _base_unit
        else:
            dim_analysis = self.dim_analysis
            # drop dependent units
            dim_matrix_reduced = self.dim_matrix.drop(dim_analysis["dependent_units"], axis=1)
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
                calculated_multiplier = False
                for unit, power in zip(dim_matrix_reduced.columns, combination_solution):
                    # try to substitute unit by a dependent unit
                    if not calculated_multiplier:
                        # iterate through dependent units
                        for dependent_unit, dependent_dim in zip(dim_analysis["dependent_units"], dim_analysis["dependent_dims"]):
                            idx_unit_in_reduced_matrix = list(dim_matrix_reduced.columns).index(unit)
                            # if the power of the unit is the same as of the dimensionality in the dependent unit
                            if np.abs(dependent_dim[idx_unit_in_reduced_matrix]) == np.abs(power):
                                dim_matrix_reduced_temp = dim_matrix_reduced.drop(unit, axis=1)
                                dim_matrix_reduced_temp[dependent_unit] = self.dim_matrix[dependent_unit]
                                combination_solution_temp = np.linalg.solve(dim_matrix_reduced_temp, dim_vector)
                                if UnitHandling.check_pos_neg_boolean(combination_solution_temp):
                                    # compose relevant units to dimensionless combined unit
                                    base_combination = pd.Series(index=self.dim_matrix.columns, data=0)
                                    base_combination[dim_matrix_reduced_temp.columns] = combination_solution_temp
                                    for unit_temp, power_temp in zip(dim_matrix_reduced_temp.columns, combination_solution_temp):
                                        combined_unit *= self.ureg(unit_temp) ** (-1 * power_temp)
                                    calculated_multiplier = True
                                    break
                assert calculated_multiplier, f"Cannot establish base unit conversion for {input_unit} from base units {self.base_units.keys()}"
        if return_combination:
            return base_combination
        else:
            return combined_unit

    def get_unit_multiplier(self, input_unit, attribute_name, path=None):
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
            warnings.warn(f"Parameter {attribute_name} of {path.name} has no unit (assign unit '1' to unitless parameters)",DeprecationWarning)
            return 1
        #if input unit is 1 --> dimensionless new definition
        elif input_unit == "1":
            return 1
        else:
            combined_unit = self.calculate_combined_unit(input_unit)
            assert combined_unit.to_base_units().unitless, f"The unit conversion of unit {input_unit} did not resolve to a dimensionless conversion factor. Something went wrong."
            # magnitude of combined unit is multiplier
            multiplier = combined_unit.to_base_units().magnitude
            # check that multiplier is larger than rounding tolerance
            assert multiplier >= 10 ** (-self.rounding_decimal_points), f"Multiplier {multiplier} of unit {input_unit} in parameter {attribute_name} is smaller than rounding tolerance {10 ** (-self.rounding_decimal_points)}"
            # round to decimal points
            return round(multiplier, self.rounding_decimal_points)

    def set_base_unit_combination(self, input_unit, attribute):
        """ converts the input unit to the corresponding base unit

        :param input_unit: #TODO describe parameter/return
        :param attribute: #TODO describe parameter/return
        """
        # if input unit is already in base units --> the input unit is base unit
        if input_unit in self.base_units:
            base_unit_combination = self.calculate_combined_unit(input_unit, return_combination=True)
        # if input unit is nan --> dimensionless old definition
        elif type(input_unit) != str and np.isnan(input_unit):
            base_unit_combination = pd.Series(index=self.dim_matrix.columns, data=0)
        #if input unit is 1 --> dimensionless new definition
        elif input_unit == "1":
            base_unit_combination = pd.Series(index=self.dim_matrix.columns, data=0)
        else:
            base_unit_combination = self.calculate_combined_unit(input_unit, return_combination=True)
        if (base_unit_combination != 0).any():
            self.dict_attribute_values[attribute] = {"base_combination": base_unit_combination, "values": None}

    def set_attribute_values(self, df_output, attribute):
        """ saves the attributes values of an attribute

        :param df_output: #TODO describe parameter/return
        :param attribute: #TODO describe parameter/return
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

        #remove rows of df_units which contain only zeros since they cannot be scaled anyway and may influence minimization convergence
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

    @staticmethod
    def check_pos_neg_boolean(array, axis=None):
        """ checks if the array has only positive or negative booleans (-1,0,1)

        :param array: numeric numpy array
        :param axis: #TODO describe parameter/return
        :return is_pos_neg_boolean """
        if axis:
            is_pos_neg_boolean = np.apply_along_axis(lambda row: np.array_equal(np.abs(row), np.abs(row).astype(bool)), 1, array).any()
        else:
            is_pos_neg_boolean = np.array_equal(np.abs(array), np.abs(array).astype(bool))
        return is_pos_neg_boolean
