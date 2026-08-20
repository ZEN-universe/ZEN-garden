import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp
import xarray as xr

from zen_garden.utils import get_label_position

if TYPE_CHECKING:
    from linopy import Model as LinopyModel

    from zen_garden.model import Config

logger = logging.getLogger(__name__)


# ToDo get rid of A matrix dependency -> for big models slowest part;
# can we use the data structure of linopy directly to determine column and
# row scaling factors.
# ToDo slight numerical errors after rescaling -> dependent on solver ->
# for gurobi very accurate
class Scaling:
    """This class scales the optimization model before solving it and rescales the
    solution.
    """

    def __init__(
        self,
        config: "Config",
        lp_model: "LinopyModel",
        algorithm: list[str] | None = None,
        include_rhs: bool = True,
    ):
        """Initializes scaling instance.

        Args:
            model: optimization model
            algorithm: list of scaling algorithms
            include_rhs: bool whether to include the right hand side in the scaling

        """
        # optimization model to perform scaling on
        if algorithm is None:
            algorithm = ["geom"]
        elif type(algorithm) is str:
            logging.warning(
                "Please provide a list of scaling algorithms, not a single string."
            )
            algorithm = [algorithm]

        self.config = config
        self.lp_model = lp_model
        self.algorithm = algorithm
        self.include_rhs = include_rhs
        # For Numerical Range Improvement
        self.last_lhs_range = 0
        self.last_rhs_range = 0
        # For benchmarking
        self.scaling_time = 0

    def initiate_A_matrix(self):
        """Constructs the A matrix and the right hand side of the constraints."""
        self.A_matrix = self.lp_model.constraints.to_matrix(filter_missings=False)
        self.A_matrix_copy = self.A_matrix.copy()  # necessary for printing of numerics
        self.D_r_inv = np.ones(self.A_matrix.get_shape()[0])
        self.D_c_inv = np.ones(self.A_matrix.get_shape()[1])
        self.rhs = []
        for name in self.lp_model.constraints:
            constraint = self.lp_model.constraints[name]
            labels = constraint.labels.data
            mask = np.atleast_1d(labels != -2).nonzero()
            try:
                self.rhs += constraint.rhs.data[mask].tolist()
            except Exception:
                self.rhs += [constraint.rhs.data]
        self.rhs = np.array(self.rhs)
        # np.abs(np.array(self.rhs)) -> could get rid of all the other np.ads
        # in iter_sclaing() etc. but then print numerics only includes absolute
        # values
        self.rhs[self.rhs == np.inf] = 0
        self.rhs_copy = self.rhs.copy()  # necessary for printing of numerics

    def re_scale(self):
        """Rescales the solution of the optimization model."""
        for name_var in self.lp_model.variables:
            var = self.lp_model.variables[name_var]
            cond = var.labels != -1
            var.solution = var.solution.where(
                ~cond,  # Where condition is False, keep original data
                var.solution * self.D_c_inv[var.labels],  # Where True, apply math
            )

        if self.config.solver.save_duals:
            for name_con in self.lp_model.constraints:
                con = self.lp_model.constraints[name_con]
                cond = con.labels != -1
                con.dual = con.dual.where(
                    ~cond,  # Where condition is False, keep original data
                    con.dual * self.D_r_inv[con.labels],  # Where True, apply math
                )

    def rescale_dataarray(self, arr: xr.DataArray, name: str) -> xr.DataArray:
        """Rescales a dataarray with the scaling factors of the optimization model.

        Args:
            arr: dataarray to be rescaled
            name: name of the variable or constraint to which the dataarray belongs

        Returns:
            rescaled dataarray
        """
        if name in self.lp_model.variables:
            component = self.lp_model.variables[name]
            D_inv = self.D_c_inv[component.labels]
        elif name in self.lp_model.constraints:
            component = self.lp_model.constraints[name]
            D_inv = self.D_r_inv[component.labels]
        else:
            raise ValueError(f"{name} is not a variable or constraint in the model.")

        cond = component.labels != -1
        return arr.where(
            ~cond,  # Where condition is False, keep original data
            arr * D_inv,  # Where True, apply math
        )

    def analyze_numerics(self):
        """Analyzes the numerics of the optimization model."""
        # print numerics if no scaling is activated
        self.initiate_A_matrix()
        self.A_matrix.eliminate_zeros()
        self.print_numerics(0, True)

    def run_scaling(self):
        """Runs the scaling algorithm. Function called in runner.py."""
        logger.info("\n--- Start Scaling ---\n")
        t0 = time.perf_counter()
        self.initiate_A_matrix()
        self.iter_scaling()
        self.overwrite_problem()
        t1 = time.perf_counter()
        self.scaling_time = t1 - t0  # for benchmarking
        logger.info(f"\nTime to Scale Problem: {t1 - t0:0.1f} seconds\n")

    def replace_data(self, name):
        """Replaces the data (coefficients) of the lhs and rhs of the constraint
        with the scaled data.

        Args:
            name: name of the constraint for which the data is replaced with
                the scaled data
        """
        constraint = self.lp_model.constraints[name]
        # Get data
        lhs = constraint.coeffs.data
        mask_skip_constraints = constraint.labels.data
        mask_variables = constraint.vars.data
        rhs = constraint.rhs.data
        # Find the indices where constraint_mask is not equal to -1
        indices = np.atleast_1d(mask_skip_constraints != -1).nonzero()
        if indices[0].size > 0:
            # Update rhs
            try:
                rhs[indices] = (
                    rhs[indices] * self.D_r_inv[mask_skip_constraints[indices]]
                )
            except IndexError:
                constraint.rhs.data = rhs * self.D_r_inv[mask_skip_constraints]
            # Update lhs
            non_nan_mask = ~np.isnan(lhs)
            entries_to_overwrite = np.where(non_nan_mask & (mask_variables != -1))
            lhs[entries_to_overwrite] *= (
                self.D_r_inv[mask_skip_constraints[entries_to_overwrite[:-1]]]
                * self.D_c_inv[mask_variables[entries_to_overwrite]]
            )

    def adjust_upper_lower_bounds_variables(self):
        """Adjusts the upper and lower bounds of the variables whose coefficients
        are scaled. If the bounds are not scaled, the problem might get
        infeasible.
        """
        vars = self.lp_model.variables
        for var in vars:
            mask = np.where(vars[var].labels.data != -1)
            scaling_factors = self.D_c_inv[vars[var].labels.data[mask]]
            vars[var].upper.data[mask] = vars[var].upper.data[
                mask
            ] * scaling_factors ** (-1)
            vars[var].lower.data[mask] = vars[var].lower.data[
                mask
            ] * scaling_factors ** (-1)

    def adjust_scaling_factors_of_skipped_rows(self, name):
        """Adjusts the column scaling factors corresponding to variables that are
        part of rows that are skipped. If the scaling factors are not adjusted,
        the problem cannot be rescaled to the original problem.

        Args:
            name: name of the constraint for which the scaling factors are
                adjusted
        """
        constraint = self.lp_model.constraints[name]
        # rows -> unnecessary to adjust scaling factor of rows with binary and
        # integer variables as skipped anyways
        # cols
        mask_variables = constraint.vars.data
        indices = np.where(mask_variables != -1)
        self.D_c_inv[mask_variables[indices]] = 1

    def adjust_int_variables(self):
        """Adjusts the column scaling factors corresponding to binary and integer
        variables. These columns are skipped in the scaling process since
        scaling is solely valid for continuous variables.
        """
        vars = self.lp_model.variables
        for var in vars:
            if vars[var].attrs["binary"] or vars[var].attrs["integer"]:
                mask = np.where(vars[var].labels.data != -1)
                self.D_c_inv[vars[var].labels.data[mask]] = 1

    def overwrite_problem(self):
        """Overwrites the optimization problem with the scaled data."""
        # pre-check variables -> skip binary and integer variables
        self.adjust_int_variables()
        # adjust scaling factors that have inf or nan values -> not
        # really necessary anymore but might be a good security check
        self.D_c_inv[self.D_c_inv == np.inf] = 1
        self.D_r_inv[self.D_r_inv == np.inf] = 1
        self.D_c_inv = np.nan_to_num(self.D_c_inv, nan=1)
        self.D_r_inv = np.nan_to_num(self.D_r_inv, nan=1)
        # pre-check rows -> otherwise inconsistency in scaling
        for name_con in self.lp_model.constraints:
            if self.lp_model.constraints[name_con].coeffs.dtype == int:
                self.adjust_scaling_factors_of_skipped_rows(name_con)
        self.print_numerics_of_last_iteration()
        # Include adjust upper/lower bounds of variables that are scaled
        self.adjust_upper_lower_bounds_variables()
        # overwrite constraints
        for name_con in self.lp_model.constraints:
            # overwrite data
            # check if only integers are allowed in scaling: if yes skip
            # and overwrite scaling vector
            if self.lp_model.constraints[name_con].coeffs.dtype == int:
                continue
            else:
                self.replace_data(name_con)
        # overwrite objective
        vars = self.lp_model.objective.vars.data
        scale_factors = self.D_c_inv[vars]
        self.lp_model.objective.coeffs.data = (
            self.lp_model.objective.coeffs.data * scale_factors
        )

    def get_min(self, A_matrix):
        """Gets the minimum values each column or row of the A matrix.

        Args:
            A_matrix: A matrix of the optimization model (scipy.sparse.csr_matrix)

        Returns:
            np.array: Minimum values of each column or row
        """
        d = A_matrix.data
        try:
            mins_values = np.minimum.reduceat(np.abs(d), A_matrix.indptr[:-1])

        # necessary if multiple columns and rows at the end of the matrix
        # without entries -> if not only last entry of indptr is len(data) and
        # therefore out of range
        except Exception:
            last_empty_entries = A_matrix.indptr[A_matrix.indptr == len(d)]
            non_empty_entries = A_matrix.indptr[A_matrix.indptr < len(d)]
            mins_values = np.minimum.reduceat(np.abs(d), non_empty_entries)
            mins_values = np.hstack(
                (mins_values, np.ones((len(last_empty_entries) - 1,)))
            )
        return mins_values

    def get_full_geom(
        self, A_matrix, axis
    ):  # Very slow and less effective than simplified geom norm
        """Gets the full geometric mean of each column or row of the A matrix.
        Note, this funtcion is very slow and is not yet ready to be used in the
        scaling process.

        Args:
            A_matrix: A matrix of the optimization model
                (scipy.sparse.csr_matrix)
            axis: axis along which the geometric mean is calculated

        Returns:
            geometric mean of each column or row
        """
        d = A_matrix.data
        geom = np.ones(len(A_matrix.indptr) - 1)
        nonzero_entries = np.unique(list(A_matrix.nonzero()[axis]))
        idx_unique = np.unique(A_matrix.indptr[:-1])
        d_slices = np.split(d, idx_unique[1:])
        geom[nonzero_entries] = list(map(lambda x: sp.stats.gmean(np.abs(x)), d_slices))
        return geom

    def update_A(self, vector, axis):
        """Updates the A matrix with the current scaling vector.
        This function does not overwrite the original optimization model
        but is used for the scaling process.

        Args:
            vector: vector to update current scaling vectors
            axis: axis for which the scaling vector is updated (0 for
                rows, 1 for columns)

        """
        if axis == 1:
            self.A_matrix = sp.sparse.diags(vector, 0, format="csr").dot(self.A_matrix)
            self.D_r_inv = self.D_r_inv * vector
            self.rhs = self.rhs * vector
        elif axis == 0:
            self.A_matrix = self.A_matrix.dot(sp.sparse.diags(vector, 0, format="csr"))
            self.D_c_inv = self.D_c_inv * vector

    def print_numerics_of_last_iteration(self):
        """Prints the numerics of the last iteration of the scaling process."""
        self.A_matrix = (
            sp.sparse.diags(self.D_r_inv, 0, format="csr")
            .dot(self.A_matrix_copy)
            .dot(sp.sparse.diags(self.D_c_inv, 0, format="csr"))
        )
        self.rhs = self.rhs_copy * self.D_r_inv
        self.print_numerics(len(self.algorithm))

    def generate_numerics_string(
        self, label, index=None, A_matrix=None, var=None, is_rhs=False
    ):
        """Generates a string for log-outputs during scaling.

        :param label: label of the constraint
        :param index: index of the A matrix
        :param A_matrix: A matrix of the optimization model
        :param var: variable of the optimization model
        :param is_rhs: bool whether the string is computed for the right hand side
        :return: string for log-outputs
        """
        if is_rhs:
            cons_str = get_label_position(self.lp_model.constraints, label)
            cons_str = (
                f"{cons_str[0]}[{','.join([str(k) for k in cons_str[1].values()])}]"
            )
            return f"{self.rhs[label]} in {cons_str}"
        else:
            cons_str = get_label_position(self.lp_model.constraints, label)
            cons_str = (
                f"{cons_str[0]}[{','.join([str(k) for k in cons_str[1].values()])}]"
            )
            var_str = get_label_position(self.lp_model.variables, var)
            var_str = f"{var_str[0]}[{','.join([str(k) for k in var_str[1].values()])}]"
            return f"{A_matrix[index]} {var_str} in {cons_str}"

    def print_numerics(self, i, no_scaling=False, benchmarking_output=False):
        """Prints the numerics of the optimization model.

        Args:
            i: iteration of the scaling process
            no_scaling: bool whether no scaling is activated. Then only numerics
                are printed.
            benchmarking_output: bool whether data for benchmarking is collected
            cond_number: bool whether the condition number of the A matrix is
                computed

        Returns:
            numerical range of the A matrix and the right hand side as well as
                the condition number of the A matrix (if benchmarking_output is
                True
        """
        data_coo = self.A_matrix.tocoo()
        A_abs = np.abs(data_coo.data)
        A_abs_nonzero = np.ma.masked_equal(A_abs, 0.0, copy=False)
        index_max = np.argmax(A_abs_nonzero)
        index_min = np.argmin(A_abs_nonzero)
        row_max = data_coo.row[index_max]
        col_max = data_coo.col[index_max]
        row_min = data_coo.row[index_min]
        col_min = data_coo.col[index_min]
        rhs_max_index = np.where(
            np.abs(self.rhs) == np.max(np.abs(self.rhs)[self.rhs != np.inf])
        )[0][0]
        rhs_min_index = np.where(
            np.abs(self.rhs) == np.min(np.abs(self.rhs)[np.abs(self.rhs) > 0])
        )[0][0]
        # Max Matrix String
        cons_str_max = self.generate_numerics_string(
            row_max, index=index_max, A_matrix=data_coo.data, var=col_max
        )
        # Min Matrix String
        cons_str_min = self.generate_numerics_string(
            row_min, index=index_min, A_matrix=data_coo.data, var=col_min
        )
        # RHS values
        cons_rhs_max = self.generate_numerics_string(rhs_max_index, is_rhs=True)
        cons_rhs_min = self.generate_numerics_string(rhs_min_index, is_rhs=True)
        # Ranges
        # LHS
        range_lhs = np.floor(np.log10(A_abs[index_max]) - np.log10(A_abs[index_min]))
        # RHS
        range_rhs = np.floor(
            np.log10(np.abs(self.rhs[rhs_max_index]))
            - np.log10(np.abs(self.rhs[rhs_min_index]))
        )
        if benchmarking_output:  # for postprocessing
            range_lhs = np.log10(A_abs[index_max]) - np.log10(A_abs[index_min])
            range_rhs = np.log10(np.abs(self.rhs[rhs_max_index])) - np.log10(
                np.abs(self.rhs[rhs_min_index])
            )
            return range_lhs, range_rhs
        else:
            # Prints
            if no_scaling:
                logger.info("\n--- Analyze Numerics ---\n")
            else:
                logger.info(f"\n--- Numerics at iteration {i} ---\n")
            logger.info(
                "\n".join(
                    [
                        f"Max value of A matrix: {cons_str_max}",
                        f"Min value of A matrix: {cons_str_min}",
                        f"Max value of RHS: {cons_rhs_max}",
                        f"Min value of RHS: {cons_rhs_min}",
                        "Numerical Range:",
                        "    LHS : {}".format(
                            [
                                format(A_abs[index_min], ".1e"),
                                format(A_abs[index_max], ".1e"),
                            ]
                        ),
                        "    RHS : {}".format(
                            [
                                format(np.abs(self.rhs[rhs_min_index]), ".1e"),
                                format(np.abs(self.rhs[rhs_max_index]), ".1e"),
                            ]
                        ),
                    ]
                )
            )
            if i > 0:
                logger.info(
                    "Numerical Range Improvement:\n"
                    f"    LHS : {range_lhs - self.last_lhs_range}\n"
                    f"    RHS : {range_rhs - self.last_rhs_range}"
                )
            self.last_lhs_range = range_lhs
            self.last_rhs_range = range_rhs
            return range_lhs, range_rhs

    def iter_scaling(self):
        """Generates the row and column scaling factors."""
        # transform A matrix to csr matrix for better computational properties
        self.A_matrix.eliminate_zeros()
        self.A_matrix = sp.sparse.csr_matrix(self.A_matrix)
        # initiate iteration counter
        i = 0
        self.print_numerics(i)
        for algo in self.algorithm:
            i += 1
            # update row scaling vector
            if algo == "infnorm":
                # update row scaling vector
                max_rows = sp.sparse.linalg.norm(self.A_matrix, ord=np.inf, axis=1)
                if self.include_rhs:
                    max_rows = np.maximum(
                        max_rows,
                        np.abs(self.rhs),
                        out=max_rows,
                        where=self.rhs != np.inf,
                    )
                max_rows[max_rows == 0] = 1  # to avoid warning outputs
                r_vector = 1 / max_rows
                r_vector = np.power(2, np.round(np.emath.logn(2, r_vector)))
                # update A and row scaling matrix
                self.update_A(r_vector, 1)
                # update column scaling vector
                max_cols = sp.sparse.linalg.norm(self.A_matrix, ord=np.inf, axis=0)
                max_cols[max_cols == 0] = 1  # to avoid warning outputs
                c_vector = 1 / max_cols
                c_vector = np.power(2, np.round(np.emath.logn(2, c_vector)))
                # update A and column scaling matrix
                self.update_A(c_vector, 0)
                # Print Numerics
                if i < len(self.algorithm):
                    self.print_numerics(i)

            elif algo == "geom":
                # update row scaling vector
                max_rows = sp.sparse.linalg.norm(self.A_matrix, ord=np.inf, axis=1)
                min_rows = self.get_min(self.A_matrix)
                if self.include_rhs:
                    max_rows = np.maximum(
                        max_rows,
                        np.abs(self.rhs),
                        out=max_rows,
                        where=self.rhs != np.inf,
                    )
                    min_rows = np.minimum(
                        min_rows,
                        np.abs(self.rhs),
                        out=min_rows,
                        where=np.abs(self.rhs) > 0,
                    )
                geom = (max_rows * min_rows) ** 0.5
                geom[geom == 0] = 1  # to avoid warning outputs
                r_vector = 1 / geom
                r_vector = np.power(2, np.round(np.emath.logn(2, r_vector)))
                # update A and row scaling matrix
                self.update_A(r_vector, 1)
                # update column scaling vector
                max_cols = sp.sparse.linalg.norm(self.A_matrix, ord=np.inf, axis=0)
                min_cols = self.get_min(self.A_matrix.tocsc())
                geom = (max_cols * min_cols) ** 0.5
                geom[geom == 0] = 1  # to avoid warning outputs
                c_vector = 1 / geom
                c_vector = np.power(2, np.round(np.emath.logn(2, c_vector)))
                # update A and column scaling matrix
                self.update_A(c_vector, 0)
                # Print Numerics
                if i < len(self.algorithm):
                    self.print_numerics(i)

            elif algo == "arithm":
                # update row scaling vector
                mean_rows = sp.sparse.linalg.norm(self.A_matrix, ord=1, axis=1) / (
                    np.diff(self.A_matrix.indptr)
                    + np.ones(self.A_matrix.get_shape()[0])
                )
                if self.include_rhs:
                    mean_rows = mean_rows + np.abs(self.rhs) / (
                        np.diff(self.A_matrix.indptr)
                        + np.ones(self.A_matrix.get_shape()[0])
                    )
                mean_rows[mean_rows == 0] = 1  # to avoid warning outputs
                c_vector = 1 / mean_rows
                c_vector = np.power(2, np.round(np.emath.logn(2, c_vector)))
                # update A and row scaling matrix
                self.update_A(c_vector, 1)
                # update column scaling vector
                mean_cols = sp.sparse.linalg.norm(
                    self.A_matrix, ord=1, axis=0
                ) / np.diff(self.A_matrix.tocsc().indptr)
                mean_cols[mean_cols == 0] = 1  # to avoid warning outputs
                r_vector = 1 / mean_cols
                r_vector = np.power(2, np.round(np.emath.logn(2, r_vector)))
                # update A and column scaling matrix
                self.update_A(r_vector, 0)
                # Print Numerics
                if i < len(self.algorithm):
                    self.print_numerics(i)
