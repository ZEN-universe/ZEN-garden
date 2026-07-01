import logging
import re

import numpy as np
import xarray as xr


class IISConstraintParser(object):
    """This class is used to parse the IIS constraints and print them in a nice way
    Most functions here are just copied from linopy 0.2.x.
    """

    EQUAL = "="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    SIGNS = {EQUAL, GREATER_EQUAL, LESS_EQUAL}
    SIGNS_pretty = {EQUAL: "=", GREATER_EQUAL: "≥", LESS_EQUAL: "≤"}

    def __init__(self, iis_file, model):
        """Initializes the IIS constraint parser.

        :param iis_file: The file containing the IIS
        :param model: The model to which the IIS belongs
        """
        # disable logger temporarily
        logging.disable(logging.WARNING)
        self.iis_file = iis_file
        self.model = model
        # write gurobi IIS to file
        self.write_gurobi_iis()
        # get the labels
        self.constraint_labels, self.var_labels, self.var_lines = self.read_labels()
        # enable logger again
        logging.disable(logging.NOTSET)

    def write_parsed_output(self, outfile=None):
        """Writes the parsed output to a file.

        :param outfile: The file to write to
        """
        # avoid truncating the expression
        # write the outfile
        if outfile is None:
            outfile = self.iis_file
        seen_constraints = []
        seen_variables = []
        with open(outfile, "w") as f:
            f.write("Constraints:\n")
            constraints = self.model.constraints
            for label in self.constraint_labels:
                name, coord = self.get_label_position(constraints, label)
                constraint = constraints[name]
                expr_str = self.print_single_constraint(constraint, coord)
                coords_str = self.print_coord(coord)
                cons_str = f"\t{coords_str}:\t{expr_str}\n"
                if name not in seen_constraints:
                    seen_constraints.append(name)
                    cons_str = f"\n{name}:\n{cons_str}"
                f.write(cons_str)
            f.write("\n\nVariables:\n")
            variables = self.model.variables
            for label in self.var_labels:
                pos = variables.get_label_position(label)
                if pos is not None:
                    name, coord = pos
                    var_str = f"\t{self.print_coord(coord)}:\t{self.var_lines[label]}\n"
                    if name not in seen_variables:
                        seen_variables.append(name)
                        var_str = f"\n{name}:\n{var_str}"
                else:
                    var_str = f"\t{label}:\t{self.var_lines[label]}\n"
                f.write(var_str)

    def write_gurobi_iis(self):
        """Writes IIS to file."""
        # get the gurobi model
        gurobi_model = self.model.solver_model
        # write the IIS
        gurobi_model.computeIIS()
        gurobi_model.write(self.iis_file)

    def read_labels(self):
        """Reads the labels from the IIS file.

        :return: A list of labels
        """
        labels_c = []
        labels_v = []
        lines_v = {}
        with open(self.iis_file, "rb") as f:
            for line in f.readlines():
                line = line.decode()
                if line.startswith(" c"):
                    labels_c.append(int(line.split(":")[0][2:]))
                elif line.startswith(" x"):
                    pattern = r"\sx(\d+)\s(.*)"
                    match = re.match(pattern, line)
                    if match:
                        labels_v.append(int(match.group(1)))
                        lines_v[int(match.group(1))] = match.group(2).rstrip()
        return labels_c, labels_v, lines_v

    def print_single_constraint(self, constraint, coord):
        """Print a single constraint based on the constraint object.

        :param constraint: The constraint object
        :param coord: The coordinates of the constraint
        :return: The string representation of the constraint
        """
        coeffs, vars, sign, rhs = xr.broadcast(
            constraint.coeffs, constraint.vars, constraint.sign, constraint.rhs
        )
        coeffs = coeffs.sel(coord).values
        vars = vars.sel(coord).values
        sign = sign.sel(coord)[0].item()
        rhs = rhs.sel(coord)[0].item()

        expr = self.print_single_expression(coeffs, vars, self.model)
        # sign = self.SIGNS_pretty[sign]

        return f"{expr} {sign} {rhs:.12g}"

    @staticmethod
    def print_coord(coord):
        """Print the coordinates.

        :param coord: The coordinates to print
        :return: The string representation of the coordinates
        """
        if isinstance(coord, dict):
            coord = coord.values()
        return "[" + ", ".join([str(c) for c in coord]) + "]" if len(coord) else ""

    @staticmethod
    def get_label_position(constraints, value):
        """Get tuple of name and coordinate for variable labels.

        :param constraints: The constraints object
        :param value: The value to get the label for
        :return: The name and coordinate of the variable
        """
        name = constraints.get_name_by_label(value)
        con = constraints[name]
        indices = [i[0] for i in np.where(con.labels.values == value)]

        # Extract the coordinates from the indices
        coord = {
            dim: con.labels.indexes[dim][i]
            for dim, i in zip(con.labels.dims, indices, strict=False)
        }

        return name, coord

    @staticmethod
    def print_single_expression(c, v, model):
        """This is a linopy routine but without max terms
        Print a single linear expression based on the coefficients and variables.

        :param c: The coefficients of the expression
        :param v: The variables of the expression
        :param model: The model to which the expression belongs
        :return: The string representation of the expression
        """

        # catch case that to many terms would be printed
        def print_line(expr):
            res = []
            for i, (coeff, (name, coord)) in enumerate(expr):
                coord_string = IISConstraintParser.print_coord(coord)
                if i:
                    # split sign and coefficient
                    coeff_string = f"{float(coeff):+.4}"
                    res.append(
                        f"{coeff_string[0]} {coeff_string[1:]} {name}{coord_string}"
                    )
                else:
                    res.append(f"{float(coeff):.4} {name}{coord_string}")
            return " ".join(res) if len(res) else "None"

        if isinstance(c, np.ndarray):
            mask = v != -1
            c, v = c[mask], v[mask]

        expr = list(zip(c, model.variables.get_label_position(v), strict=False))
        return print_line(expr)
