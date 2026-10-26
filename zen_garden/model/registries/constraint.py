"""Class defining constraints for the optimization model."""

import logging

import linopy as lp
import numpy as np
import xarray as xr
from linopy import Model as LinopyModel

from zen_garden.model.registries.base import Registry
from zen_garden.model.registries.set_registry import SetRegistry

logger = logging.getLogger(__name__)


class ConstraintRegistry(Registry):
    def __init__(self, lp_model: "LinopyModel"):
        """Initialization of a constraint.

        :param index_sets: A reference to the index sets of the model
        :param model: A reference to the linopy model
        """
        self.lp_model = lp_model
        super().__init__()

    def add_constraint(self, name, constraint, doc="", mask=None):
        """Initialization of a constraint.

        :param name: name of variable
        :param constraint: a linopy constraint or a dictionary of constraints or None
        :param doc: docstring of variable
        """
        if name in self.docs.keys():
            raise ValueError(
                f"Constraint {name!r} already added. Can only be added once"
            )

        if constraint is None or constraint == []:
            return
        elif isinstance(constraint, dict):
            for key, cons in constraint.items():
                if cons is None or cons == []:
                    return
                assert isinstance(cons, lp.constraints.Constraint) or isinstance(
                    cons, lp.constraints.AnonymousConstraint
                ), (
                    f"Constraint {key} has wrong format. "
                    f"Must be a linopy constraint but is {type(cons).__name__}"
                )
                if isinstance(key, tuple):
                    _key = "-".join([str(k) for k in key])
                else:
                    _key = str(key)
                _name = f"{name}--{key}"
                self._add_single_constraint(_name, cons, mask=mask)
                self.docs[name] = self.compile_doc_string(
                    doc, index_list=list(cons.indexes), name=_name
                )
        elif isinstance(constraint, lp.constraints.Constraint) or isinstance(
            constraint, lp.constraints.AnonymousConstraint
        ):
            self._add_single_constraint(name, constraint, mask=mask)
            self.docs[name] = self.compile_doc_string(
                doc, index_list=list(constraint.indexes), name=name
            )
        else:
            raise TypeError(
                f"Constraint {name} has wrong format. Must be either a linopy "
                f"constraint or a dictionary of constraints but "
                f"is {type(constraint).__name__}"
            )

    def _add_single_constraint(self, name, constraint, mask=None):
        """Adds a single constraint to the model.

        :param name: name of variable
        :param constraint: linopy constraint
        """
        # A row has solver support only where at least one active variable label
        # remains. Coefficients of every active term must be finite.
        active_terms = constraint.lhs.vars >= 0
        variable_support = active_terms.any("_term")
        finite_active_coefficients = (
            np.isfinite(constraint.lhs.coeffs).where(active_terms, True).all("_term")
        )
        rhs = constraint.rhs
        finite_rhs = np.isfinite(rhs)
        valid_sign = constraint.sign.isin(("=", "==", "<=", ">="))

        constant_rows = ~variable_support & finite_rhs & valid_sign
        constant_satisfied = xr.where(
            constraint.sign.isin(("=", "==")),
            rhs == 0,
            xr.where(constraint.sign == "<=", rhs >= 0, rhs <= 0),
        )
        violated_constants = constant_rows & ~constant_satisfied
        if bool(violated_constants.any().item()):
            violated = violated_constants.to_series()
            locations = violated.index[violated].tolist()
            raise ValueError(
                f"Constraint {name!r} contains violated constant-only rows at "
                f"{locations!r}"
            )

        finite_mask = (
            variable_support & finite_active_coefficients & finite_rhs & valid_sign
        )
        if constraint.mask is not None:
            finite_mask &= constraint.mask
        if mask is not None:
            finite_mask &= mask
        if finite_mask.ndim == 0 and not bool(finite_mask.item()):
            return

        # Linopy/scaling may inspect masked data, so sanitize only rows excluded
        # by the validated mask while preserving all active solver rows.
        data = constraint.data.assign(
            coeffs=constraint.lhs.coeffs.where(finite_mask, 0),
            rhs=rhs.where(finite_mask, 0),
            sign=constraint.sign.where(valid_sign, "="),
        )
        constraint = lp.constraints.Constraint(data, self.lp_model, skip_broadcast=True)
        self.lp_model.add_constraints(constraint, name=name, mask=finite_mask)

    def _add_con(self, name, lhs, sign, rhs, mask=None):
        """Adds a constraint to the model.

        :param name: name of the constraint
        :param lhs: left hand side of the constraint
        :param sign: sign of the constraint
        :param rhs: right hand side of the constraint
        :param mask: An optional mask to only add the constraint for certain indices
        """
        # get the mask, where rhs is not nan and rhs is finite
        if mask is not None:
            mask = ~np.isnan(rhs) & np.isfinite(rhs) & mask
        else:
            mask = ~np.isnan(rhs) & np.isfinite(rhs)
        # turn scalar masks into bool (otherwise it will use np.bool)
        if isinstance(mask, np.bool_):
            mask = bool(mask)
        else:
            self.lp_model.add_constraints(lhs, sign, rhs, name=name, mask=mask)

    def reorder_group(
        self, lhs, sign, rhs, index_values, index_names, model, drop=None
    ):
        """Reorders constraints in a group to get full shape based on indexes and names.

        :param lhs: The lhs of the constraints
        :param sign: The sign of the constraints, can be None if only lhs should
            be restructured
        :param rhs: The rhs of the constraints, can be None if only lhs should
            be restructured
        :param index_values: The index values corresponding to the group numbers
        :param index_names: The index names of the indices
        :param model: The model
        :param drop: Which group to drop (the dummy group
        :return: An anonymous constraint
        """
        # drop if necessary
        lhs = lhs.data
        if drop is not None:
            lhs = lhs.drop_sel(group=drop, errors="ignore")
            rhs = rhs.drop_sel(group=drop, errors="ignore")
            sign = sign.drop_sel(group=drop, errors="ignore")

        # drop the unncessessary dimensions
        lhs = lhs.drop_vars(list(set(lhs.coords) - set(lhs.dims)))

        # get the coordinates
        index_arrs = SetRegistry.tuple_to_arr(index_values, index_names)
        coords = {
            name: np.unique(arr.data)
            for name, arr in zip(index_names, index_arrs, strict=False)
        }
        coords.update(
            {
                cname: lhs.coords[cname]
                for cname in lhs.coords
                if cname != "group" and cname != "_term"
            }
        )
        coords_shape = tuple(len(c) for c in coords.values())
        dims = index_names + list(lhs.dims)
        dims.remove("group")

        # create the full arrays, note that the lhs needs a _term dimension
        xr_coeffs = xr.DataArray(
            np.full(shape=coords_shape + (lhs.coeffs.shape[-1],), fill_value=np.nan),
            dims=dims,
            coords=coords,
        )
        xr_vars = xr.DataArray(
            np.full(shape=coords_shape + (lhs.vars.shape[-1],), fill_value=-1),
            dims=dims,
            coords=coords,
        )

        # rhs and sign do not have a _term dimension
        xr_rhs = xr.DataArray(
            np.full(shape=coords_shape, fill_value=np.nan),
            dims=dims[:-1],
            coords=coords,
        )
        xr_sign = xr.DataArray(
            np.full(shape=coords_shape, fill_value="="), dims=dims[:-1], coords=coords
        ).astype("U2")

        # Assign everything
        for num, index_val in enumerate(index_values):
            if num in lhs.coords["group"]:
                xr_coeffs.loc[index_val] = lhs.coeffs.sel(group=num).data
                xr_vars.loc[index_val] = lhs.vars.sel(group=num).data
                if rhs is not None:
                    xr_rhs.loc[index_val] = rhs.sel(group=num).data
                if sign is not None:
                    xr_sign.loc[index_val] = sign.sel(group=num).data

        if rhs is None and sign is None:
            return lp.LinearExpression(
                xr.Dataset({"coeffs": xr_coeffs, "vars": xr_vars}), model
            )
        else:
            # to full arrays
            xr_lhs = xr.Dataset(
                {"coeffs": xr_coeffs, "vars": xr_vars, "sign": xr_sign, "rhs": xr_rhs}
            )
            return lp.constraints.Constraint(xr_lhs, model)
