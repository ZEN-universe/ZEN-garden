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

    def add_constraint(self, name, constraint, doc=""):
        """Initialization of a constraint.

        :param name: name of variable
        :param constraint: a linopy constraint or a dictionary of constraints or None
        :param doc: docstring of variable
        """
        if name in self.docs.keys():
            logger.warning(f"{name} already added. Can only be added once")
            return

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
                self._add_single_constraint(_name, cons)
                self.docs[name] = self.compile_doc_string(
                    doc, index_list=list(cons.indexes), name=_name
                )
        elif isinstance(constraint, lp.constraints.Constraint) or isinstance(
            constraint, lp.constraints.AnonymousConstraint
        ):
            self._add_single_constraint(name, constraint)
            self.docs[name] = self.compile_doc_string(
                doc, index_list=list(constraint.indexes), name=name
            )
        else:
            raise TypeError(
                f"Constraint {name} has wrong format. Must be either a linopy "
                f"constraint or a dictionary of constraints but "
                f"is {type(constraint).__name__}"
            )

    def _add_single_constraint(self, name, constraint):
        """Adds a single constraint to the model.

        :param name: name of variable
        :param constraint: linopy constraint
        """
        lhs = constraint.lhs
        sign = constraint.sign
        rhs = constraint.rhs
        mask = constraint.mask
        self._add_con(name, lhs, sign, rhs, mask=mask)

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
