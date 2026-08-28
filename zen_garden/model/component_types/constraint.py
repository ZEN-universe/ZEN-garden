"""Generic constraint class for ZenModel."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import pandas as pd
import xarray as xr
from linopy import Variable, merge
from linopy.expressions import LinearExpression

from zen_garden.model.registries.set import IndexedSet

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor


class GenericConstraint(ABC):
    """Base class for constraints.

    Constraints are stateless: :meth:`build` is a classmethod that takes the
    :class:`~zen_garden.model.constructor.ModelConstructor` and adds the
    constraint's rows to the optimization model. The helper methods below are
    stateless too.
    """

    @classmethod
    @abstractmethod
    def build(cls, model_constructor: "ModelConstructor") -> None:
        """Build the constraint(s) and add them to the optimization model."""

    # helper methods for constraint rules
    @staticmethod
    def get_year_time_step_array(model_constructor: "ModelConstructor", storage=False):
        """Returns array with year and time steps of each year.

        :param storage: boolean indicating if object is a storage object
        """
        time_steps = model_constructor.time_steps
        # create times xarray with 1 where the operation time step is in the year
        if storage:
            meth = time_steps.get_time_steps_year2storage
            time_step_name = "set_time_steps_storage"
        else:
            meth = time_steps.get_time_steps_year2operation
            time_step_name = "set_time_steps_operation"
        years = model_constructor.zen_model.sets["set_years"]
        year_ts_pairs = [(y, t) for y in years for t in meth(y)]
        index = pd.MultiIndex.from_tuples(
            year_ts_pairs, names=["set_years", time_step_name]
        )
        times = pd.Series(index=index, data=1).to_xarray()
        return times.fillna(0.0)

    @staticmethod
    def get_year_time_step_duration_array(model_constructor: "ModelConstructor"):
        """Returns array with year and duration of time steps of each year."""
        times = GenericConstraint.get_year_time_step_array(model_constructor)
        time_steps_operation_duration = cast(
            xr.DataArray | None,
            model_constructor.zen_model.parameters.time_steps_operation_duration,
        )
        assert time_steps_operation_duration is not None
        times = times * time_steps_operation_duration
        return times

    @staticmethod
    def get_previous_storage_time_step_array(model_constructor: "ModelConstructor"):
        """Returns array with storage time steps and previous storage time steps."""
        zen_model = model_constructor.zen_model
        time_steps = model_constructor.time_steps
        times_prev = []
        mask_values = []
        for ts in zen_model.sets["set_time_steps_storage"]:
            ts_end = time_steps.get_time_steps_storage_startend(ts)
            if ts_end is not None:
                if model_constructor.config.system.storage_periodicity:
                    times_prev.append(ts_end)
                    mask_values.append(True)
                else:
                    times_prev.append(ts)
                    mask_values.append(False)
            else:
                ts_prev = time_steps.get_previous_storage_time_step(ts)
                times_prev.append(ts_prev)
                mask_values.append(True)
        mask = xr.DataArray(
            mask_values,
            dims="set_time_steps_storage",
            coords={"set_time_steps_storage": zen_model.sets["set_time_steps_storage"]},
        )
        return times_prev, mask

    @staticmethod
    def get_power2energy_time_step_array(model_constructor: "ModelConstructor"):
        """Returns array with power2energy time steps."""
        zen_model = model_constructor.zen_model
        mapping = {
            st: model_constructor.time_steps.convert_time_step_energy2power(st)
            for st in zen_model.sets["set_time_steps_storage"]
        }
        times = pd.Series(mapping, name="set_time_steps_operation")
        times.index.name = "set_time_steps_storage"
        return times

    @staticmethod
    def get_storage2year_time_step_array(model_constructor: "ModelConstructor"):
        """Returns array with storage2year time steps."""
        zen_model = model_constructor.zen_model
        mapping = {
            st: y
            for y in zen_model.sets["set_years"]
            for st in model_constructor.time_steps.get_time_steps_year2storage(y)
        }
        times = pd.Series(mapping, name="set_years")
        times.index.name = "set_time_steps_storage"
        return times

    @staticmethod
    def map_and_expand(array, mapping):
        """Maps and expands array.

        :param array: xarray to map and expand
        :param mapping: pd.Series with mapping values
        """
        assert isinstance(mapping, pd.Series) or isinstance(
            mapping.index, pd.Index
        ), "Mapping must be a pd.Series or with a single-level pd.Index"
        # get mapping values
        array = array.sel({mapping.name: mapping.values})
        # rename
        array = array.rename({mapping.name: mapping.index.name})
        # assign coordinates
        array = array.assign_coords({mapping.index.name: mapping.index})
        return array

    @staticmethod
    def align_and_mask(expr, mask):
        """Aligns and masks expr.

        :param expr: expression to align and mask
        :param mask: mask to apply
        """
        if isinstance(expr, xr.DataArray):
            aligner = expr
        elif isinstance(expr, Variable):
            aligner = expr.lower
        else:
            aligner = expr.const
        mask = xr.align(mask, aligner, join="right")[0]
        expr = expr.where(mask)
        return expr

    @staticmethod
    def get_flow_expression_conversion(
        model_constructor: "ModelConstructor", techs, nodes, factor=None, rename=False
    ):
        """Return the flow expression for conversion technologies."""
        zen_model = model_constructor.zen_model
        reference_carriers = cast(IndexedSet, zen_model.sets["set_reference_carriers"])
        input_carriers = cast(IndexedSet, zen_model.sets["set_input_carriers"])
        reference_flows = []
        for t in techs:
            rc = reference_carriers[t][0]
            if factor is not None:
                mult = factor.loc[t, nodes]
            else:
                mult = 1
            # TODO can we avoid the indexing here?
            if rc in input_carriers[t]:
                reference_flows.append(
                    mult
                    * zen_model.variables["flow_conversion_input"].loc[t, rc, nodes, :]
                )
            else:
                reference_flows.append(
                    mult
                    * zen_model.variables["flow_conversion_output"].loc[t, rc, nodes, :]
                )
        if rename:
            term_reference_flow = merge(
                reference_flows,
                dim="set_technologies",
                join="outer",
                coords="minimal",
                compat="override",
                cls=LinearExpression,
            ).rename({"set_nodes": "set_location"})
        else:
            term_reference_flow = merge(
                reference_flows,
                dim="set_conversion_technologies",
                join="outer",
                coords="minimal",
                compat="override",
                cls=LinearExpression,
            )
        return term_reference_flow

    @staticmethod
    def get_flow_expression_storage(model_constructor: "ModelConstructor", rename=True):
        """Return the flow expression for storage technologies."""
        zen_model = model_constructor.zen_model
        term = (
            zen_model.variables["flow_storage_charge"]
            + zen_model.variables["flow_storage_discharge"]
        )
        if rename:
            return term.rename(
                {
                    "set_storage_technologies": "set_technologies",
                    "set_nodes": "set_location",
                }
            )
        else:
            return term
