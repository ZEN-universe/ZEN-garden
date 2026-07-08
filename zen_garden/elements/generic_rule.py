from typing import TYPE_CHECKING, cast

import linopy as lp
import pandas as pd
import xarray as xr
from linopy.expressions import LinearExpression

if TYPE_CHECKING:
    from zen_garden.elements.energy_system import EnergySystem
    from zen_garden.model.config import Config
    from zen_garden.model.zen_model import ZenModel


class GenericRule(object):
    """This class implements a generic rule for the model, which can be used to init the
    other rules of the technologies and carriers.
    """

    def __init__(
        self,
        config: "Config",
        zen_model: "ZenModel",
        energy_system: "EnergySystem",
    ):
        """Constructor for generic rule.

        :param optimization_setup: The optimization setup to use for the setup
        """
        self.config = config
        self.zen_model = zen_model
        self.energy_system = energy_system

    # helper methods for constraint rules
    def get_year_time_step_array(self, storage=False):
        """Returns array with year and time steps of each year.

        :param storage: boolean indicating if object is a storage object
        """
        # create times xarray with 1 where the operation time step is in the year
        if storage:
            meth = self.energy_system.time_steps.get_time_steps_year2storage
            time_step_name = "set_time_steps_storage"
        else:
            meth = self.energy_system.time_steps.get_time_steps_year2operation
            time_step_name = "set_time_steps_operation"
        times = [
            (y, t)
            for y in self.zen_model.sets["set_time_steps_yearly"]
            for t in meth(y)
        ]
        times = pd.MultiIndex.from_tuples(times)
        times.names = ["set_time_steps_yearly", time_step_name]
        times = pd.Series(index=times, data=1)
        times = times.to_xarray()
        times = times.fillna(0.0)
        return times

    def get_year_time_step_duration_array(self):
        """Returns array with year and duration of time steps of each year."""
        times = self.get_year_time_step_array()
        time_steps_operation_duration = cast(
            xr.DataArray | None, self.zen_model.parameters.time_steps_operation_duration
        )
        assert time_steps_operation_duration is not None
        times = times * time_steps_operation_duration
        return times

    def get_previous_storage_time_step_array(self):
        """Returns array with storage time steps and previous storage time steps."""
        times_prev = []
        mask = []
        for ts in self.zen_model.sets["set_time_steps_storage"]:
            ts_end = self.energy_system.time_steps.get_time_steps_storage_startend(ts)
            if ts_end is not None:
                if self.config.system.storage_periodicity:
                    times_prev.append(ts_end)
                    mask.append(True)
                else:
                    times_prev.append(ts)
                    mask.append(False)
            else:
                ts_prev = self.energy_system.time_steps.get_previous_storage_time_step(
                    ts
                )
                times_prev.append(ts_prev)
                mask.append(True)
        mask = xr.DataArray(
            mask,
            dims="set_time_steps_storage",
            coords={
                "set_time_steps_storage": self.zen_model.sets["set_time_steps_storage"]
            },
        )
        return times_prev, mask

    def get_power2energy_time_step_array(self):
        """Returns array with power2energy time steps."""
        times = {
            st: self.energy_system.time_steps.convert_time_step_energy2power(st)
            for st in self.zen_model.sets["set_time_steps_storage"]
        }
        times = pd.Series(times, name="set_time_steps_operation")
        times.index.name = "set_time_steps_storage"
        return times

    def get_storage2year_time_step_array(self):
        """Returns array with storage2year time steps."""
        times = {
            st: y
            for y in self.zen_model.sets["set_time_steps_yearly"]
            for st in self.energy_system.time_steps.get_time_steps_year2storage(y)
        }
        times = pd.Series(times, name="set_time_steps_yearly")
        times.index.name = "set_time_steps_storage"
        return times

    def map_and_expand(self, array, mapping):
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

    def align_and_mask(self, expr, mask):
        """Aligns and masks expr.

        :param expr: expression to align and mask
        :param mask: mask to apply
        """
        if isinstance(expr, xr.DataArray):
            aligner = expr
        elif isinstance(expr, lp.Variable):
            aligner = expr.lower
        else:
            aligner = expr.const
        mask = xr.align(mask, aligner, join="right")[0]
        expr = expr.where(mask)
        return expr

    def get_flow_expression_conversion(self, techs, nodes, factor=None, rename=False):
        """Return the flow expression for conversion technologies."""
        reference_flows = []
        for t in techs:
            rc = self.zen_model.sets["set_reference_carriers"][t][0]
            if factor is not None:
                mult = factor.loc[t, nodes]
            else:
                mult = 1
            # TODO can we avoid the indexing here?
            if rc in self.zen_model.sets["set_input_carriers"][t]:
                reference_flows.append(
                    mult
                    * self.zen_model.lp_model.variables["flow_conversion_input"].loc[
                        t, rc, nodes, :
                    ]
                )
            else:
                reference_flows.append(
                    mult
                    * self.zen_model.lp_model.variables["flow_conversion_output"].loc[
                        t, rc, nodes, :
                    ]
                )
        if rename:
            term_reference_flow = lp.merge(
                reference_flows,
                dim="set_technologies",
                join="outer",
                coords="minimal",
                compat="override",
                cls=LinearExpression,
            ).rename({"set_nodes": "set_location"})
        else:
            term_reference_flow = lp.merge(
                reference_flows,
                dim="set_conversion_technologies",
                join="outer",
                coords="minimal",
                compat="override",
                cls=LinearExpression,
            )
        return term_reference_flow

    def get_flow_expression_storage(self, rename=True):
        """Return the flow expression for storage technologies."""
        term = (
            self.zen_model.lp_model.variables["flow_storage_charge"]
            + self.zen_model.lp_model.variables["flow_storage_discharge"]
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
