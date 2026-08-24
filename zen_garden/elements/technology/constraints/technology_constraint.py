from abc import ABC
from typing import cast

import numpy as np

from zen_garden.topology.generic_constraint import GenericConstraint


class TechnologyConstraint(GenericConstraint, ABC):
    def get_lifetime_range(self, tech, year, use_depreciation_time=False):
        """Get active year range of technology: either lifetime or depreciation time.

        :param tech: name of the technology
        :param year: yearly time step
        :param use_depreciation_time: boolean indicating whether to use depreciation
            time instead of lifetime, namely for CAPEX calculation
        :return: lifetime or depreciation time range of technology
        """
        first_lifetime_year = self.get_first_lifetime_time_step(
            tech,
            year,
            use_depreciation_time,
        )
        first_lifetime_year = max(
            first_lifetime_year,
            cast(int, self.zen_model.sets["set_years"][0]),
        )
        return range(first_lifetime_year, year + 1)

    def get_first_lifetime_time_step(self, tech, year, use_depreciation_time=False):
        """Get first time step of active capacity of technology.

        Returns the first time step within the lifetime or depreciation time of the
        technology, i.e., the earliest past time step whose installed capacity is
        still active at the given time step.

        :param tech: name of the technology
        :param year: current yearly time step
        :param use_depreciation_time: boolean indicating whether to use depreciation
            time instead of standard lifetime for capacity calculation
        :return: first time step where capacity or investment is still valid
        """
        # get params and system
        params = self.zen_model.parameters.dict_parameters
        lifetime = (
            params.depreciation_time[tech]
            if use_depreciation_time
            else params.lifetime[tech]
        )
        # conservative estimate of lifetime (floor)
        del_lifetime = (
            int(np.floor(lifetime / self.config.system.interval_between_years)) - 1
        )
        return year - del_lifetime
