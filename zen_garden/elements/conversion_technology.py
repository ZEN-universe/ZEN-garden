"""Class defining the parameters, variables, and constraints of the conversion
technologies. The class takes the abstract optimization model as an input and adds
parameters, variables, and constraints of the conversion technologies.
"""

import logging
from typing import cast, override

import numpy as np
import pandas as pd

from zen_garden.elements.technology import Technology

logger = logging.getLogger(__name__)


class ConversionTechnology(Technology):
    """Class defining conversion technologies."""

    # set label
    label = "set_conversion_technologies"
    location_type = "set_nodes"

    @override
    def _initialize(self):
        """Retrieves and stores information on reference, input and output carriers."""
        # get reference carrier from class <Technology>
        super().initialize_reference_carrier()
        # define input and output carrier
        self.input_carrier = cast(
            list[str], self.data_input.extract_carriers(carrier_type="input_carrier")
        )
        self.output_carrier = cast(
            list[str], self.data_input.extract_carriers(carrier_type="output_carrier")
        )
        self.energy_system.set_technology_of_carrier(
            self.name, self.input_carrier + self.output_carrier
        )
        # check if reference carrier in input and output carriers and
        #   set technology to correspondent carrier
        self.input_data_checks.check_carrier_configuration(
            input_carrier=self.input_carrier,
            output_carrier=self.output_carrier,
            reference_carrier=self.reference_carrier,
            name=self.name,
        )

    def store_input_data(self):
        """Retrieves and stores input data for element as attributes.

        Each Child class overwrites method to store different attributes.
        """
        # get attributes from class <Technology>
        super().store_input_data()
        # get conversion efficiency and capex
        self.get_conversion_factor()
        self.opex_specific_fixed = self.data_input.extract_input_data(
            "opex_specific_fixed",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"money": 1, "energy_quantity": -1, "time": 1},
        )
        self.min_full_load_hours_fraction = self.data_input.extract_input_data(
            "min_full_load_hours_fraction",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={},
        )

        self.convert_to_fraction_of_capex()

    def get_conversion_factor(self):
        """Retrieves and stores conversion_factor."""
        dependent_carrier = list(
            set(self.input_carrier + self.output_carrier).difference(
                self.reference_carrier
            )
        )
        if not dependent_carrier:
            self.raw_time_series["conversion_factor"] = None
        else:
            index_sets = ["set_nodes", "set_time_steps"]
            time_steps = "set_base_time_steps_yearly"
            cf_dict = {}
            for carrier in dependent_carrier:
                cf_dict[carrier] = self.data_input.extract_input_data(
                    "conversion_factor",
                    index_sets=index_sets,
                    unit_category=None,
                    time_steps=time_steps,
                    subelement=carrier,
                )
            cf_dict = pd.DataFrame.from_dict(cf_dict)
            cf_dict.columns.name = "carrier"
            cf_dict = cf_dict.stack()
            conversion_factor_levels = [cf_dict.index.names[-1]] + cf_dict.index.names[
                :-1
            ]
            level_positions = [
                cf_dict.index.names.index(level_name)
                for level_name in conversion_factor_levels
            ]
            cf_dict = cf_dict.reorder_levels(level_positions)
            # extract yearly variation
            self.data_input.extract_yearly_variation("conversion_factor", index_sets)
            self.raw_time_series["conversion_factor"] = cf_dict

    def convert_to_fraction_of_capex(self):
        """This method retrieves the total capex and converts it to annualized capex."""
        pwa_capex, self.capex_is_pwa = self.data_input.extract_pwa_capex()
        assert pwa_capex is not None
        # annualize cost_capex_overnight
        fraction_year = self.calculate_fraction_of_year()
        self.opex_specific_fixed = self.opex_specific_fixed * fraction_year
        if not self.capex_is_pwa:
            self.capex_specific_conversion = pwa_capex["capex"] * fraction_year
        else:
            self.pwa_capex = pwa_capex
            self.pwa_capex["capex"] = [
                value * fraction_year for value in self.pwa_capex["capex"]
            ]
            # set bounds
            self.pwa_capex["bounds"]["capex"] = tuple(
                [(bound * fraction_year) for bound in self.pwa_capex["bounds"]["capex"]]
            )
        # calculate capex of existing capacity
        self.capex_capacity_existing = self.calculate_capex_of_capacities_existing()

    def calculate_capex_of_single_capacity(self, capacity, index, **kwargs):
        """This method calculates the annualized capex of a single existing capacity.

        :param capacity: existing capacity of technology
        :param index: index of capacity specifying node and time
        :return: annualized capex of a single existing capacity
        """
        if capacity == 0:
            return 0
        # linear
        if not self.capex_is_pwa:
            capex = self.capex_specific_conversion[index[0]].iloc[0] * capacity
        else:
            capex = np.interp(
                capacity, self.pwa_capex["capacity"], self.pwa_capex["capex"]
            )
        return capex
