"""Class defining retrofitting technologies."""

import logging

from zen_garden.elements.conversion_technology import ConversionTechnology

logger = logging.getLogger(__name__)


class RetrofittingTechnology(ConversionTechnology):
    """Class defining retrofitting technologies."""

    # set label
    label = "set_retrofitting_technologies"
    location_type = "set_nodes"

    def store_input_data(self):
        """Retrieves and stores input data for element as attributes.

        Each Child class overwrites method to store different attributes.
        """
        # get attributes from class <Technology>
        super().store_input_data()
        # get retrofit base technology
        self.retrofit_base_technology = (
            self.data_input.extract_retrofit_base_technology()
        )
        # get flow_coupling factor and capex
        self.raw_time_series["retrofit_flow_coupling_factor"] = (
            self.data_input.extract_input_data(
                "retrofit_flow_coupling_factor",
                index_sets=["set_nodes", "set_hours"],
                unit_category={},
            )
        )
