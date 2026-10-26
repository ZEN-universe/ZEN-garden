import pandas as pd

from zen_garden.model.component_types.parameter import (
    GenericParameter,
    ParameterInputData,
)


class ConversionFactor(GenericParameter):
    """Conversion factor."""

    name = "conversion_factor"
    symbol = r"\eta^{\mathrm{conv}}_{h,c,t}"
    indices = (
        "set_conversion_technologies",
        "set_dependent_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )
    doc = (
        "Parameter which specifies the slope of the conversion efficiency if "
        "approximated linearly."
    )
    unit_category = {}
    time_series = True
    input_indices = ("set_nodes", "set_hours")

    @classmethod
    def get_input_data(cls, element):
        """Load one conversion-factor series per dependent carrier."""
        dependent_carriers = list(
            set(element.input_carrier + element.output_carrier).difference(
                element.reference_carrier
            )
        )
        if not dependent_carriers:
            return ParameterInputData(None, None)

        values = {
            carrier: element.element_data_loader.extract_input_data(
                cls.input_name or cls.name,
                index_sets=cls._input_indices(element),
                unit_category=cls.unit_category,
                subelement=carrier,
            )
            for carrier in dependent_carriers
        }
        combined = pd.DataFrame.from_dict(values)
        combined.columns.name = "carrier"
        combined = combined.stack()
        levels = [combined.index.names[-1], *combined.index.names[:-1]]
        return ParameterInputData(
            combined.reorder_levels(levels), element.units.get(cls.name)
        )
