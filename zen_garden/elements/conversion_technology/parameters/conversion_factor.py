import pandas as pd

from zen_garden.topology.generic_parameter import GenericParameter


class ConversionFactor(GenericParameter):
    """Conversion factor."""

    name = "conversion_factor"
    indices = (
        "set_conversion_technologies",
        "set_dependent_carriers",
        "set_nodes",
        "set_hours",
    )
    doc = "Conversion factor"
    unit_category = {}
    time_series = True
    input_indices = ("set_nodes", "set_hours")

    @classmethod
    def store_input_data(cls, element):
        """Load one conversion-factor series per dependent carrier."""
        dependent_carriers = list(
            set(element.input_carrier + element.output_carrier).difference(
                element.reference_carrier
            )
        )
        if not dependent_carriers:
            cls._store_value(element, cls.name, None)
            return

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
        cls._store_value(element, cls.name, combined.reorder_levels(levels))
