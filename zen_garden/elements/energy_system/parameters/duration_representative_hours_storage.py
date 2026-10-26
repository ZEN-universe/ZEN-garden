from zen_garden.model.component_types.parameter import GenericParameter


class DurationRepresentativeHoursStorage(GenericParameter):
    """Duration of each storage representative hour."""

    name = "duration_representative_hours_storage"
    symbol = r"\tau^{\mathrm{stor}}_{\tilde{r}}"
    indices = ("set_years", "set_representative_hours_storage")
    doc = "Duration represented by a storage time step."
    unit_category = {"time": 1}
    squeeze_singleton = False

    @classmethod
    def get_input_data(cls, element):
        """Retain durations populated by time-series aggregation."""
        return None
