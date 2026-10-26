from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.model.component_types.set import GenericSet


class SetRepresentativeHoursStorage(GenericSet):
    """Storage representative hours indexed by modeled year."""

    name = "set_representative_hours_storage"
    symbol = r"\tilde{r} \in \widetilde{\mathcal{R}}_{y}"
    doc = "Set of representative storage hours used to track storage levels."
    index_set = SetYears

    @classmethod
    def get_data(cls, model_constructor):
        representative_hours = model_constructor.time_steps.representative_hours_storage
        assert representative_hours is not None
        return {
            year: representative_hours[year]
            for year in model_constructor.model_schema.set_years
        }
