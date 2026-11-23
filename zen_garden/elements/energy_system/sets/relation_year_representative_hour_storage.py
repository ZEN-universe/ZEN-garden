from zen_garden.elements.energy_system.sets.set_representative_hours_storage import (
    SetRepresentativeHoursStorage,
)
from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.model.component_types.set import GenericRelationSet


class RelationYearRepresentativeHourStorage(GenericRelationSet):
    """Representative storage hours belonging to each modeled year."""

    name = "relation_year_representative_hour_storage"
    dimensions = (SetYears, SetRepresentativeHoursStorage)

    @classmethod
    def get_combinations(cls, model_constructor):
        representative_hours = model_constructor.time_steps.representative_hours_storage
        assert representative_hours is not None
        return {
            year: representative_hours[year]
            for year in model_constructor.model_schema.set_years
        }
