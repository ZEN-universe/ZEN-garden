from zen_garden.model.component_types.set import GenericDomainSet


class SetRepresentativeHoursStorage(GenericDomainSet):
    """Canonical domain of storage representative hours."""

    name = "set_representative_hours_storage"
    symbol = r"\tilde{r} \in \widetilde{\mathcal{R}}_{y}"
    doc = "Set of representative storage hours used to track storage levels."

    @classmethod
    def get_data(cls, model_constructor):
        representative_hours = model_constructor.time_steps.representative_hours_storage
        assert representative_hours is not None
        return list(
            dict.fromkeys(
                hour
                for year in model_constructor.model_schema.set_years
                for hour in representative_hours[year]
            )
        )
