from zen_garden.topology.generic_parameter import GenericComputedParameters


class LifetimeExisting(GenericComputedParameters):
    """Parameter specifying the remaining lifetime of an existing technology."""

    name = "lifetime_existing"
    indices = ("set_technologies", "set_location", "set_technologies_existing")
    doc = "Parameter specifying the remaining lifetime of an existing technology"
    unit_category = {}
    input_loader = "existing_lifetime"
    dependencies = ["lifetime"]

    @classmethod
    def store_input_data(cls, element, loader):
        loader.load_into(cls, element)
