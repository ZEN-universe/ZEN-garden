from zen_garden.topology.generic_parameter import GenericParameter


class LifetimeExisting(GenericParameter):
    """Parameter specifying the remaining lifetime of an existing technology."""

    name = "lifetime_existing"
    indices = ("set_technologies", "set_location", "set_technologies_existing")
    doc = "Parameter specifying the remaining lifetime of an existing technology"
    unit_category = {}
    input_loader = "existing_lifetime"
    input_dependencies = ("lifetime",)
