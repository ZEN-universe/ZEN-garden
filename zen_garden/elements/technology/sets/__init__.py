"""Technology set specifications."""

from zen_garden.topology.generic_set import GenericSet


def _energy_system_attribute(attribute):
    def get_data(cls, constructor):
        return getattr(constructor.energy_system, attribute)

    return classmethod(get_data)


class SetConversionTechnologies(GenericSet):
    name = "set_conversion_technologies"
    doc = "Set of conversion technologies"
    get_data = _energy_system_attribute(name)


class SetRetrofittingTechnologies(GenericSet):
    name = "set_retrofitting_technologies"
    doc = "Set of retrofitting technologies"
    get_data = _energy_system_attribute(name)


class SetTransportTechnologies(GenericSet):
    name = "set_transport_technologies"
    doc = "Set of transport technologies"
    get_data = _energy_system_attribute(name)


class SetStorageTechnologies(GenericSet):
    name = "set_storage_technologies"
    doc = "Set of storage technologies"
    get_data = _energy_system_attribute(name)


class SetTechnologiesExisting(GenericSet):
    name = "set_technologies_existing"
    doc = "Set of existing technology vintages"
    index_set = "set_technologies"

    @classmethod
    def get_data(cls, constructor):
        return constructor.element_registry.get_attribute_of_all_elements(
            constructor.element_class, cls.name
        )


class SetReferenceCarriers(GenericSet):
    name = "set_reference_carriers"
    doc = "Reference carriers indexed by technology"
    index_set = "set_technologies"

    @classmethod
    def get_data(cls, constructor):
        return constructor.element_registry.get_attribute_of_all_elements(
            constructor.element_class, "reference_carrier"
        )


TECHNOLOGY_SETS: list[type[GenericSet]] = [
    SetConversionTechnologies,
    SetRetrofittingTechnologies,
    SetTransportTechnologies,
    SetStorageTechnologies,
    SetTechnologiesExisting,
    SetReferenceCarriers,
]
