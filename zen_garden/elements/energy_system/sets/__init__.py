"""Energy-system set specifications."""

from zen_garden.topology.generic_set import GenericSet


class SetNodes(GenericSet):
    name = "set_nodes"
    doc = "Set of nodes"

    @classmethod
    def get_data(cls, constructor):
        return constructor.energy_system.set_nodes


class SetEdges(GenericSet):
    name = "set_edges"
    doc = "Set of edges"

    @classmethod
    def get_data(cls, constructor):
        return constructor.energy_system.set_edges


class SetNodesOnEdges(GenericSet):
    name = "set_nodes_on_edges"
    doc = "Set of nodes that constitute an edge"
    index_set = "set_edges"

    @classmethod
    def get_data(cls, constructor):
        return constructor.energy_system.set_nodes_on_edges


class SetCarriers(GenericSet):
    name = "set_carriers"
    doc = "Set of carriers"

    @classmethod
    def get_data(cls, constructor):
        return constructor.energy_system.set_carriers


class SetTechnologies(GenericSet):
    name = "set_technologies"
    doc = "Set of technologies"

    @classmethod
    def get_data(cls, constructor):
        return constructor.energy_system.set_technologies


class SetElements(GenericSet):
    name = "set_elements"
    doc = "Set of elements"
    indexing_set = True

    @classmethod
    def get_data(cls, constructor):
        return list(
            set(constructor.energy_system.set_technologies)
            | set(constructor.energy_system.set_carriers)
        )


class SetHoursAllYears(GenericSet):
    name = "set_hours_all_years"
    doc = "Set of base time steps"

    @classmethod
    def get_data(cls, constructor):
        return constructor.energy_system.set_hours_all_years


class SetYears(GenericSet):
    name = "set_years"
    doc = "Set of yearly time steps"

    @classmethod
    def get_data(cls, constructor):
        return constructor.energy_system.set_years


class SetYearsEntireHorizon(GenericSet):
    name = "set_years_entire_horizon"
    doc = "Set of yearly time steps of the entire optimization horizon"

    @classmethod
    def get_data(cls, constructor):
        return constructor.energy_system.set_years_entire_horizon


class SetTimeStepsOperation(GenericSet):
    name = "set_time_steps_operation"
    doc = "Set of operational time steps"

    @classmethod
    def get_data(cls, constructor):
        return constructor.energy_system.time_steps.time_steps_operation


class SetTimeStepsStorage(GenericSet):
    name = "set_time_steps_storage"
    doc = "Set of storage level time steps"

    @classmethod
    def get_data(cls, constructor):
        return constructor.energy_system.time_steps.time_steps_storage


ENERGY_SYSTEM_SETS: list[type[GenericSet]] = [
    SetNodes,
    SetEdges,
    SetNodesOnEdges,
    SetCarriers,
    SetTechnologies,
    SetElements,
    SetHoursAllYears,
    SetYears,
    SetYearsEntireHorizon,
    SetTimeStepsOperation,
    SetTimeStepsStorage,
]
