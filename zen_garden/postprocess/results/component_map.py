from dataclasses import dataclass

from zen_garden.postprocess.results.component_type import ComponentType


@dataclass
class ComponentMap:
    sets: list[str]
    variable: list[str]
    parameter: list[str]
    dual: list[str]
    reduced_cost: list[str]

    @property
    def all_components(self) -> list[str]:
        """Get a list of all components."""
        return (
            self.sets + self.variable + self.parameter + self.dual + self.reduced_cost
        )

    def find_type(self, component_name: str) -> ComponentType:
        """Find the type of a component given its name.

        :param component_name: The name of the component.
        :return: The type of the component.
        """
        if component_name in self.sets:
            return ComponentType.sets
        elif component_name in self.variable:
            return ComponentType.variable
        elif component_name in self.parameter:
            return ComponentType.parameter
        elif component_name in self.dual:
            return ComponentType.dual
        elif component_name in self.reduced_cost:
            return ComponentType.reduced_costs
        else:
            raise KeyError(
                (
                    f"Component {component_name} not found in scenario. "
                    f"Available components: {self.all_components}"
                )
            )
