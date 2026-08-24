from enum import Enum


class ComponentType(Enum):
    parameter = "parameter"
    variable = "variable"
    dual = "dual"
    sets = "sets"
    reduced_costs = "reduced_costs"

    @classmethod
    def get_names(cls) -> list[str]:
        """Get a list of component type names."""
        return [component_type.value for component_type in cls]

    @classmethod
    def get_file_names_maps(cls) -> dict[str, "ComponentType"]:
        """Get a dictionary that maps file names to component types."""
        return {
            "param_dict.h5": ComponentType.parameter,
            "var_dict.h5": ComponentType.variable,
            "set_dict.h5": ComponentType.sets,
            "dual_dict.h5": ComponentType.dual,
        }

    def get_file_name(self) -> str:
        """Find the file name of a component given its name.

        :param component_name: The name of the component.
        :return: The file name of the component.
        """
        FILE_NAME_MAP = {
            ComponentType.dual: "duals.nc",
            ComponentType.variable: "variables.nc",
            ComponentType.parameter: "parameters.nc",
            ComponentType.sets: "sets.h5",
        }
        return FILE_NAME_MAP[self]

    def get_units_file_name(self) -> str:
        """Find the file name of a component given its name.

        :param component_name: The name of the component.
        :return: The file name of the component.
        """
        if self is ComponentType.sets:
            raise ValueError("Sets do not have units.")
        FILE_NAME_MAP = {
            ComponentType.dual: "duals_units.h5",
            ComponentType.variable: "variables_units.h5",
            ComponentType.parameter: "parameters_units.h5",
        }
        return FILE_NAME_MAP[self]
