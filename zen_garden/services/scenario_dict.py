import os
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zen_garden.model.config import Config
    from zen_garden.services.dataset_path_resolver import DatasetPathResolver
    from zen_garden.services.element_registry import ElementRegistry


class ScenarioDict(dict):
    """Dictionary for the scenario analysis that has some convenience functions."""

    _param_dict_keys = {
        "file",
        "part_file",
        "file_op",
        "default",
        "default_op",
        "value",
    }
    _special_elements = ["base_scenario", "sub_folder", "param_map"]
    _setting_elements = ["system", "analysis", "solver"]

    def __init__(
        self,
        init_dict: dict,
        dataset_path_resolver: "DatasetPathResolver",
        config: "Config",
        element_type_classes: dict[str, type],
    ):
        """Initializes the dictionary from a normal dictionary.

        :param init_dict: The dictionary to initialize from
        :param optimization_setup: The optimization setup corresponding to the scenario
        :param paths: The paths to the elements
        """
        # set the attributes and expand the dict
        self.dataset_path_resolver = dataset_path_resolver
        self.config = config
        self.element_type_classes = element_type_classes

        self.dict = self.expand_subsets(init_dict)

        # super init TODO adds both system and "system"  (same for analysis)
        # to the dict - necessary?
        super().__init__(self.dict)

        # finally we update the analysis, system, and solver in the config
        self.update_config()

    def update_config(self):
        """Updates the analysis, system, and solver in the config."""
        config_parts = {
            "analysis": self.config.analysis,
            "system": self.config.system,
            "solver": self.config.solver,
        }
        for key, value in config_parts.items():
            if key in self.dict:
                for sub_key, sub_value in self.dict[key].items():
                    assert sub_key in value.keys(), (
                        f"Trying to update {key} with key {sub_key} and value "
                        f"{sub_value}, but the {key} does not have this key!"
                    )
                    if type(value[sub_key]) is type(sub_value):
                        value[sub_key] = sub_value
                    elif isinstance(
                        sub_value, dict
                    ):  # ToDO check and generalize -> here only for SolverOptions
                        try:
                            for sub_sub_key, sub_sub_value in sub_value.items():
                                value[sub_key][sub_sub_key] = sub_sub_value
                        except Exception as err:
                            raise ValueError(
                                f"Trying to update {key} with key {sub_key} and value "
                                f"{sub_value} of type {type(sub_value)}, "
                                f"but the {key} has already a value of type "
                                f"{type(value[sub_key])}"
                            ) from err
                    else:
                        raise ValueError(
                            f"Trying to update {key} with key {sub_key} and value "
                            f"{sub_value} of type {type(sub_value)}, "
                            f"but the {key} has already a value of type "
                            f"{type(value[sub_key])}"
                        )

    @staticmethod
    def expand_lists(scenarios: dict):
        """Expand parameter lists in the all scenarios and return a new dict.

        Args:
            scenarios (dict): The initial dict of scenarios

        Returns:
            dict: The expanded dict, where all necessary parameters are expanded
            and subpaths are set
        """
        # Important, all for-loops through keys or items in this routine should be
        # sorted!

        expanded_scenarios = dict()
        for scenario_name, scenario_dict in sorted(
            scenarios.items(), key=lambda x: x[0]
        ):
            assert (
                type(scenario_dict) is dict
            ), f"Scenario {scenario_name} is not a dictionary!"
            scenario_dict["base_scenario"] = scenario_name
            scenario_dict["sub_folder"] = ""
            scenario_dict["param_map"] = dict()
            scenario_list = ScenarioDict._expand_scenario(scenario_dict)

            # add the scenarios to the dict
            for scenario in scenario_list:
                if scenario["sub_folder"] == "":
                    name = scenario["base_scenario"]
                else:
                    name = scenario["base_scenario"] + "_" + scenario["sub_folder"]
                expanded_scenarios[name] = scenario

        return expanded_scenarios

    @staticmethod
    def _expand_scenario(scenario: dict, param_map=None, counter=0):
        """Expands a scenario, returns a list of scenarios.

        :param scenario: The scenario to expand
        :param param_map: The parameter map for the scenario
        :param counter: The counter for the scenario
        :return: A list of scenarios
        """
        # get the default
        if param_map is None:
            param_map = dict()

        # list for the expanded scenarios
        expanded_scenarios = []

        # iterate over all elements
        for element, element_dict in sorted(scenario.items(), key=lambda x: x[0]):
            # we do not expand these
            if element in ScenarioDict._special_elements:
                continue
            # check for 'system' analysis' and 'solver' keys and see whether they are
            # dicts and have a list in them, only then do the list expansion,
            # otherwise proceed as always.
            for param, param_dict in sorted(element_dict.items(), key=lambda x: x[0]):
                if element in ScenarioDict._setting_elements:
                    if not isinstance(param_dict, dict):
                        continue
                    elif isinstance(param_dict, dict) and not isinstance(
                        param_dict["value"], list
                    ):
                        scenario[element][param] = param_dict["value"]
                for key in sorted(ScenarioDict._param_dict_keys):
                    if key in param_dict and isinstance(param_dict[key], list):
                        # get the old param dict entry
                        if scenario["sub_folder"] != "":
                            old_param_map_entry = param_map.pop(scenario["sub_folder"])
                        else:
                            old_param_map_entry = dict()

                        # we need to expand this
                        for num, value in enumerate(param_dict[key]):
                            # copy the scenario
                            new_scenario = deepcopy(scenario)

                            # set the new value
                            if element in ScenarioDict._setting_elements:
                                new_scenario[element][param] = value
                            else:
                                new_scenario[element][param][key] = value

                            # create the name
                            if key + "_fmt" in param_dict:
                                if "{}" not in param_dict[key + "_fmt"]:
                                    raise SyntaxError(
                                        "When setting a format for a name, you need to "
                                        "include a placeholder '{}' for its value! No "
                                        f"placeholder found for {key} in {param} in "
                                        f"{element} in {scenario['base_scenario']}"
                                    )
                                name = param_dict[key + "_fmt"].format(value)
                                if element not in ScenarioDict._setting_elements:
                                    del new_scenario[element][param][key + "_fmt"]
                                # don't need to increment the param for next expansion
                                param_up = 0
                            else:
                                name = f"p{counter:02d}_{num:03d}"
                                # we need to increment the param for the next expansion
                                param_up = 1

                            # set the sub_folder
                            if new_scenario["sub_folder"] == "":
                                new_scenario["sub_folder"] = name
                            else:
                                new_scenario["sub_folder"] += "_" + name

                            # update the param_map
                            param_map[new_scenario["sub_folder"]] = deepcopy(
                                old_param_map_entry
                            )
                            if element not in param_map[new_scenario["sub_folder"]]:
                                param_map[new_scenario["sub_folder"]][element] = dict()
                            if (
                                param
                                not in param_map[new_scenario["sub_folder"]][element]
                            ):
                                param_map[new_scenario["sub_folder"]][element][
                                    param
                                ] = dict()
                            if element in ScenarioDict._setting_elements:
                                param_map[new_scenario["sub_folder"]][element][
                                    param
                                ] = value
                            else:
                                param_map[new_scenario["sub_folder"]][element][param][
                                    key
                                ] = value

                            # set the param_map of the scenario
                            new_scenario["param_map"] = param_map

                            # expand this scenario as well
                            expanded_scenarios.extend(
                                ScenarioDict._expand_scenario(
                                    new_scenario, param_map, counter + param_up
                                )
                            )

                        # expansion done
                        return expanded_scenarios

        # nothing was expanded, so we just return the scenario
        expanded_scenarios.append(scenario)

        # return the list
        return expanded_scenarios

    def expand_subsets(self, init_dict):
        """Expands a dictionary, e.g. expands sets etc.

        :param init_dict: The initial dict
        :return: A new dict which can be used for the scenario analysis
        """
        new_dict = init_dict.copy()
        for element_class in reversed(list(self.element_type_classes.values())):
            current_set = element_class.label
            if current_set not in new_dict:
                continue

            for param, param_dict in new_dict[current_set].items():
                # dict for expansion
                base_dict = param_dict

                # get the exlusion list
                if "exclude" in base_dict:
                    exclude_list = base_dict["exclude"]
                    del base_dict["exclude"]
                else:
                    exclude_list = []

                # expand the sets
                elements = self.dataset_path_resolver.elements_of_set(current_set)
                for element in elements:
                    if element != "folder" and element not in exclude_list:
                        # create dicts if necessary
                        if element not in new_dict:
                            new_dict[element] = {}
                        # we only set the param dict if it is not already set
                        if param not in new_dict[element]:
                            new_dict[element][param] = base_dict.copy()
            # delete the old set
            del new_dict[current_set]

        self.validate_dict(new_dict)

        return new_dict

    def validate_dict(self, vali_dict):
        """Validates a dictionary, raises an error if it is not valid.

        :param vali_dict: The dictionary to validate
        """
        for element, element_dict in vali_dict.items():
            if element in self._special_elements or element in self._setting_elements:
                continue

            if not isinstance(element_dict, dict):
                raise ValueError(f"The entry for {element} is not a dictionary!")

            for param, param_dict in element_dict.items():
                if len(diff := (set(param_dict.keys()) - self._param_dict_keys)) > 0:
                    raise ValueError(
                        f"The entry for element {element} and param {param} "
                        f"contains invalid entries: {diff}!"
                    )

    def check_if_all_elements_in_model(self, element_registry: "ElementRegistry"):
        """Checks if all elements in scenario_dict are present in the element_dict.

        This is used to ensure that all elements in the scenario are defined in
        the model.

        :param scenario_dict: Dictionary containing the scenario elements
        :param element_dict: Dictionary containing the element definitions
        """
        ignored_elements = (
            ScenarioDict._setting_elements
            + ScenarioDict._special_elements
            + list(ScenarioDict._param_dict_keys)
            + ["EnergySystem"]
        )
        relevant_elements = set(self.keys()) - set(ignored_elements)
        existing_elements = [e.name for e in element_registry.all_elements()]
        for element in relevant_elements:
            if element not in existing_elements:
                raise KeyError(
                    f"The element '{element}', defined in the scenario file, "
                    "is not defined in the model."
                )

    @staticmethod
    def validate_file_name(fname):
        """Checks if the file name has an extension.

        It is expected to not have an extension.

        :param fname: The file name to validte
        :return: The validated file name
        """
        fname, ext = os.path.splitext(fname)
        if ext != "":
            warnings.warn(
                f"The file name {fname}{ext} has an extension {ext}, removing it.",
                stacklevel=2,
            )
        return fname

    def get_default(self, element, param):
        """Return the name where the default value should be read out.

        Args:
            element: The element name
            param: The parameter of the element

        Returns: If the entry is overwritten by the scenario analysis the entry
            and factor are returned, otherwise the default entry is returned
            with a factor of 1
        """
        # These are the default values
        default_f_name = "attributes"
        default_factor = 1.0

        if element in self.dict and param in (element_dict := self.dict[element]):
            param_dict = element_dict[param]
            default_f_name = param_dict.get("default", default_f_name)
            default_f_name = self.validate_file_name(default_f_name)
            default_factor = param_dict.get("default_op", default_factor)
            self._check_if_numeric_default_factor(
                default_factor,
                element=element,
                param=param,
                default_f_name=default_f_name,
                op_type="default_op",
            )

        return default_f_name, default_factor

    def get_param_file(self, element, param):
        """Return the file name where the parameter values should be read out.

        Args:
            element: The element name
            param: The parameter of the element

        Returns:
            If the entry is overwritten by the scenario analysis the entry and
            factor are returned, otherwise the default entry is returned with
            a factor of 1
        """
        # These are the default values
        default_f_name = param
        default_factor = 1.0

        if element in self.dict and param in (element_dict := self.dict[element]):
            param_dict = element_dict[param]
            default_f_name = param_dict.get("file", default_f_name)
            default_f_name = self.validate_file_name(default_f_name)
            default_factor = param_dict.get("file_op", default_factor)
            self._check_if_numeric_default_factor(
                default_factor,
                element=element,
                param=param,
                default_f_name=default_f_name,
                op_type="file_op",
            )

        return default_f_name, default_factor

    def get_param_part_file(self, element, param):
        """Return the partial file name where the parameter values should be read out.

        Args:
            element: the element name
            param: the parameter of the element for which the partial file name is
                returned

        Returns:
            If the entry is overwritten by the scenario analysis the entry,
            otherwise None.
        """
        if element in self.dict and param in (element_dict := self.dict[element]):
            param_dict = element_dict[param]
            if "part_file" in param_dict:
                part_file = param_dict["part_file"]
                part_file = self.validate_file_name(part_file)
                return part_file
        return None

    def _check_if_numeric_default_factor(
        self, default_factor, element, param, default_f_name, op_type
    ):
        """Check if the default factor is numeric.

        :param default_factor: The default factor to check
        """
        if not isinstance(default_factor, (int, float)):
            raise ValueError(
                f"Default factor {default_factor} of type {type(default_factor)} in "
                f"{op_type} ({element} -> {param} -> {default_f_name}) is not numeric!"
            )
