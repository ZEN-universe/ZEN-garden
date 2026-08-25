"""Generic, metadata-driven loading of element parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from zen_garden.topology.generic_parameter import GenericParameter

# The loader deliberately supports EnergySystem as well as Element instances.
Element = Any


class ParameterInputLoader:
    """Load physical inputs described by ``GenericParameter`` subclasses."""

    def load_into(self, parameter: type[GenericParameter], element: Element) -> None:
        """Load and store a parameter using its declared input strategy."""
        strategy = getattr(self, f"_load_{parameter.input_loader}", None)
        if strategy is None:
            raise ValueError(
                f"Unknown input loader {parameter.input_loader!r} for "
                f"parameter {parameter.name!r}"
            )

        values = strategy(parameter, element)
        for name, value in values.items():
            if parameter.time_series:
                element.raw_time_series[name] = value
            else:
                setattr(element, name, value)

    def _load_standard(
        self, parameter: type[GenericParameter], element: Element
    ) -> dict[str, Any]:
        name = parameter.input_name or parameter.name
        indices = self._input_indices(parameter, element)
        value = element.data_input.extract_input_data(
            name,
            index_sets=indices,
            unit_category=parameter.unit_category,
        )
        values = {parameter.name: value}

        if parameter.capacity_types and self._has_energy_capacity(element):
            energy_name = f"{name}_energy"
            energy_units = dict(parameter.unit_category)
            energy_units.pop("time", None)
            values[f"{parameter.name}_energy"] = element.data_input.extract_input_data(
                energy_name,
                index_sets=indices,
                unit_category=energy_units,
            )
        return values

    def _load_depreciation_time(
        self, parameter: type[GenericParameter], element: Element
    ) -> dict[str, Any]:
        if parameter.name in element.data_input.attribute_dict:
            value = element.data_input.extract_input_data(
                parameter.name, index_sets=[], unit_category={}
            )
            value[0] = max(
                element.config.system.interval_between_years,
                value[0],
            )
        else:
            value = element.lifetime.copy()
        return {parameter.name: value}

    def _load_existing_lifetime(
        self, parameter: type[GenericParameter], element: Element
    ) -> dict[str, Any]:
        indices = self._input_indices(parameter, element)
        return {
            parameter.name: element.data_input.extract_lifetime_existing(
                "capacity_existing", index_sets=indices
            )
        }

    def _load_dependent_carrier(
        self, parameter: type[GenericParameter], element: Element
    ) -> dict[str, Any]:
        dependent_carriers = list(
            set(element.input_carrier + element.output_carrier).difference(
                element.reference_carrier
            )
        )
        if not dependent_carriers:
            return {parameter.name: None}

        indices = self._input_indices(parameter, element)
        values = {
            carrier: element.data_input.extract_input_data(
                parameter.input_name or parameter.name,
                index_sets=indices,
                unit_category=parameter.unit_category,
                subelement=carrier,
            )
            for carrier in dependent_carriers
        }
        combined = pd.DataFrame.from_dict(values)
        combined.columns.name = "carrier"
        combined = combined.stack()
        levels = [combined.index.names[-1], *combined.index.names[:-1]]
        combined = combined.reorder_levels(levels)
        return {parameter.name: combined}

    def _load_transport_loss(
        self, parameter: type[GenericParameter], element: Element
    ) -> dict[str, Any]:
        attributes = element.data_input.attribute_dict
        has_linear = "transport_loss_factor_linear" in attributes
        has_exponential = "transport_loss_factor_exponential" in attributes
        if has_linear and has_exponential:
            raise AttributeError("Only one transport loss factor can be specified.")
        if not has_linear and not has_exponential:
            raise AttributeError(
                f"The transport technology {element.name} has neither of the "
                "attributes transport_loss_factor_linear nor "
                "transport_loss_factor_exponential."
            )

        input_name = (
            "transport_loss_factor_linear"
            if has_linear
            else "transport_loss_factor_exponential"
        )
        factor = element.data_input.extract_input_data(
            input_name,
            index_sets=[],
            unit_category={"distance": -1},
        )[0]
        if has_linear:
            value = factor * element.distance
        else:
            value = 1 - np.exp(-factor * element.distance)
            element.config.system.set_transport_technologies_loss_exponential.append(
                element.name
            )
        return {parameter.name: value}

    def _load_transport_capex(
        self, parameter: type[GenericParameter], element: Element
    ) -> dict[str, Any]:
        attributes = element.data_input.attribute_dict
        indices = ["set_edges", "set_years"]
        specific_units = {"money": 1, "energy_quantity": -1, "time": 1}
        distance_units = {"money": 1, "distance": -1}

        if element.config.system.double_capex_transport:
            specific = element.data_input.extract_input_data(
                "capex_specific_transport", indices, specific_units
            )
            per_distance = element.data_input.extract_input_data(
                "capex_per_distance_transport", indices, distance_units
            )
        elif "capex_per_distance_transport" in attributes:
            per_distance_input = element.data_input.extract_input_data(
                "capex_per_distance_transport",
                indices,
                {
                    "money": 1,
                    "distance": -1,
                    "energy_quantity": -1,
                    "time": 1,
                },
            )
            specific = per_distance_input * element.distance
            per_distance = specific * 0.0
        elif "capex_specific_transport" in attributes:
            specific = element.data_input.extract_input_data(
                "capex_specific_transport", indices, specific_units
            )
            per_distance = specific * 0.0
        else:
            raise AttributeError(
                f"The transport technology {element.name} has neither "
                "capex_per_distance_transport nor capex_specific_transport attribute."
            )

        return {
            parameter.name: specific,
            "capex_per_distance_transport": per_distance,
        }

    def _load_fixed_opex(
        self, parameter: type[GenericParameter], element: Element
    ) -> dict[str, Any]:
        if getattr(element, "location_type", None) != "set_edges":
            return self._load_standard(parameter, element)

        attributes = element.data_input.attribute_dict
        indices = ["set_edges", "set_years"]
        if "opex_specific_fixed_per_distance" in attributes:
            per_distance = element.data_input.extract_input_data(
                "opex_specific_fixed_per_distance",
                indices,
                {
                    "money": 1,
                    "distance": -1,
                    "energy_quantity": -1,
                    "time": 1,
                },
            )
            value = per_distance * element.distance
        elif parameter.name in attributes:
            value = element.data_input.extract_input_data(
                parameter.name,
                indices,
                parameter.unit_category,
            )
        else:
            raise AttributeError(
                f"The transport technology {element.name} has neither "
                "opex_specific_fixed_per_distance nor opex_specific_fixed attribute."
            )
        return {parameter.name: value}

    def _load_storage_capex(
        self, parameter: type[GenericParameter], element: Element
    ) -> dict[str, Any]:
        indices = ["set_nodes", "set_years"]
        return {
            parameter.name: element.data_input.extract_input_data(
                parameter.name,
                indices,
                {"money": 1, "energy_quantity": -1, "time": -1},
            ),
            f"{parameter.name}_energy": element.data_input.extract_input_data(
                f"{parameter.name}_energy",
                indices,
                {"money": 1, "energy_quantity": -1},
            ),
        }

    def _load_skip(
        self, parameter: type[GenericParameter], element: Element
    ) -> dict[str, Any]:
        return {}

    def _load_carbon_intensity(
        self, parameter: type[GenericParameter], element: Element
    ) -> dict[str, Any]:
        value = self._load_standard(parameter, element)[parameter.name]
        unit = element.units[parameter.name]["unit_in_base_units"].units
        if getattr(element, "location_type", None) == "set_edges" and (
            "/ kilometer" in str(unit)
        ):
            value = element.data_input.extract_input_data(
                parameter.name,
                index_sets=["set_edges"],
                unit_category={
                    "emissions": 1,
                    "energy_quantity": -1,
                    "distance": -1,
                },
            )
            value *= element.distance
        return {parameter.name: value}

    @staticmethod
    def _has_energy_capacity(element: Element) -> bool:
        return element.name in element.config.system.set_storage_technologies

    @staticmethod
    def _input_indices(
        parameter: type[GenericParameter], element: Element
    ) -> list[str]:
        if parameter.input_indices is not None:
            indices = list(parameter.input_indices)
        else:
            owner_labels = {
                base.__dict__["label"]
                for base in type(element).mro()
                if "label" in base.__dict__
            }
            indices = [
                index
                for index in parameter.indices
                if index not in owner_labels and index != "set_capacity_types"
            ]

        location_type = getattr(element, "location_type", None)
        if "set_location" in indices and location_type is None:
            raise ValueError(f"Element {element.name!r} has no location type")
        return [
            str(location_type) if index == "set_location" else index
            for index in indices
        ]
