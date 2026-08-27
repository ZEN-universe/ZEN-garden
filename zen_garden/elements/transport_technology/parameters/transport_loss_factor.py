import numpy as np

from zen_garden.topology.generic_parameter import GenericParameter


class TransportLossFactor(GenericParameter):
    """Carrier losses due to transport."""

    name = "transport_loss_factor"
    indices = ("set_transport_technologies", "set_edges")
    doc = "Carrier losses due to transport"
    unit_category = {}
    dependencies = ["distance"]

    @classmethod
    def store_input_data(cls, element):
        """Calculate transport loss from its linear or exponential input."""
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
            input_name, index_sets=[], unit_category={"distance": -1}
        )[0]
        if has_linear:
            value = factor * element.distance
        else:
            value = 1 - np.exp(-factor * element.distance)
            element.config.system.set_transport_technologies_loss_exponential.append(
                element.name
            )
        cls._store_value(element, cls.name, value)
