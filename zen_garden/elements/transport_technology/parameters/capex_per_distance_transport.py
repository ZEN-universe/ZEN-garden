from zen_garden.elements.transport_technology.parameters.capex_specific_transport import (  # noqa: E501
    CapexSpecificTransport,
)
from zen_garden.model.component_types.parameter import (
    GenericParameter,
    ParameterInputData,
)


class CapexPerDistanceTransport(GenericParameter):
    """Capex per distance."""

    name = "capex_per_distance_transport"
    symbol = r"\kappa^{\mathrm{cap,dist}}_{h,e,y}"
    indices = ("set_transport_technologies", "set_edges", "set_years")
    doc = "Capex per distance for transport technologies."
    unit_category = {"money": 1, "distance": -1}
    dependencies = (CapexSpecificTransport,)

    @classmethod
    def get_input_data(cls, element):
        """Return the explicit distance CAPEX or zero when it is folded in."""
        indices = ["set_edges", "set_years"]
        if element.config.system.double_capex_transport:
            value = element.element_data_loader.extract_input_data(
                cls.name, indices, cls.unit_category
            )
        else:
            value = element.capex_specific_transport * 0.0
        return ParameterInputData(value, element.units.get(cls.name))
