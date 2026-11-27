from types import SimpleNamespace

import pytest

from zen_garden.input.scenario_dict import ScenarioDict

# element folders in the dataset; natural_gas_storage has a folder but is not
# selected in the model
DATASET_FOLDERS = {
    "set_technologies": {
        "folder": "set_technologies",
        "natural_gas_boiler": {},
        "natural_gas_pipeline": {},
        "natural_gas_storage": {},
    },
}
MODEL_ELEMENTS = ["natural_gas_boiler", "natural_gas_pipeline"]


class _PathResolver:
    def elements_of_set(self, set_name):
        return DATASET_FOLDERS[set_name].keys()


class _ElementRegistry:
    def all_elements(self):
        return [SimpleNamespace(name=name) for name in MODEL_ELEMENTS]


def _scenario_dict(init_dict):
    config = SimpleNamespace(analysis={}, system={}, solver={})
    model_schema = SimpleNamespace(
        config=config,
        configurable_element_type_classes={
            "Technology": SimpleNamespace(label="set_technologies"),
        },
    )
    return ScenarioDict(init_dict, _PathResolver(), model_schema)


def test_set_expansion_skips_elements_not_in_model():
    scenario_dict = _scenario_dict(
        {"set_technologies": {"max_diffusion_rate": {"default": "attributes_ter"}}}
    )
    assert "natural_gas_storage" in scenario_dict

    scenario_dict.check_if_all_elements_in_model(_ElementRegistry())

    assert "natural_gas_storage" not in scenario_dict
    for element in MODEL_ELEMENTS:
        assert scenario_dict.get_default(element, "max_diffusion_rate") == (
            "attributes_ter",
            1,
        )


def test_explicit_element_not_in_model_raises():
    scenario_dict = _scenario_dict(
        {
            "set_technologies": {"max_diffusion_rate": {"default": "attributes_ter"}},
            "natural_gas_storage": {"lifetime": {"default": "attributes_ter"}},
        }
    )

    with pytest.raises(KeyError, match="natural_gas_storage"):
        scenario_dict.check_if_all_elements_in_model(_ElementRegistry())
