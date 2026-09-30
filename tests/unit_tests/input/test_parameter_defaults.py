import pytest

from zen_garden.input import element_data_loader
from zen_garden.input.element_data_loader import ElementDataLoader


def test_missing_attribute_uses_parameter_default():
    attributes = {"existing_parameter": {"default_value": 2, "unit": "MW"}}
    loader = object.__new__(ElementDataLoader)

    with pytest.warns(DeprecationWarning, match="Automatic assign"):
        value, unit = loader._extract_attribute_value(
            "new_parameter",
            attributes,
            default_value=0,
            default_unit="existing_parameter",
        )

    assert value == 0
    assert unit == "MW"


def test_parameter_change_log_only_renames_attributes(monkeypatch):
    attributes = {"old_parameter": {"default_value": 2, "unit": "MW"}}
    loader = object.__new__(ElementDataLoader)
    monkeypatch.setattr(
        element_data_loader,
        "PARAMETER_CHANGE_LOG",
        {"new_parameter": "old_parameter"},
    )

    with pytest.warns(DeprecationWarning, match="is now called"):
        value, unit = loader._extract_attribute_value("new_parameter", attributes)

    assert value == 2
    assert unit == "MW"
    assert "old_parameter" not in attributes
