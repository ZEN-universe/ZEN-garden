"""Tests for simple and indexed ZEN set representations."""

from zen_garden.model.registries.set import IndexedSet, SimpleSet


def test_simple_zen_set_exposes_ordered_members_as_coordinate_values():
    zen_set = SimpleSet(["CH", "DE", "CH"], name="set_nodes")

    assert list(zen_set) == ["CH", "DE"]
    assert list(zen_set.coordinate_values) == ["CH", "DE"]
    assert zen_set[0] == "CH"
    assert not zen_set.is_indexed()


def test_indexed_zen_set_maps_keys_to_simple_sets_and_flattens_coordinates():
    zen_set = IndexedSet(
        {
            "boiler": ["gas", "heat"],
            "heat_pump": ["electricity", "heat"],
        },
        name="set_input_carriers",
        index_set="set_conversion_technologies",
    )

    assert list(zen_set) == ["boiler", "heat_pump"]
    assert isinstance(zen_set["boiler"], SimpleSet)
    assert list(zen_set["boiler"]) == ["gas", "heat"]
    assert list(zen_set.coordinate_values) == ["gas", "heat", "electricity"]
    assert zen_set.get_index_name() == "set_conversion_technologies"
