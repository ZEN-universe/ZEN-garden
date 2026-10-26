"""Tests for canonical ZEN domain and relation representations."""

import pandas as pd
import pytest

from zen_garden.model.registries.set import DomainSet, RelationSet


def test_domain_set_exposes_ordered_canonical_members():
    domain = DomainSet("set_nodes", ["CH", "DE"])

    assert isinstance(domain.members, pd.Index)
    assert domain.members.name == "set_nodes"
    assert list(domain) == ["CH", "DE"]
    assert list(domain.coordinate_values) == ["CH", "DE"]
    assert domain[0] == "CH"


def test_domain_set_rejects_duplicate_members():
    with pytest.raises(ValueError, match="duplicate members"):
        DomainSet("set_nodes", ["CH", "DE", "CH"])


def test_relation_set_normalizes_mapping_and_supports_named_projection():
    relation = RelationSet(
        "relation_technology_input_carrier",
        ("set_conversion_technologies", "set_input_carriers"),
        {
            "boiler": ["gas", "heat"],
            "heat_pump": ["electricity", "heat"],
        },
    )

    assert isinstance(relation.combinations, pd.MultiIndex)
    assert relation.combinations.names == [
        "set_conversion_technologies",
        "set_input_carriers",
    ]
    assert list(
        relation.project("set_input_carriers", set_conversion_technologies="boiler")
    ) == ["gas", "heat"]
    assert list(
        relation.project("set_conversion_technologies", set_input_carriers="heat")
    ) == ["boiler", "heat_pump"]
