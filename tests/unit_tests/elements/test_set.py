"""Tests for canonical ZEN domain and relation representations."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from zen_garden.model.component_types.set import GenericDomainSet, GenericRelationSet
from zen_garden.model.construction_service import ModelConstructionService
from zen_garden.model.registries.set import DomainSet, RelationSet
from zen_garden.model.registries.set_registry import SetRegistry

BUILD_ORDER: list[str] = []


class SetLocations(GenericDomainSet):
    name = "set_test_locations"
    symbol = "l"
    doc = "Test locations."

    @classmethod
    def get_data(cls, model_constructor):
        BUILD_ORDER.append(cls.name)
        return ["node_a", "node_b", "edge_a_b"]


class SetNodes(GenericDomainSet):
    name = "set_test_nodes"
    symbol = "n"
    doc = "Test nodes."
    subset_of = SetLocations
    dependencies = (SetLocations,)

    @classmethod
    def get_data(cls, model_constructor):
        BUILD_ORDER.append(cls.name)
        return ["node_a", "node_b"]


class SetSpecialNodes(GenericDomainSet):
    name = "set_test_special_nodes"
    symbol = "n_s"
    doc = "Test special nodes."
    subset_of = SetNodes

    @classmethod
    def get_data(cls, model_constructor):
        return ["node_b"]


class SetCarriers(GenericDomainSet):
    name = "set_test_carriers"
    symbol = "c"
    doc = "Test carriers."

    @classmethod
    def get_data(cls, model_constructor):
        BUILD_ORDER.append(cls.name)
        return ["gas", "power"]


class RelationLocationCarrier(GenericRelationSet):
    name = "relation_test_location_carrier"
    doc = "Allowed test location and carrier pairs."
    dimensions = (SetLocations, SetCarriers)

    @classmethod
    def get_combinations(cls, model_constructor):
        BUILD_ORDER.append(cls.name)
        return [
            ("node_a", "gas"),
            ("node_b", "power"),
            ("edge_a_b", "gas"),
        ]


class RelationNodeCarrier(GenericRelationSet):
    name = "relation_test_node_carrier"
    doc = "Alternative test node and carrier pairs."
    dimensions = (SetNodes, SetCarriers)

    @classmethod
    def get_combinations(cls, model_constructor):
        return [("node_a", "gas")]


class SetInvalidMapping(GenericDomainSet):
    name = "set_test_invalid_mapping"
    symbol = "x"
    doc = "Invalid mapping-backed domain."

    @classmethod
    def get_data(cls, model_constructor):
        return {"node_a": ["gas"]}


class DuplicateDomain(GenericDomainSet):
    name = "duplicate_test_set"
    symbol = "d"
    doc = "Duplicate test domain."

    @classmethod
    def get_data(cls, model_constructor):
        return []


class DuplicateRelation(GenericRelationSet):
    name = "duplicate_test_set"
    doc = "Duplicate test relation."
    dimensions = (SetLocations, SetCarriers)

    @classmethod
    def get_combinations(cls, model_constructor):
        return []


def make_registry() -> SetRegistry:
    """Return an isolated canonical set registry for unit tests."""
    return SetRegistry(MagicMock(), MagicMock())


def make_constructor(registry: SetRegistry, declarations):
    """Return the minimal constructor surface used by set declarations."""
    return SimpleNamespace(
        sets=declarations,
        optimization_model=SimpleNamespace(sets=registry),
    )


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


def test_mixed_own_sets_build_domains_before_relations_in_dependency_order():
    registry = make_registry()
    constructor = make_constructor(
        registry,
        [RelationLocationCarrier, SetNodes, SetCarriers, SetLocations],
    )
    service = object.__new__(ModelConstructionService)
    service.model_schema = MagicMock()
    service.model_schema.config.solver.run_diagnostics = False
    service.optimization_model = constructor.optimization_model
    service._model_constructors = [constructor]
    BUILD_ORDER.clear()

    service._construct_sets()

    assert BUILD_ORDER == [
        SetCarriers.name,
        SetLocations.name,
        SetNodes.name,
        RelationLocationCarrier.name,
    ]
    assert list(registry.sets) == [
        SetCarriers.name,
        SetLocations.name,
        SetNodes.name,
    ]
    assert list(registry.relations) == [RelationLocationCarrier.name]


def test_mixed_own_sets_reject_duplicate_names_across_declaration_types():
    registry = make_registry()
    constructor = make_constructor(registry, [DuplicateDomain, DuplicateRelation])
    service = object.__new__(ModelConstructionService)
    service._model_constructors = [constructor]

    with pytest.raises(ValueError, match="Multiple set specifications"):
        service._set_declarations()


def test_domain_declaration_rejects_mapping_data():
    registry = make_registry()
    constructor = make_constructor(registry, [SetInvalidMapping])

    with pytest.raises(TypeError, match="use a GenericRelationSet"):
        SetInvalidMapping.build(constructor)


def test_subset_aware_resolution_filters_parent_relation_transitively():
    registry = make_registry()
    constructor = make_constructor(registry, [])
    for declaration in (SetLocations, SetNodes, SetSpecialNodes, SetCarriers):
        declaration.build(constructor)
    registry.validate_subsets()
    RelationLocationCarrier.build(constructor)

    resolved = registry.resolve_indices(
        (SetSpecialNodes.name, SetCarriers.name),
        index_relations=(RelationLocationCarrier,),
    )

    assert resolved.combinations.tolist() == [("node_b", "power")]
    assert registry.project_relation(
        RelationLocationCarrier,
        SetCarriers,
        set_test_special_nodes="node_b",
    ).tolist() == ["power"]
    mask = registry.relation_mask(
        RelationLocationCarrier,
        (SetSpecialNodes, SetCarriers),
    )
    assert bool(mask.sel(set_test_special_nodes="node_b", set_test_carriers="power"))


def test_empty_relation_resolves_to_an_empty_structural_index():
    registry = make_registry()
    registry.add_domain(DomainSet(SetLocations.name, ["node_a"]))
    registry.add_domain(DomainSet(SetCarriers.name, ["gas"]))
    registry.add_relation(
        RelationSet(
            RelationLocationCarrier.name,
            (SetLocations.name, SetCarriers.name),
            [],
        )
    )

    resolved = registry.resolve_indices(
        (SetLocations.name, SetCarriers.name),
        index_relations=(RelationLocationCarrier,),
    )

    assert resolved.combinations.empty
    assert not bool(resolved.structural_mask.any())


def test_subset_validation_rejects_missing_parents_members_and_cycles():
    registry = make_registry()
    registry.add_domain(DomainSet("child", ["outside"]))
    registry.subset_parents["child"] = "missing"
    with pytest.raises(KeyError, match="Unknown subset parent"):
        registry.validate_subsets()

    registry.add_domain(DomainSet("parent", ["inside"]))
    registry.subset_parents["child"] = "parent"
    with pytest.raises(ValueError, match="outside parent"):
        registry.validate_subsets()

    registry.sets["child"] = DomainSet("child", ["inside"])
    registry.subset_parents["parent"] = "child"
    with pytest.raises(ValueError, match="Cyclic domain subset"):
        registry.validate_subsets()


def test_auto_resolution_rejects_ambiguous_parent_and_subset_relations():
    registry = make_registry()
    constructor = make_constructor(registry, [])
    for declaration in (SetLocations, SetNodes, SetCarriers):
        declaration.build(constructor)
    registry.validate_subsets()
    RelationLocationCarrier.build(constructor)
    RelationNodeCarrier.build(constructor)

    with pytest.raises(ValueError, match="Ambiguous default index relations"):
        registry.resolve_indices((SetNodes.name, SetCarriers.name))
