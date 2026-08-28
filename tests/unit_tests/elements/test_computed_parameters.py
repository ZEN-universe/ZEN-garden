"""Tests for computed-parameter dependency ordering."""

import pytest

from zen_garden.model.component_types.parameter import GenericParameter


class InputParameter(GenericParameter):
    name = "input"
    indices = ()
    doc = "Ordinary input"
    unit_category = {}


class FirstComputed(GenericParameter):
    name = "first"
    indices = ()
    doc = "First computed parameter"
    unit_category = {}
    dependencies = ["input"]


class SecondComputed(GenericParameter):
    name = "second"
    indices = ()
    doc = "Second computed parameter"
    unit_category = {}
    dependencies = ["first"]


def test_dependent_parameters_are_topologically_ordered():
    parameters = [SecondComputed, InputParameter, FirstComputed]
    assert GenericParameter.construction_order(parameters) == [
        InputParameter,
        FirstComputed,
        SecondComputed,
    ]


def test_all_parameters_are_globally_ordered():
    assert GenericParameter.construction_order(
        [SecondComputed, InputParameter, FirstComputed]
    ) == [
        InputParameter,
        FirstComputed,
        SecondComputed,
    ]


def test_unknown_dependency_is_rejected():
    class UnknownDependency(GenericParameter):
        name = "unknown_dependency"
        indices = ()
        doc = "Invalid computed parameter"
        unit_category = {}
        dependencies = ["not_registered"]

    with pytest.raises(ValueError, match="unknown dependencies"):
        GenericParameter.construction_order([UnknownDependency])


def test_computed_parameter_cycle_is_rejected():
    class ComputedA(GenericParameter):
        name = "computed_a"
        indices = ()
        doc = "Computed A"
        unit_category = {}
        dependencies = ["computed_b"]

    class ComputedB(GenericParameter):
        name = "computed_b"
        indices = ()
        doc = "Computed B"
        unit_category = {}
        dependencies = ["computed_a"]

    with pytest.raises(ValueError, match="Cyclic parameter dependencies"):
        GenericParameter.construction_order([ComputedA, ComputedB])
