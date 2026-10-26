"""Tests for computed-parameter dependency ordering."""

import pytest

from zen_garden.model.component_types.parameter import GenericParameter


class InputParameter(GenericParameter):
    name = "input"
    symbol = "i"
    indices = ()
    doc = "Ordinary input"
    unit_category = {}


class FirstComputed(GenericParameter):
    name = "first"
    symbol = "f"
    indices = ()
    doc = "First computed parameter"
    unit_category = {}
    dependencies = (InputParameter,)


class SecondComputed(GenericParameter):
    name = "second"
    symbol = "s"
    indices = ()
    doc = "Second computed parameter"
    unit_category = {}
    dependencies = (FirstComputed,)


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


def test_ignore_missing_treats_absent_dependency_as_satisfied():
    # FirstComputed depends on "input", which is not in the list. With
    # ignore_missing the sort still succeeds and orders the present parameters.
    assert GenericParameter.construction_order(
        [SecondComputed, FirstComputed], ignore_missing=True
    ) == [FirstComputed, SecondComputed]


def test_unknown_dependency_is_rejected():
    class NotRegistered(GenericParameter):
        name = "not_registered"
        symbol = "n"
        indices = ()
        doc = "Missing dependency"
        unit_category = {}

    class UnknownDependency(GenericParameter):
        name = "unknown_dependency"
        symbol = "u"
        indices = ()
        doc = "Invalid computed parameter"
        unit_category = {}
        dependencies = (NotRegistered,)

    with pytest.raises(ValueError, match="unknown dependencies"):
        GenericParameter.construction_order([UnknownDependency])


def test_computed_parameter_cycle_is_rejected():
    class ComputedA(GenericParameter):
        name = "computed_a"
        symbol = "a"
        indices = ()
        doc = "Computed A"
        unit_category = {}

    class ComputedB(GenericParameter):
        name = "computed_b"
        symbol = "b"
        indices = ()
        doc = "Computed B"
        unit_category = {}

    ComputedA.dependencies = (ComputedB,)
    ComputedB.dependencies = (ComputedA,)

    with pytest.raises(ValueError, match="Cyclic parameter dependencies"):
        GenericParameter.construction_order([ComputedA, ComputedB])
