"""Tests for computed-parameter dependency ordering."""

import pytest

from zen_garden.topology.generic_parameter import (
    GenericComputedParameters,
    GenericParameter,
)


class InputParameter(GenericParameter):
    name = "input"
    indices = ()
    doc = "Ordinary input"
    unit_category = {}


class FirstComputed(GenericComputedParameters):
    name = "first"
    indices = ()
    doc = "First computed parameter"
    unit_category = {}
    dependencies = ["input"]


class SecondComputed(GenericComputedParameters):
    name = "second"
    indices = ()
    doc = "Second computed parameter"
    unit_category = {}
    dependencies = ["first"]


def test_computed_parameters_are_topologically_ordered():
    parameters = [SecondComputed, InputParameter, FirstComputed]
    ordered_computed = [
        parameter
        for parameter in GenericParameter.construction_order(parameters)
        if issubclass(parameter, GenericComputedParameters)
    ]
    assert ordered_computed == [
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


def test_computed_parameter_requires_explicit_dependencies():
    with pytest.raises(TypeError, match="must define 'dependencies'"):

        class MissingDependencies(GenericComputedParameters):
            name = "missing_dependencies"
            indices = ()
            doc = "Invalid computed parameter"
            unit_category = {}


def test_unknown_dependency_is_rejected():
    class UnknownDependency(GenericComputedParameters):
        name = "unknown_dependency"
        indices = ()
        doc = "Invalid computed parameter"
        unit_category = {}
        dependencies = ["not_registered"]

    with pytest.raises(ValueError, match="unknown dependencies"):
        GenericParameter.construction_order([UnknownDependency])


def test_computed_parameter_cycle_is_rejected():
    class ComputedA(GenericComputedParameters):
        name = "computed_a"
        indices = ()
        doc = "Computed A"
        unit_category = {}
        dependencies = ["computed_b"]

    class ComputedB(GenericComputedParameters):
        name = "computed_b"
        indices = ()
        doc = "Computed B"
        unit_category = {}
        dependencies = ["computed_a"]

    with pytest.raises(ValueError, match="Cyclic parameter dependencies"):
        GenericParameter.construction_order([ComputedA, ComputedB])
