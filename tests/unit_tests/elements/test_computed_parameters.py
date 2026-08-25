"""Tests for computed-parameter dependency ordering."""

import pytest

from zen_garden.elements.element import Element
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


class OrderedElement(Element):
    own_parameters = [SecondComputed, InputParameter, FirstComputed]


def test_computed_parameters_are_topologically_ordered():
    assert OrderedElement._ordered_computed_parameters() == [
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

    class InvalidElement(Element):
        own_parameters = [UnknownDependency]

    with pytest.raises(ValueError, match="unknown dependencies"):
        InvalidElement._ordered_computed_parameters()


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

    class CyclicElement(Element):
        own_parameters = [ComputedA, ComputedB]

    with pytest.raises(ValueError, match="Cyclic computed-parameter dependencies"):
        CyclicElement._ordered_computed_parameters()
