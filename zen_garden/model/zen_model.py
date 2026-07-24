"""ZEN-model to combine sets, paramters, variables and constraints
from all elements into a single model."""

from typing import TYPE_CHECKING

from linopy import Model as LinopyModel

from zen_garden.model.components.constraint import Constraint
from zen_garden.model.components.index_set import IndexSet
from zen_garden.model.components.parameter import Parameter
from zen_garden.model.components.variable import Variable
from zen_garden.model.config import Config

if TYPE_CHECKING:
    from zen_garden.elements.energy_system import EnergySystem
    from zen_garden.preprocess.unit_handling import UnitHandling
    from zen_garden.services.element_registry import ElementRegistry


class ZenModel:
    def __init__(
        self,
        config: Config,
        energy_system: "EnergySystem",
        unit_handling: "UnitHandling",
        element_registry: "ElementRegistry",
    ) -> None:
        self.config = config
        self.energy_system = energy_system
        self.unit_handling = unit_handling
        self.element_registry = element_registry

        self.indexing_sets = [key for key in self.config.system.keys() if "set" in key]

        self.lp_model = LinopyModel(solver_dir=self.config.solver.solver_dir)
        self.sets = IndexSet()
        self.variables = Variable(
            self.unit_handling,
            self.sets,
            self.lp_model,
            self.config,
            self.element_registry,
        )
        self.parameters = Parameter(self.sets)
        self.constraints = Constraint(self.lp_model)

    def add_set(self, *args, **kwargs):
        """Add sets to the model.
        See :meth:`zen_garden.model.components.index_set.IndexSet.add_set`.
        """
        self.sets.add_set(*args, **kwargs)

    def add_variable(self, *args, **kwargs):
        """Add variables to the model.
        See :meth:`zen_garden.model.components.variable.Variable.add_variable`.
        """
        self.variables.add_variable(*args, **kwargs)

    def add_parameter(self, *args, **kwargs):
        """Add parameters to the model.
        See :meth:`zen_garden.model.components.parameter.Parameter.add_parameter`.
        """
        self.parameters.add_parameter(*args, **kwargs)

    def add_constraint(self, *args, **kwargs):
        """Add constraints to the model.
        See :meth:`zen_garden.model.components.constraint.Constraint.add_constraint`.
        """
        self.constraints.add_constraint(*args, **kwargs)
