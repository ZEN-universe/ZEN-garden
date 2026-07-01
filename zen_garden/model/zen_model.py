from typing import TYPE_CHECKING

from linopy import Model as LinopyModel

from zen_garden.model.components.constraint import Constraint
from zen_garden.model.components.index_set import IndexSet
from zen_garden.model.components.parameter import Parameter
from zen_garden.model.components.variable import Variable
from zen_garden.model.config import Config
from zen_garden.model.context import Context

if TYPE_CHECKING:
    from zen_garden.model.energy_system import EnergySystem
    from zen_garden.preprocess.unit_handling import UnitHandling


class ZenModel:
    lp_model: LinopyModel
    sets: IndexSet
    variables: Variable
    parameters: Parameter
    constraints: Constraint

    def __init__(
        self,
        config: Config,
        context: Context,
        energy_system: "EnergySystem",
        unit_handling: "UnitHandling",
    ) -> None:
        self.config = config
        self.context = context
        self.energy_system = energy_system
        self.unit_handling = unit_handling

        self.lp_model = LinopyModel(solver_dir=self.config.solver.solver_dir)
        self.sets = IndexSet()
        self.variables = Variable(self)
        self.parameters = Parameter(self)
        self.constraints = Constraint(self)
