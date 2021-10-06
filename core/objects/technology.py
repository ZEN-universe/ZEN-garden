# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                               Labratory of Risk and Reliability Engineering
#                                         ETH Zurich, September 2021

# ======================================================================================================================
#                                    DEFINITION OF THE GENERIC TECHNOLOGY OBJECT
# ======================================================================================================================

import logging
import pyomo.environ as pe

class Technology:
    """This class defines a generic technology"""

    def __init__(self, model):
        """init generic technology object"""
        self.model = model
        self.constraint_names = list()

    def addTechnologyParams(self):
        """add technology params"""

        ## INPUT DATA
        self.model.typeIn = pe.param(
            self.model.setTechnologies,
            self.model.setCarriers,
            default=0,
            doc='Binary parameter which specifies the input carrier of the technology')
        self.model.typeOut = pe.param(
            self.model.setTechnologies,
            self.model.setCarriers,
            default=0,
            doc='Binary parameter which specifies the output carrier of the technology')
        self.model.sizeMin = pe.param(
            self.model.setTechnologies,
            self.model.setCarriers,
            default=0,
            doc='Parameter which specifies the minimum technology size that can be installed')
        self.model.sizeMax = pe.param(
            self.model.setTechnologies,
            self.model.setCarriers,
            default=0,
            doc='Parameter which specifies the maximum technology size that can be installed')
        self.model.alphaL = pe.param(
            self.model.setCarriersCarrier,
            self.model.setAliasCarriers,
            self.model.setTechnologies,
            doc='Parameter which specifies the linear conversion efficiency of a technology')


        self.efficiency = dict()                            # efficiency data (used in other modules such as eventTrees)
        self.efficiency['alphaL'] = mytech['alphaL']
        self.efficiency['alpha'] = mytech['alpha']
        self.efficiency['beta'] = mytech['beta']
        self.efficiency['eta0'] = mytech['eta0']
        self.efficiency['eta1'] = mytech['eta1']
        self.efficiency['eta2'] = mytech['eta2']
        self.efficiency['eta'] = mytech['eta']
        self.efficiency['lambda'] = mytech['lambda']

        ## DECISION VARIABLES
        #TODO figure out a way to add the decision variables with pyomo
        timeHorizon = self.analysis['T']                    # length of the time horizon
        self.select = 0                                     # technology selection
        self.size = 0                                       # technology size         [kW]
        self.input = 0


    def addTechnologyVariables(self):
        """add technology variables"""


    def addTechnologyConstraints(self):
        """add technology constraints"""

        # max. carrier import from the grid
        self.model.constraint_max_carrier_import = pe.Constraint(
            self.model.setCarriers,
            self.model.setNodes,
            self.model.setTimeSteps,
            rule=constraint_max_carrier_import_rule
        )

        self.constraint_names.append('carrier_constraints')