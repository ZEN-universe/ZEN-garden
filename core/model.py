# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                               Labratory of Risk and Reliability Engineering
#                                         ETH Zurich, September 2021

# ======================================================================================================================
#                                    MES: VALUE CHAIN DEFINITION
# ======================================================================================================================
import logging
import pyomo.environ as pe
from pyomo.opt import SolverStatus, TerminationCondition
from objects.carrier import Carrier
from objects.technology import Technology

class Model:
    """
    Definition of the value chain
    """

    def __init__(self, analysis, system):
        """
        create Pyomo Abstract Model
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        """

        self.analysis = analysis  # analysis structure
        self.solver = analysis.solver  # solver structure

        self.model = pe.AbstractModel()
        self.addSets()
        self.addCarriers()
        self.addTechnologies()


    def addCarriers(self):
        """
        This method sets up the parameters, variables and constraints of the carriers of the optimization problem.
        """

        carrier = Carrier(self.model)
        carrier.addCarrierParams()
        carrier.addCarrierVars()
        carrier.addCarrierConstr()

    def addSets(self):
        """
        This method sets up the sets of the optimization problem.
        Some sets are initialized with default values, if the value is not specified by the input data
        """

        ## TECHNOLOGIES
        self.model.setTechnologies = pe.Set()  # set of all technologies
        self.model.setProduction = pe.Set()  # subset containing the production technologies
        self.model.setTransport = pe.Set()  # subset containing transport technologies
        self.model.setStorage = pe.Set()  # subset containing the storage technologies

        ## CARRIERS
        self.model.setCarriers = pe.Set()  # set of carriers
        self.model.setAliasCarriers = pe.Set()  # auxiliary set of carriers

        ## NETWORK
        self.model.setNodes = pe.Set()  # set of nodes
        self.model.setAliasNodes = pe.Set()  # auxiliary set of nodes

        ## TIME STEPS
        self.model.setTimeSteps = pe.Set()  # set of time steps

    def addTechnologies(self):
        """
        This method sets up the parameters, variables and constraints of the technologies of the optimization problem.
        """

        technology = Technology(self.model)
        technology.addTechnologyParams()
        technology.addTechnologyVars()
        technology.addTechnologyConstr()

    def addTechnologies(self, data):
        """
        Add  a technology object to value chain
        :param data:
        :return:
        """

        for k in range(len(self.setTechnologies)):
            technology = self.setTechnologies[k]
            technology_type = 'renewable' #TODO somehow get type of technology -- maybe from data?
            if technology_type == 'renewable':
                mes.technologies[k] = technologyR(self, data, self.setTechnologies[k])
            elif technology_type == 'conventional':
                mes.technologies[k] = technologyC(self, data, self.setTechnologies[k])
            elif technology_type == 'co-generation':
                mes.technologies[k] = technologyCoGen(self, data, self.setTechnologies[k])
            elif technology_type == 'storage':
                mes.technologies[k] = technologyS(self, data, self.setTechnologies[k])

    def addConstraints(self):
        """
        :return:
        """

        #TODO figure out a way to formulate constraints using pyomo?

    def addObjecctive(self):
        """
        :return:
        """
        ## OBJECTIVE FUNCTIONS
        self.objective = []  # objective function of the optimization problem
        self.costTotal = 0  # total cost of the system                             [CHF/y]
        self.costInstallation = 0  # total installation cost of the system (annual value) [CHF/y]
        self.costOperation = 0  # total operation cost of the system                   [CHF/y]
        self.emissionsTotal = 0  # total CO2 emissions of the system                    [gCO2/y]
        self.expEnergyNotSupplied = 0  # expected energy not supplied

        # TODO figure out a way to formulate objective function using pyomo?

    def solve(self):
        """
        :return:
        """

        options = solveroptions(self)

        # TODO specify which objective function should be used depending on the settings

    def save(self, diagnostic, sepc):
        """
        :param diagnostic:
        :param sepc:
        :return:
        """

        #TODO save results

