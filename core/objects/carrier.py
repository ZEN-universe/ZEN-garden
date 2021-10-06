# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                               Labratory of Risk and Reliability Engineering
#                                         ETH Zurich, September 2021

# ======================================================================================================================
#                                    DEFINITION OF THE CARRIERS
# ======================================================================================================================
import logging
import pyomo.environ as pe

def Carrier:
    """This class defines the generic carrier"""

    def __init__(self, model):
        """init generic carrier object"""
        self.model = model
        self.constraint_names = list()


    def addCarrierParams(self):
        """add carrier params"""
        logging.info('add parameters of a generic carrier')

        self.model.demand = pe.Param(
            self.model.setCarriers,
            self.model.setNodes,
            self.model.setTimeSteps,
            default = 0,
            within = pe.NonNegativeReals,
            doc = 'Parameter which specifies the node- and time-dependent carrier demand')
        self.model.price = pe.Param(
            self.model.setCarriers,
            self.model.setNodes,
            self.model.setTimeSteps,
            default=0,
            within=pe.NonNegativeReals,
            doc='Parameter which specifies the node- and time-dependent carrier price')
        self.model.Cfootprint = pe.Param(
            self.model.setCarriers,
            default=0,
            within=pe.NonNegativeReals,
            doc='Parameter which specifies the carbon intensity of a carrier')
        self.model.gridIn = pe.Param(
            self.model.setCarriers,
            default=0,
            within=pe.NonNegativeReals,
            doc='Parameter which specifies the maximum energy that can be imported from the grid per unit of time')

    def addCarrierVariables(self):
        """add carrier variables"""
        logging.info('add variables of a generic carrier')

        # node- and time-dependent carrier imported from the grid
        self.model.importCarrier = pe.Var(
            self.model.setCarriers,
            self.model.setNodes,
            self.model.setTimeSteps,
            within=pe.NonNegativeReals)
        # energy involved in conversion within MES (??)
        self.model.conver = pe.Var(
            self.model.setCarriers,
            self.model.setNodes,
            self.model.TimeSteps,
            within=pe.NonNegativeReals)

    def addCarrierVariables(self):
        """add carrier constraints"""
        logging.info('add generic carrier constraints')

        # max. carrier import from the grid
        self.model.constraint_max_carrier_import = pe.Constraint(
            self.model.setCarriers,
            self.model.setNodes,
            self.model.setTimeSteps,
            rule=constraint_max_carrier_import_rule
        )

        self.constraint_names.append('carrier_constraints')


