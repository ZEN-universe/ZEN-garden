from .model.objects.carrier.carrier import Carrier
from .model.objects.carrier.conditioning_carrier import ConditioningCarrier

from .model.objects.element import Element
from .model.objects.energy_system import EnergySystem
from .model.objects.parameter import Parameter

from .model.objects.technology.conditioning_technology import ConditioningTechnology
from .model.objects.technology.conversion_technology import ConversionTechnology
from .model.objects.technology.storage_technology import StorageTechnology
from .model.objects.technology.technology import Technology
from .model.objects.technology.transport_technology import TransportTechnology

from .model.optimization_setup import OptimizationSetup

from .postprocess.results import Postprocess

from .preprocess.functions.time_series_aggregation import TimeSeriesAggregation

def restore_default_state():
    """
    Restores the default state of the packages, i.e. sets all Class attributes to their initial values
    """
    
    # Carrier
    #########
    
    # set label
    Carrier.label = "setCarriers"
    # empty list of elements
    Carrier.listOfElements = []
    
    # Conditional Carrier
    #####################
    
    # set label
    ConditioningCarrier.label = "setConditioningCarriers"
    # empty list of elements
    ConditioningCarrier.listOfElements = []
    
    # Element
    #########

    # set label
    Element.label = "setElements"
    # empty list of elements
    Element.listOfElements = []

    # EnergySystem
    ##############

    # energySystem
    EnergySystem.energySystem = None
    # pe.ConcreteModel
    EnergySystem.concreteModel = None
    # analysis
    EnergySystem.analysis = None
    # system
    EnergySystem.system = None
    # paths
    EnergySystem.paths = None
    # solver
    EnergySystem.solver = None
    # unit handling instance
    EnergySystem.unitHandling = None
    # empty list of indexing sets
    EnergySystem.indexingSets = []
    # empty dict of technologies of carrier
    EnergySystem.dictTechnologyOfCarrier = {}
    # empty dict of sequence of time steps operation
    EnergySystem.dictSequenceTimeStepsOperation = {}
    # empty dict of sequence of time steps invest
    EnergySystem.dictSequenceTimeStepsInvest = {}
    # empty dict of sequence of time steps yearly
    EnergySystem.dictSequenceTimeStepsYearly = {}
    # empty dict of conversion from energy time steps to power time steps for storage technologies
    EnergySystem.dictTimeStepsEnergy2Power = {}
    # empty dict of conversion from operational time steps to invest time steps for technologies
    EnergySystem.dictTimeStepsOperation2Invest = {}
    # empty dict of matching the last time step of the year in the storage domain to the first
    EnergySystem.dictTimeStepsStorageLevelStartEndYear = {}
    # empty dict of element classes
    # EnergySystem.dictElementClasses = {}
    # empty list of class names
    EnergySystem.elementList = {}

    # Parameter
    ###########

    # initialize parameter object
    Parameter.parameterObject = None

    # ConditioningTechnology
    ########################

    # set label
    ConditioningTechnology.label = "setConditioningTechnologies"
    # empty list of elements
    ConditioningTechnology.listOfElements = []

    # ConversionTechnology
    ######################

    # set label
    ConversionTechnology.label           = "setConversionTechnologies"
    ConversionTechnology.locationType    = "setNodes"
    # empty list of elements
    ConversionTechnology.listOfElements = []

    # StorageTechnology
    ###################

    # set label
    StorageTechnology.label           = "setStorageTechnologies"
    StorageTechnology.locationType    = "setNodes"
    # empty list of elements
    StorageTechnology.listOfElements = []

    # Technology
    ############

    # set label
    Technology.label           = "setTechnologies"
    Technology.locationType    = None
    # empty list of elements
    Technology.listOfElements = []

    # TransportTechnology
    #####################

    # set label
    TransportTechnology.label           = "setTransportTechnologies"
    TransportTechnology.locationType    = "setEdges"
    # empty list of elements
    TransportTechnology.listOfElements = []
    # dict of reversed edges
    TransportTechnology.dictReversedEdges = {}

    # OptimizationSetup
    ###################

    OptimizationSetup.baseScenario      = ""
    OptimizationSetup.baseConfiguration = {}

    # Postprocess
    #############

    Postprocess.system    = dict()
    Postprocess.varDict   = dict()
    Postprocess.varDf     = dict()
    Postprocess.paramDict = dict()
    Postprocess.paramDf   = dict()
    Postprocess.modelName = str()

    # TimeSeriesAggregation
    #######################

    TimeSeriesAggregation.timeSeriesAggregation = None
