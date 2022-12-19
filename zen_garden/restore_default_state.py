from .model.objects.carrier.carrier import Carrier
from .model.objects.carrier.conditioning_carrier import ConditioningCarrier

from .model.objects.element import Element
from .model.objects.energy_system import EnergySystem
from .model.objects.component import Parameter

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
    Carrier.list_of_elements = []
    
    # Conditional Carrier
    #####################
    
    # set label
    ConditioningCarrier.label = "setConditioningCarriers"
    # empty list of elements
    ConditioningCarrier.list_of_elements = []
    
    # Element
    #########

    # set label
    Element.label = "set_elements"
    # empty list of elements
    Element.list_of_elements = []

    # EnergySystem
    ##############

    # energy_system
    EnergySystem.energy_system = None
    # pe.ConcreteModel
    EnergySystem.pyomo_model = None
    # analysis
    EnergySystem.analysis = None
    # system
    EnergySystem.system = None
    # paths
    EnergySystem.paths = None
    # solver
    EnergySystem.solver = None
    # unit handling instance
    EnergySystem.unit_handling = None
    # empty list of indexing sets
    EnergySystem.indexing_sets = []
    # empty dict of technologies of carrier
    EnergySystem.dict_technology_of_carrier = {}
    # empty dict of sequence of time steps operation
    EnergySystem.dict_sequence_time_steps_operation = {}
    # empty dict of sequence of time steps yearly
    EnergySystem.dict_sequence_time_steps_yearly = {}
    # empty dict of conversion from energy time steps to power time steps for storage technologies
    EnergySystem.dict_time_steps_energy2power = {}
    # empty dict of conversion from operational time steps to invest time steps for technologies
    EnergySystem.dict_time_steps_operation2invest = {}
    # empty dict of matching the last time step of the year in the storage domain to the first
    EnergySystem.dict_time_steps_storage_level_startend_year = {}
    # empty dict of element classes
    # EnergySystem.dict_element_classes = {}
    # empty list of class names
    EnergySystem.element_list = {}

    # Parameter
    ###########

    # initialize parameter object
    Parameter.parameterObject = None

    # ConditioningTechnology
    ########################

    # set label
    ConditioningTechnology.label = "setConditioningTechnologies"
    # empty list of elements
    ConditioningTechnology.list_of_elements = []

    # ConversionTechnology
    ######################

    # set label
    ConversionTechnology.label           = "setConversionTechnologies"
    ConversionTechnology.location_type    = "setNodes"
    # empty list of elements
    ConversionTechnology.list_of_elements = []

    # StorageTechnology
    ###################

    # set label
    StorageTechnology.label           = "setStorageTechnologies"
    StorageTechnology.location_type    = "setNodes"
    # empty list of elements
    StorageTechnology.list_of_elements = []

    # Technology
    ############

    # set label
    Technology.label           = "setTechnologies"
    Technology.location_type    = None
    # empty list of elements
    Technology.list_of_elements = []

    # TransportTechnology
    #####################

    # set label
    TransportTechnology.label           = "setTransportTechnologies"
    TransportTechnology.location_type    = "setEdges"
    # empty list of elements
    TransportTechnology.list_of_elements = []
    # dict of reversed edges
    TransportTechnology.dictReversedEdges = {}

    # OptimizationSetup
    ###################

    OptimizationSetup.base_scenario      = ""
    OptimizationSetup.base_configuration = {}

    # TimeSeriesAggregation
    #######################

    TimeSeriesAggregation.time_series_aggregation = None
