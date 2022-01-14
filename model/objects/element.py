"""===========================================================================================================================================================================
Title:          ENERGY-CARBON OPTIMIZATION PLATFORM
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining a standard element. Contains methods to add parameters, variables and constraints to the
                optimization problem. Parent class of the Carrier and Technology classes .The class takes the abstract
                optimization model as an input.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe

class Element:
    # empty list of elements
    listOfElements = []
    # pe.ConcreteModel
    concreteModel = None
    # input dictionary
    pyoDict = None
    # analysis
    analysis = None
    # system
    system = None

    subsets     = dict()
    params      = dict()
    variables   = dict()
    constraints = dict()

    def __init__(self,object,element):
        """ initialization of an element
        :param object: object of the abstract optimization model
        :param element: element that is added to the model"""

        self.model = object.model
        self.analysis = object.analysis
        self.system = object.system
        # set attributes
        self.name = element
        
        # add element to list
        Element.addElement(self)

    def getProperties(self, properties):
        """get properties (doc, dimensions, domain)
        :param  properties:      parameter, variable or constraint properties
        :return doc:             documentation
        :return dimensions:      dimensions of the parameter, variable, constraint
                                 (note: in case of constraints additional information can be passed with the keyword "Other")
        :return domain:          variable domain, empty for parameters and constraints"""

        doc        = properties
        dimensions = []
        domain     = []

        for propertyName in properties.split('.')[1:]:
            if 'Dimensions' in propertyName:
                dimensions = propertyName.split('.')[-1].split(':')[-1]
                if ',' in dimensions:
                    dimensions = dimensions.split(',')
                    dimensions = [getattr(self.model, dim.strip()) for dim in dimensions]
                else:
                    dimensions = [getattr(self.model, dimensions.strip())]
            elif 'Domain' in propertyName:
                domain = propertyName.split(':')[-1].strip()
                domain = getattr(pe, domain)

        return doc, dimensions, domain

    def addSets(self, subsets):
        """add sets or subsets to model
        :param sets: dictionary containing set names and properties"""

        for setName, setProperty in subsets.items():
            if 'Subset' in setProperty:
                subsetOf = setProperty.split(':')[-1].strip()
                peSet  = pe.Set(within= getattr(self.model, subsetOf), doc=setProperty)
                setattr(self.model, setName, peSet)
            if 'Alias' in setName:
                aliasOf   = setName.replace('Alias','')
                peSet = pe.SetOf(getattr(self.model, aliasOf))
            elif 'Index' in setName:
                index = setProperty.split(':').strip()[-1]
                peSet = pe.Set(index, doc=setProperty)
            elif 'Dimension' in setName:
                dimension = len(setProperty.split(':')[-1].split(','))
                peSet = pe.Set(dimen = dimension)
            elif 'Rule' in setName:
                rule = setProperty.split(':').strip()[-1]
                peSet = pe.Set(initialize=rule, doc=setProperty)
            else:
                peSet = pe.Set(doc=setProperty)

            setattr(self.model, setName, peSet)

    def addParams(self, params):
        """add parameters to model
        :param params: dictionary containing param names and properties"""

        for param, paramProperties in params.items():

            doc, dimensions, _ = self.getProperties(paramProperties)
            peParam            = pe.Param(*dimensions, doc=doc)
            
            setattr(self.model, param, peParam)

    def addVars(self, variables):
        """add variables to model
        :param variables: dictionary containing var names and properties"""

        for var, varProperties in variables.items():
            if not 'Dimensions' in varProperties:
                raise ValueError('Dimensions of variable {0} are undefined'.format(var))
            if not 'Domain' in varProperties:
                raise ValueError('Domain of variable {0} are undefined'.format(var))

            doc, dimensions, domain  = self.getProperties(varProperties)
            peVar                    = pe.Var(*dimensions, within=domain, doc=doc)

            setattr(self.model, var, peVar)

    def addConstr(self, constraints, replace = ['', ''], passValues = None):
        """add constraints to model
        :param constraints: dictionary containing var names and properties"""

        for constr, constrProperties in constraints.items():
            if not 'Dimensions' in constrProperties:
                raise ValueError('Dimensions of constraint {0} are undefined'.format(constr))

            doc, dimensions,_ = self.getProperties(constrProperties)
            if passValues:
                dimensions.insert(0, pe.Set(initialize=passValues))

            peConstr   = pe.Constraint(*dimensions, rule=getattr(self, f'constraint{constr.replace(*replace)}Rule'), doc=doc)

            setattr(self.model, f'constraint{constr}', peConstr)
                      
    ### --- classmethods --- ###
    # setter/getter classmethods
    @classmethod
    def addElement(cls,element):
        """ add element to element list. Inherited by child classes.
        :param element: new element that is to be added to the list """
        cls.listOfElements.append(element)

    @classmethod
    def setOptimizationAttributes(cls,analysis, system, pyoDict,model):
        """ set attributes of class <Element> with inputs 
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        :param pyoDict: input dictionary of optimization
        :param model: empty pe.ConcreteModel """
        # set analysis
        cls.analysis = analysis
        # set system
        cls.system = system
        # set input dictionary
        cls.pyoDict = pyoDict
        # set concreteModel
        cls.concreteModel = model
        
    @classmethod
    def getConcreteModel(cls):
        """ get concreteModel of the class <Element>. Every child class can access model and add components.
        :return concreteModel: pe.ConcreteModel """
        return cls.concreteModel

    @classmethod
    def getPyoDict(cls):
        """ get pyoDict 
        :return pyoDict: input dictionary of optimization """
        return cls.pyoDict

    @classmethod
    def getAnalysis(cls):
        """ get analysis of the class <Element>. 
        :return analysis: pe.analysis """
        return cls.analysis

    @classmethod
    def getSystem(cls):
        """ get system 
        :return system: input dictionary of optimization """
        return cls.system

    @classmethod
    def getAllElements(cls):
        """ get all elements in class. Inherited by child classes.
        :return cls.listOfElements: list of elements in this class """
        return cls.listOfElements

    @classmethod
    def getElement(cls,name:str):
        """ get single element in class by name. Inherited by child classes.
        :param name: name of element
        :return element: return element whose name is matched """
        for _element in cls.listOfElements:
            if _element.name == name:
                return _element
        return None

    @classmethod
    def getAllSubclasses(cls):
        """ get all subclasses (child classes) of cls 
        :return subclasses: subclasses of cls """
        return cls.__subclasses__()
    
    @classmethod
    def getAttributeOfAllElements(cls,attributeName:str):
        """ get attribute values of all elements in this class 
        :param attributeName: str name of attribute
        :return dictOfAttributes: returns dict of attribute values """
        _classElements = cls.getAllElements()
        dictOfAttributes = {}
        for _element in _classElements:
            if hasattr(_element,attributeName):
                _attribute = getattr(_element,attributeName)
                if isinstance(_attribute,dict):
                    # if attribute is dict
                    for _key in _attribute:
                        if isinstance(_key,tuple):
                            dictOfAttributes[(_element.name,)+_key] = _attribute[_key]
                        else:
                            dictOfAttributes[(_element.name,_key)] = _attribute[_key]
                else:
                    dictOfAttributes[_element.name] = _attribute

        return dictOfAttributes

    ### --- classmethods to define sets, parameters, variables, and constraints, that correspond to Element --- ###
    # Here, after defining Element-specific components, the components of the other classes are defined
    @classmethod
    def defineModelComponents(cls):
        """ defines the model components of the class <Element> """
        # define pe.Sets
        cls.defineSets()
        # define pe.Params
        cls.defineParams()
        # define pe.Vars
        cls.defineVars()
        # define pe.Constraints
        cls.defineConstraints()
        # define pe.Objective
        cls.defineObjective()

    @classmethod
    def defineSets(cls):
        """ defines the pe.Sets of the class <Element> """
        # define pe.Sets of the class <Element>
        model = cls.getConcreteModel()
        pyoDict = cls.getPyoDict()

        # nodes
        model.setNodes = pe.Set(
            initialize=pyoDict["setNodes"], 
            doc='Set of nodes')
        # connected nodes
        model.setAliasNodes = pe.Set(
            initialize=pyoDict["setNodes"],
            doc='Copy of the set of nodes to model edges. Subset: setNodes')
        # edges
        model.setEdges = pe.Set(
            initialize = pyoDict["setEdges"].keys(),
            doc = 'Set of edges'
        )
        # nodes on edges
        model.setNodesOnEdges = pe.Set(
            model.setEdges,
            initialize = pyoDict["setEdges"],
            doc = 'Set of nodes that constitute an edge. Edge connects first node with second node.'
        )
        # carriers
        model.setCarriers = pe.Set(
            initialize=pyoDict["setCarriers"],
            doc='Set of carriers')
        # technologies
        model.setTechnologies = pe.Set(
            initialize=pyoDict["setTechnologies"],
            doc='Set of technologies')
        # time-steps
        model.setTimeSteps = pe.Set(
            initialize=pyoDict["setTimeSteps"],
            doc='Set of time-steps')
        # scenarios
        model.setScenarios = pe.Set(
            initialize=pyoDict["setScenarios"],
            doc='Set of scenarios')

        # define pe.Sets of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.defineSets()

    @classmethod
    def defineParams(cls):
        """ defines the pe.Params of the class <Element> """
        # define pe.Params of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.defineParams()

    @classmethod
    def defineVars(cls):
        """ defines the pe.Vars of the class <Element> """
        # define pe.Vars of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.defineVars()

    @classmethod
    def defineConstraints(cls):
        """ defines the pe.Constraints of the class <Element> """
        # define pe.Constraints of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.defineConstraints()
    
    @classmethod
    def defineObjective(cls):
        """ defines the pe.Objective of the class <Element> """
        # get model
        model = cls.getConcreteModel()

        # get selected objective rule
        if cls.getAnalysis()["objective"] == "TotalCost":
            objectiveRule = objectiveTotalCostRule
        elif cls.getAnalysis()["objective"] == "CarbonEmissions":
            logging.info("Objective of carbon emissions not yet implemented")
            objectiveRule = objectiveCarbonEmissionsRule
        elif cls.getAnalysis()["objective"] == "Risk":
            logging.info("Objective of carbon emissions not yet implemented")
            objectiveRule = objectiveRiskRule
        else:
            logging.error("Objective type {} not known".format(cls.getAnalysis()["objective"]))

        # get selected objective sense
        if cls.getAnalysis()["sense"] == "minimize":
            objectiveSense = pe.minimize
        elif cls.getAnalysis()["sense"] == "maximize":
            objectiveSense = pe.maximize
        else:
            logging.error("Objective sense {} not known".format(cls.getAnalysis()["sense"]))

        # define objective
        model.objective = pe.Objective(
            rule = objectiveRule,
            sense = objectiveSense
        )

# different objective
def objectiveTotalCostRule(model):
    """objective function to minimize the total cost"""

    # CARRIERS
    carrierImport = sum(sum(sum(model.importCarrierFlow[carrier, node, time] * model.importPriceCarrier[carrier, node, time]
                            for time in model.setTimeSteps)
                        for node in model.setNodes)
                    for carrier in model.setImportCarriers)

    carrierExport = sum(sum(sum(model.exportCarrierFlow[carrier, node, time] * model.exportPriceCarrier[carrier, node, time]
                            for time in model.setTimeSteps)
                        for node in model.setNodes)
                    for carrier in model.setExportCarriers)

    return(carrierImport - carrierExport + model.capexTotalTechnology)

def objectiveCarbonEmissionsRule(model):
    """objective function to minimize total emissions"""

    # TODO implement objective functions for emissions
    return pe.Constraint.Skip

def objectiveRiskRule(model):
    """objective function to minimize total risk"""

    # TODO implement objective functions for risk
    return pe.Constraint.Skip

