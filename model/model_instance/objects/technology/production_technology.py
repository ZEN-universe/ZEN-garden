"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining the parameters, variables and constraints of the production technologies.
              The class takes the abstract optimization model as an input, and adds parameters, variables and
              constraints of the production technologies.
==========================================================================================================================================================================="""

import logging
from model.model_instance.objects.technology.technology import Technology

class ProductionTechnology(Technology):

    def __init__(self, object):
        """init generic technology object
        :param object: object of the abstract model"""

        logging.info('initialize object of a production technology')
        super().__init__(object, 'Production')

        subsets = {}
        subsets = {**subsets, **self.getTechSubsets()}
        self.addSubsets(subsets)

        # %% Paramaters
        params = {'converAvailability': 'Parameter that links production technology input, and output carriers. \
                                         \n\t Dimensions: setProductionTechnologies, setInputCarriers, setOutputCarriers'}
        # merge new items with parameters dictionary from Technology class
        params = {**params, **self.getTechParams()}
        self.addParams(params)

        # %% Decision variables
        variables = {
            'inputProductionTechnologies': 'Input stream of a carrier into production technology. \
                                            \n\t Dimensions: setInputCarriers, setProductionTechnologies, setNodes, setTimeSteps. \
                                            \n\t Domain: NonNegativeReals',
            'outputProductionTechnologies': 'Output stream of a carrier into production technology. \
                                             \n\t Dimensions: setOutputCarriers, setProductionTechnologies, setNodes, setTimeSteps. \
                                             \n\t Domain: NonNegativeReals',
        }
        # merge new items with variables dictionary from Technology class
        variables = {**variables, **self.getTechVars()}
        self.addVars(variables)

        # %% Constraints
        constr = {
            # operation
            'constraintProductionTechnologiesFlowCapacity': 'Couple the flow with the installed capacity. \
                                                             \n\t Dimensions: setProductionTechnologies, setInputCarriers, setOutputCarriers, setNodes, setTimeSteps.',
            'constraintProductionTechnologiesMaxLoad':      'Max amount of carrier produced with production technology. \
                                                             \n\t Dimensions: setProductionTechnologies, setOutputCarriers, setNodes, setTimeSteps.'}
        # merge new items with constraints dictionary from Technology class
        constr = {**constr, **self.getTechConstr()}
        self.addConstr(constr)

        for tech in self.system['setProductionTechnologies']:
            if self.analysis['technologyApproximationCapex'] == 'nonlinear': #TODO extend technologyApproximationCapex by tech --> parameter should be tech-specific
                self.addNonlinearFormulation(tech, 'Capex')
            else:
                self.addLinearFromulation(tech, 'Capex')
            if self.analysis['technologyApproximationEfficiency'] == 'nonlinear':
                self.addNonlinearFormulation(tech, 'Efficiency')
            else:
                self.addLinearFormulation(tech, 'ConverEfficiency')


    def addNonlinearFormulation(self, type):
        """add subsets, parameters, variables, and constraints for nonlinear problem formulation
        :param tech: production technology
        :param type: parameter type of the nonlinear function (capex or efficiency)"""

        pass
       #TODO add non-linear formulations

    def addLinearFormulation(self, tech, type):
        """add subsets, parameters, variables, and constraints for linearized problem formulation
        :param tech: production technology
        :param type: type of the function that is linearized (capex or efficiency)"""

        # %% Subsets
        subsets = {f'set{type}Segments{tech}': f'Set of support points for piecewise affine linearization of {type} for {tech}.'}
        # alternatively, it is also possible to create indexed subsets. I.e. segments[tech]. For implementing this, element would need to be updated
        self.addSubsets(subsets)

        # %% Parameters
        params = {f'slope{type}{tech}':    f'Parameter which specifies the slope of the segment of the {type} of the {tech}.\
                                           \n\t Dimensions: set{type}Segments{tech}',
                  f'intercept{type}{tech}': f'Parameter which specifies the intercept of the segment of {type} of the {tech}.\
                                           \n\t Dimensions: set{type}Segments{tech}',
                  f'lbSegment{type}{tech}': f'Parameter which specifies the lower bound of the segment of {type} of the {tech}.\
                                           \n\t Dimensions: set{type}Segments{tech}',
                  f'ubSegment{type}{tech}': f'Parameter which specifies the upper bound of the segment of{type} of the {tech}.\
                                           \n\t Dimensions: set{type}Segments{tech}'}
        self.addParams(params)

        # %% Constraints
        if type == 'Capex':
            dim = 'setNodes, setTimeSteps'
        elif type == 'converEfficiency':
            dim = 'setInputCarriers, setNodes, setTimeSteps'

        variables = {f'select{type}Segment{tech}':  f'Binary variable to model the activation of a segment in the PWA approaximation of {type} of the {tech}. \
                                                    \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps, setPWACapex.\
                                                    \n\t Domain: Binary',
                     f'{type}{tech}Aux':            f'Auxiliary variable to model {type} of the {tech}. \
                                                    \n\t Dimensions: set{type}Segments{tech}, {tech}, {dim}.'}
        self.addVars(variables)

        # %% Constraints
        if type == 'converEfficiency':
            dim = 'setInputCarriers, setOutputCarriers, setNodes, setTimeSteps'

        constr = {f'constraintLinear{tech}{type}':                  f'Linearization of {type} for {tech}. \
                                                                    \n\t Dimensions: {tech}, {dim}, set{type}Segments{tech}',
                  f'constraintLinear{tech}{type}LB':                f'lower bound of segment of {type} for {tech}. \
                                                                    \n\t Dimensions:  {tech}, {dim}, set{type}Segments{tech}',
                  f'constraintLinear{tech}{type}UB':                f'upper bound of segment of {type} for {tech}. \
                                                                    \n\t Dimensions: {tech}, {dim}, set{type}Segments{tech}.',
                  f'constraintLinear{tech}{type}Aux':               f'linking the auxiliary variable of {type} for {tech} with the variable. \
                                                                    \n\t Dimensions: {tech}, {dim}, set{type}Segments{tech}.',
                  f'constraintLinear{tech}{type}SegmentSelection':  f'Activation of auxiliary variable based on upper bound of PWA domain. \
                                                                    \n\t Dimensions: {tech}, setNodes, setTimeSteps.'}
        self.addConstr(constr, replace = [tech, 'ProductionTechnology'])
            
        logging.info('added production technology sets, parameters, decision variables and constraints')        
            
        
    #%% Constraint rules pre-defined in Technology class
    @staticmethod
    def constraintProductionTechnologiesRuleAvailabilityRule(model, tech, node, time):
        """limited availability of production technology"""

        return (model.availabilityProduction[tech, node, time]
                >= model.installProductionTechnologies[tech, node, time])

    @staticmethod
    def constraintProductionTechnologiesMinCapacityRule(model, tech, node, time):
        """min size of production technology"""

        return(model.minCapacityProduction[tech] * model.installProductionTechnologies[tech, node, time]
               <= model.capacityProductionTechnologies[tech, node, time])
    
    @staticmethod
    def constraintProductionTechnologiesMaxCapacityRule(model, tech, node, time):
        """max size of production technology"""

        return(model.maxCapacityProduction[tech] * model.installProductionTechnologies[tech, node, time]
                >= model.capacityProductionTechnologies[tech, node, time])
        
    #%% Constraint rules defined in current class
    @staticmethod
    def constraintProductionTechnologiesMaxLoadRule(model, carrierOut, tech, node, time):
        """max amount of carrier produced with production technology."""

        return (model.capacityProductionTechnologies[tech, node, time]
                >= model.outputProductionTechnologies[carrierOut, tech, node, time])


    #%% Constraint rules defined in current class  - Conversion Efficiency
    @staticmethod
    def constraintProductionTechnologyLinearConverEfficiencyRule(model, tech, carrierIn, carrierOut, node, time, segment):
        """linearized conversion efficiency of production technology"""

        if model.converAvailability[tech, carrierIn, carrierOut] == 1:
            # parameters
            slope     = getattr(model,f'slopeConverEfficiency{tech}')
            intercept = getattr(model,f'interceptConverEfficiency{tech}')
            # variables
            inputProductionTechnologiesAux = getattr(model,f'ConverEfficiency{tech}Aux')

            return( model.outputProductionTechnologies[carrierOut, tech, node, time]
                    == slope[segment] * inputProductionTechnologiesAux[segment, carrierIn, tech, node, time]
                    + intercept[segment])

        else:
            return(model.outputProductionTechnologies[carrierOut, tech, node, time] == 0)

    @staticmethod
    def constraintProductionTechnologyLinearConverEfficiencyLBRule(model, segment, tech, carrierIn, carrierOut, node, time):
        """lower bound of the current segment"""

        # variables
        inputProductionTechnologiesAux = getattr(model, f'ConverEfficiency{tech}Aux')

        if model.converAvailability[tech, carrierIn, carrierOut] == 1:
            # parameters
            lbSegment     = getattr(model, f'lbSegmentConverEfficiency{tech}')
            selectSegment = getattr(model, f'selectConverEfficiencySegment{tech}')

        return(selectSegment[tech] * lbSegment[tech]
                   <= inputProductionTechnologiesAux[segment, carrierIn, tech, node, time])

        else:
            return(inputProductionTechnologiesAux[segment, carrierIn, tech, node, time] == 0)

    @staticmethod
    def constraintProductionTechnologyLinearConverEfficiencyUBRule(model, segment, tech, carrierIn, carrierOut, node, time):
        """upper bound of the current segment"""

        if model.converAvailability[tech, carrierIn, carrierOut] == 1:
            # parameters
            ubSegment     = getattr(model, f'ubSegmentConverEfficiency{tech}')
            selectSegment = getattr(model, f'selectConverEfficiencySegment{tech}')
            # variables
            inputProductionTechnologiesAux = getattr(model, f'ConverEfficiency{tech}Aux')

            return (selectSegment[segment] * ubSegment[tech]
                    >= inputProductionTechnologiesAux[segment, carrierIn, tech, node, time])

        else:
            return(pe.Constraint.Skip)

    @staticmethod
    def constraintProductionTechnologyLinearConverEfficiencyAuxRule(model, tech, carrierIn, node, time):
        """link auxiliary variable and actual variable"""

        # sets
        setSegments = getattr(model, f'setConverEfficiencySegments{tech}')
        # variables
        inputProductionTechnologiesAux = getattr(model, f'ConverEfficiency{tech}Aux')

        return (sum(inputProductionTechnologiesAux[segment, carrierIn, tech, node, time] for segment in setSegments)
                == model.inputProductionTechnologies[carrierIn, tech, node, time])

    @staticmethod
    def constraintLinearProductionTechnologyLinearConverEfficiencySegmentSelectionRule(model, tech, node, time):
        """only select one segment at the time"""

        setSegments = getattr(model, f'setConverEfficiencySegments{tech}')

        return(sum(selectSegment[segment, node, time] for segment in setSegments)
               <= model.installProductionTechnologies[tech, node, time])

    # %% Constraint rules defined in current class - Capital Expenditures (Capex)
    @staticmethod
    def constraintProductionTechnologyLinearCapexRule(model, segment, tech, carrierIn, node, time):
        """linearized conversion efficiency of production technology"""

        # parameters
        slope = getattr(model, f'slopeCapex{tech}')
        intercept = getattr(model, f'interceptCapex{tech}')
        # variables
        capacityProductionTechnologiesAux = getattr(model, f'Capex{tech}Aux')

        return (model.capexProductionTechnology[tech, node, time]
                == slope[segment] * capacityProductionTechnologiesAux[segment, tech, node, time]
                + intercept[segment])

    @staticmethod
    def constraintProductionTechnologyLinearCapexLBRule(model, segment, tech, node, time):
        """lower bound of the current segment"""

        # parameters
        lbSegment = getattr(model, f'lbSegmentCapex{tech}')
        selectSegment = getattr(model, f'selectCapex{tech}')
        # variables
        capacityProductionTechnologiesAux = getattr(model, f'Capex{tech}Aux')

        return (selectSegment[tech] * lbSegment[tech]
                <= capacityProductionTechnologiesAux[segment, tech, node, time])

    @staticmethod
    def constraintProductionTechnologyLinearCapexUBRule(model, segment, tech, node, time):
        """upper bound of the current segment"""

        # parameters
        ubSegment = getattr(model, f'ubSegmentCapex{tech}')
        selectSegment = getattr(model, f'selectCapexSegment{tech}')
        # variables
        capacityProductionTechnologiesAux = getattr(model, f'Capex{tech}Aux')

        return (selectSegment[segment] * ubSegment[tech]
                >= capacityProductionTechnologiesAux[segment, tech, node, time])

    @staticmethod
    def constraintProductionTechnologyLinearCapexAuxRule(model, tech, node, time):
        """link auxiliary variable and actual variable"""

        #sets
        setSegments = getattr(model, f'setCapexSegments{tech}')
        # variables
        capacityProductionTechnologiesAux = getattr(model, f'Capex{tech}Aux')

        return (sum(capacityProductionTechnologiesAux[segment, tech, node, time] for segment in setSegments)
                == model.capacityProductionTechnologies[tech, node, time])

    @staticmethod
    def constraintLinearProductionTechnologyLinearCapexSegmentSelectionRule(model, tech, node, time):
        """only one segment can be selected at a time, and only if the technology is built (installProductionTechnology =1)"""

        # sets
        setSegments = getattr(model, f'setConverEfficiencySegments{tech}')

        return (sum(selectSegment[segment, node, time] for segment in setSegments)
                <= model.installProductionTechnologies[tech, node, time])


    #%% Constraint rules defined in current class - Design
    @staticmethod
    def constraintProductionTechnologiesCapexRule(model, tech, node, time):
        """definition of PWA capex approximation based on supporting points"""
        
        expression = (
            model.PWACapex[tech, node, time] 
            == 
            sum(model.slopePWACapex[tech, point] *
            (model.capacityAuxProductionTechnologies[tech, node, time] - model.extreme0PWACapex[tech, point] * model.auxiliaryPWACapex[tech, node, time, point]) +
            model.value0PWACapex[tech, point] * model.auxiliaryPWACapex[tech, node, time, point]
            for point in model.setPWACapex)
            ) 
        return expression

    @staticmethod
    def constraintProductionTechnologiesAuxiliaryPWACapex1Rule(model, tech, node, time):
        """constraint on activation of single segment capex approximation"""
        return (sum(model.auxiliaryPWACapex[tech, node, time, point] for point in model.setPWACapex), 1)

    @staticmethod
    def constraintProductionTechnologiesAuxiliaryPWACapex2Rule(model, tech, node, time):
        """
        activation of auxiliary variable based on lower bound of PWA domain
        """
        expression = (
            model.capacityProductionTechnologies[tech, node, time] 
            >=
            sum(model.auxiliaryPWACapex[tech, node, time, point] * model.extreme0PWACapex[tech, point] for point in model.setPWACapex)
            )
        
        return expression
        
    @staticmethod
    def constraintProductionTechnologiesAuxiliaryPWACapex3Rule(model, tech, node, time):
        """
        activation of auxiliary variable based on upper bound of PWA domain
        """        
        expression = (
            model.capacityProductionTechnologies[tech, node, time]
            <=
            sum(model.auxiliaryPWACapex[tech, node, time, point] * model.extreme1PWACapex[tech, point] for point in model.setPWACapex)
            )
        
        return expression
    
    @staticmethod
    def constraintProductionTechnologiesAuxiliaryPWACapex4Rule(model, tech, node, time, point):    
        """
        definition of auxiliary variable installation capacity with upper bound from auxiliary binary variable
        """     
        expression = (
            model.capacityAuxProductionTechnologies[tech, node, time] 
            <= 
            model.maxCapacityProduction[tech] * model.auxiliaryPWACapex[tech, node, time, point]
            )
        return expression
                   
    @staticmethod
    def constraintProductionTechnologiesAuxiliaryPWACapex5Rule(model, tech, node, time, point):    
        """
        definition of auxiliary variable installation capacity with lower bound from auxiliary binary variable
        """     
        expression = (
            model.capacityAuxProductionTechnologies[tech, node, time]
            >= model.minCapacityProduction[tech] * model.auxiliaryPWACapex[tech, node, time, point])
        return expression
        
    @staticmethod
    def constraintProductionTechnologiesAuxiliaryPWACapex6Rule(model, tech, node, time):    
        """
        connection of auxiliary variable installation capacity with variable installation capacity
        """     
        expression = (model.capacityAuxProductionTechnologies[tech, node, time]
            <= model.capacityProductionTechnologies[tech, node, time])
        return expression
        
    @staticmethod
    def constraintProductionTechnologiesAuxiliaryPWACapex7Rule(model, tech, node, time, point):    
        """
        connection of auxiliary variable installation capacity with variable installation capacity
        """     
        expression = (model.capacityAuxProductionTechnologies[tech, node, time]
            >= model.capacityProductionTechnologies[tech, node, time] - \
                model.maxCapacityProduction[tech] * (1-model.auxiliaryPWACapex[tech, node, time, point]))
        return expression

    #TODO implement conditioning for e.g. hydrogen
    
    #%% Contraint rules defined in current class - Operation    
    
    @staticmethod
    def constraintProductionTechnologiesFlowCapacityRule(model, tech, carrierIn, carrierOut, node, time):
        """
        coupling the output energy flow and the capacity of production technology
        """
        
        if model.converAvailability[tech, carrierIn, carrierOut] == 1:
            expression = (model.capacityProductionTechnologies[tech, node, time]
                >= model.outputProductionTechnologies[carrierOut, tech, node, time])
        else:
            expression = (model.outputProductionTechnologies[carrierOut, tech, node, time] == 0)
        return expression    
        
    
    #'converEnergy': 'energy involved in conversion of carrier. \n\t Dimensions: setCarriers, setNodes, setTimeSteps. \n\t Domain: NonNegativeReals'
    
    # 'outputProductionTechnologiesAux': 'Auxiliary variable to describe output stream of a carrier into production technology. \
    #                                     \n\t Dimensions: setOutputCarriers, setProductionTechnologies, setNodes, setTimeSteps. \n\t Domain: NonNegativeReals'
    
    # 'constraintMinLoadProductionTechnologies1':    'min load of production technology, part one. \
    #                                                 \n\t Dimensions: setOutputCarriers, setProductionTechnologies, setNodes, setTimeSteps',
    # 'constraintMinLoadProductionTechnologies2':    'min load of production technology, part two. \
    #                                                 \n\t Dimensions: setOutputCarriers, setProductionTechnologies, setNodes, setTimeSteps',
    # 'constraintMaxLoadProductionTechnologies1':    'max load of production technology, part one. \
    #                                                 \n\t Dimensions: setOutputCarriers, setProductionTechnologies, setNodes, setTimeSteps',
    # 'constraintMaxLoadProductionTechnologies2':    'max load of production technology, part two. \
    #                                                 \n\t Dimensions: setOutputCarriers, setProductionTechnologies, setNodes, setTimeSteps' 
    # @staticmethod
    # def constraintMinLoadProductionTechnologies1Rule(model, carrier, tech, node, time):
    #     """
    #     min amount of carrier produced with production technology
    #     """
    #     expression = (
    #         model.minLoadProduction[tech] *  model.minCapacityProduction[tech] * model.installProductionTechnologies[tech, node, time]
    #         <=
    #         model.outputProductionTechnologiesAux[carrier, tech, node, time]
    #         )
    #     return expression
    # @staticmethod
    # def constraintMinLoadProductionTechnologies2Rule(model, carrier, tech, node, time):
    #     """min amount of carrier produced with production technology between two nodes.
    #     \n\t Dimensions: setCarrier, setProductionTechnologies, setNodes, setTimeSteps"""

    #     return (model.outputProductionTechnologies[carrier, tech, node, time]
    #             - model.maxCapacityProduction[tech] * (1 - model.installProductionTechnologies[tech, node, time])
    #             <= model.outputProductionTechnologiesAux[carrier, tech, node, time])

    # @staticmethod
    # def constraintMaxLoadProductionTechnologies1Rule(model, carrier, tech, node, time):
    #     """max amount of carrier produced with production technology.
    #     \n\t Dimensions: setCarrier, setProductionTechnologies, setNodes, setTimeSteps"""

    #     return (model.capacityProductionTechnologies[tech, node, time]
    #             >= model.outputProductionTechnologiesAux[carrier, tech, node, time])

    # @staticmethod
    # def constraintMaxLoadProductionTechnologies2Rule(model, carrier, tech, node, time):
    #     """max amount of carrier produced with production technology.
    #     \n\t Dimensions: setCarrier, setProductionTechnologies, setNodes, setTimeSteps"""

    #     return (model.outputProductionTechnologies[carrier, tech, node, time]
    #             >= model.outputProductionTechnologiesAux[carrier, tech, node, time])
      