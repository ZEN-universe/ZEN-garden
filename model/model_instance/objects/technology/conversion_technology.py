"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining the parameters, variables and constraints of the conversion technologies.
              The class takes the abstract optimization model as an input, and adds parameters, variables and
              constraints of the conversion technologies.
==========================================================================================================================================================================="""

import logging
from model.model_instance.objects.technology.technology import Technology

class ConversionTechnology(Technology):

    def __init__(self, object, tech):
        """init generic technology object
        :param object: object of the abstract model"""

        logging.info('initialize object of a conversion technology')
        super().__init__(object, 'Conversion', tech)

        subsets = {f'setInputCarriers{tech}':    f'Set of input carrier of {tech}. Subset: setInputCarriers',
                   f'setOutputCarriers{tech}':   f'Set of output carriers of {tech}. Subset: setOutputCarriers'}
        subsets = {**subsets, **self.getTechSubsets()}
        self.addSets(subsets)

        # merge new items with parameters dictionary from Technology class
        params = {}
        params = {**params, **self.getTechParams()}
        self.addParams(params)

        # %% Decision variables
        variables = {
            f'input{tech}': f'Carrier input of {tech}. \
                            \n\t Dimensions: setInputCarriers{tech}, setNodes, setTimeSteps. \
                            \n\t Domain: NonNegativeReals',
            f'output{tech}': f'Carrier output {tech}. \
                             \n\t Dimensions: setOutputCarriers{tech}, setNodes, setTimeSteps. \
                             \n\t Domain: NonNegativeReals'}
        # merge new items with variables dictionary from Technology class
        variables = {**variables, **self.getTechVars()}
        self.addVars(variables)

        #%% Constraints
        constr = {
            f'{tech}MaxOutput': f'Max amount of carrier produced with {tech}. \
                                \n\t Dimensions: setOutputCarriers{tech}, setNodes, setTimeSteps.'}
        # merge new items with constraints dictionary from Technology class
        constr = {**constr, **self.getTechConstr()}
        self.addConstr(constr, replace = [tech, 'ConversionTechnology'], passValues = [tech])

        # add linear/nonlinear constraints to model capex and conversion efficiency
        for type, nonLinearTechs in self.analysis['nonlinearTechnologyApproximation'].items():
            if tech in nonLinearTechs:
                self.addNonlinearConstraints(type, tech)
            else:
                self.addLinearConstraints(type, tech)

        logging.info(f'added subsets, parameters, decision variables and constraints for {tech}')


    def addNonlinearConstraints(self, type, tech):
        """add subsets, parameters, variables, and constraints for nonlinear problem formulation
        :param tech: conversion technology
        :param type: parameter type of the nonlinear function (capex or efficiency)"""

        pass
       #TODO add non-linear formulations

    def addLinearConstraints(self, type, tech):
        """add subsets, parameters, variables, and constraints for linearized problem formulation
        :param tech: conversion technology
        :param type: type of the function that is linearized (capex or efficiency)"""

        #%% Subsets
        subsets = {f'setSegments{type}{tech}': f'Set of support points for PWA of {type} for {tech}'}
        self.addSets(subsets)

        #%% Parameters
        params = {f'slope{type}{tech}':     f'Parameter which specifies the slope of the {type} segment {tech}.\
                                            \n\t Dimensions: setSegments{type}{tech}',
                  f'intercept{type}{tech}': f'Parameter which specifies the intercept of the {type} segment {tech}.\
                                            \n\t Dimensions: setSegments{type}{tech}',
                  f'lbSegment{type}{tech}': f'Parameter which specifies the lower bound of the {type} segment {tech}.\
                                            \n\t Dimensions: setSegments{type}{tech}',
                  f'ubSegment{type}{tech}': f'Parameter which specifies the upper bound of the {type} segment {tech}.\
                                            \n\t Dimensions: setSegments{type}{tech}'}
        self.addParams(params)

        #%% Variables
        variables = {
            f'selectSegment{type}{tech}': f'Binary variable to model the activation of a segment in the PWA approximation of {type} of the {tech}. \
                                          \n\t Dimensions: setSegments{type}{tech}, setNodes, setTimeSteps.\
                                          \n\t Domain: Binary'}
        if type == 'Capex':
            variables[f'capacityAux{tech}'] = f'Auxiliary variable to model {type} of {tech} technologies. \
                                              \n\t Dimensions: setSegments{type}{tech}, setNodes, setTimeSteps.\
                                              \n\t Domain: NonNegativeReals'
        elif type == 'ConverEfficiency':
            variables[f'inputAux{tech}']   = f'Auxiliary variable to model {type} of {tech} technologies. \
                                             \n\t Dimensions: setSegments{type}{tech}, setInputCarriers{tech}, setNodes, setTimeSteps.\
                                             \n\t Domain: NonNegativeReals'
        self.addVars(variables)

        #%% Constraints
        constr = dict()
        if type == 'Capex':
            constr[f'{tech}Linear{type}']    = f'Linearization of {type} for {type} of {tech}.\
                                               \n\t Dimensions: setSegments{type}{tech}, setNodes, setTimeSteps'
            constr[f'{tech}Linear{type}LB']  = f'lower bound of segment for {type} of {tech}.\
                                               \n\t Dimensions: setSegments{type}{tech}, setNodes, setTimeSteps'
            constr[f'{tech}Linear{type}UB']  = f'upper bound of segment for {type} of {tech}.\
                                               \n\t Dimensions: setSegments{type}{tech}, setNodes, setTimeSteps'
            constr[f'{tech}Linear{type}Aux'] = f'linking the auxiliary variable and variable for {type} of {tech}.\
                                               \n\t Dimensions: setNodes, setTimeSteps'

        elif type == 'ConverEfficiency':
            constr[f'{tech}Linear{type}']    = f'Linearization of {type} for {type} of {tech}.\
                                               \n\t Dimensions: setSegments{type}{tech}, setInputCarriers{tech}, setOutputCarriers{tech}, setNodes, setTimeSteps'
            constr[f'{tech}Linear{type}LB']  = f'lower bound of segment for {type} of {tech}.\
                                               \n\t Dimensions: setSegments{type}{tech}, setInputCarriers{tech}, setNodes, setTimeSteps'
            constr[f'{tech}Linear{type}UB']  = f'upper bound of segment for {type} of {tech}.\
                                               \n\t Dimensions: setSegments{type}{tech}, setInputCarriers{tech}, setNodes, setTimeSteps'
            constr[f'{tech}Linear{type}Aux'] = f'linking the auxiliary variable and variable for {type} of {tech}.\
                                               \n\t Dimensions: setInputCarriers{tech}, setNodes, setTimeSteps'

        constr[f'{tech}Linear{type}SegmentSelection']  = f'Segment selection for {type} of {tech}.\
                                                         \n\t Dimensions: setNodes, setTimeSteps.'

        self.addConstr(constr, replace = [tech, 'ConversionTechnology'], passValues = [tech])


    #%% Constraint rules pre-defined in Technology class
    @staticmethod
    def constraintConversionTechnologyAvailabilityRule(model, tech, node, time):
        """limited availability of conversion technology"""

        # parameters
        availabilityTechnology = getattr(model, f'availability{tech}')
        # variables
        installTechnology      = getattr(model, f'install{tech}')

        return (availabilityTechnology[node, time] >= installTechnology[node, time])

    @staticmethod
    def constraintConversionTechnologyMinCapacityRule(model, tech, node, time):
        """min size of conversion technology"""

        # parameters
        minCapacityTechnology = getattr(model, f'minCapacity{tech}')
        # variables
        installTechnology     = getattr(model, f'install{tech}')
        capacityTechnology    = getattr(model, f'capacity{tech}')

        return(minCapacityTechnology * installTechnology[node, time] <= capacityTechnology[node, time])
    
    @staticmethod
    def constraintConversionTechnologyMaxCapacityRule(model, tech, node, time):
        """max size of conversion technology"""

        # parameters
        maxCapacityTechnology = getattr(model, f'minCapacity{tech}')
        # variables
        installTechnology  = getattr(model, f'install{tech}')
        capacityTechnology = getattr(model, f'capacity{tech}')

        return(maxCapacityTechnology * installTechnology[node, time] >= capacityTechnology[node, time])
        
    #%% Constraint rules defined in current class - Operation
    @staticmethod
    def constraintConversionTechnologyMaxOutputRule(model, tech, carrierOut, node, time):
        """max amount of carrier produced with conversion technology."""

        # variables
        capacityTechnology = getattr(model, f'capacity{tech}')
        outputTechnology   = getattr(model, f'output{tech}')

        return (capacityTechnology[node, time]
                >= outputTechnology[carrierOut, node, time])


    #%% Constraint rules defined in current class  - Conversion Efficiency
    @staticmethod
    def constraintConversionTechnologyLinearConverEfficiencyRule(model, tech, segment, carrierIn, carrierOut, node, time):
        """linearized conversion efficiency of conversion technology"""

        # parameters
        slope     = getattr(model,f'slopeConverEfficiency{tech}')
        intercept = getattr(model,f'interceptConverEfficiency{tech}')
        # variables
        inputTechnologyAux = getattr(model,f'inputAux{tech}')
        outputTechnology   = getattr(model,f'output{tech}')

        return(outputTechnology[carrierOut, node, time]
                    == slope[segment] * inputTechnologyAux[segment, carrierIn, node, time] + intercept[segment])


    @staticmethod
    def constraintConversionTechnologyLinearConverEfficiencyLBRule(model, tech, segment, carrierIn, node, time):
        """lower bound of the current segment"""

        # parameters
        lbSegment = getattr(model, f'lbSegmentConverEfficiency{tech}')
        # variables
        inputTechnologyAux = getattr(model, f'inputAux{tech}')
        selectSegment      = getattr(model, f'selectSegmentConverEfficiency{tech}')

        return(selectSegment[segment, node, time] * lbSegment[segment]
                   <= inputTechnologyAux[segment, carrierIn, node, time])

    @staticmethod
    def constraintConversionTechnologyLinearConverEfficiencyUBRule(model, tech, segment, carrierIn, node, time):
        """upper bound of the current segment"""

        # parameters
        ubSegment     = getattr(model, f'ubSegmentConverEfficiency{tech}')
        # variables
        inputTechnologyAux = getattr(model, f'inputAux{tech}')
        selectSegment = getattr(model, f'selectSegmentConverEfficiency{tech}')

        return (selectSegment[segment, node, time] * ubSegment[segment]
                >= inputTechnologyAux[segment, carrierIn, node, time])

    @staticmethod
    def constraintConversionTechnologyLinearConverEfficiencyAuxRule(model, tech, carrierIn, node, time):
        """link auxiliary variable and actual variable"""

        # sets
        setSegments = getattr(model, f'setSegmentsConverEfficiency{tech}')
        # variables
        inputTechnologyAux = getattr(model, f'inputAux{tech}')
        inputTechnology    = getattr(model, f'input{tech}')

        return(sum(inputTechnologyAux[segment, carrierIn, node, time] for segment in setSegments)
                == inputTechnology[carrierIn, node, time])

    @staticmethod
    def constraintConversionTechnologyLinearConverEfficiencySegmentSelectionRule(model, tech, node, time):
        """only select one segment at the time"""

        # sets
        setSegments = getattr(model, f'setSegmentsConverEfficiency{tech}')
        # variables
        installTechnology = getattr(model, f'install{tech}')
        selectSegment     = getattr(model, f'selectSegmentConverEfficiency{tech}')

        return(sum(selectSegment[segment, node, time] for segment in setSegments)
               <= installTechnology[node, time])

    # %% Constraint rules defined in current class - Capital Expenditures (Capex)
    @staticmethod
    def constraintConversionTechnologyLinearCapexRule(model, tech, segment, node, time):
        """linearized conversion efficiency of conversion technology"""

        # parameters
        slope     = getattr(model, f'slopeCapex{tech}')
        intercept = getattr(model, f'interceptCapex{tech}')
        # variables
        capexTechnology       = getattr(model, f'capex{tech}')
        capacityTechnologyAux = getattr(model, f'capacityAux{tech}')

        return (capexTechnology[node, time]
                    == slope[segment] * capacityTechnologyAux[segment, node, time] + intercept[segment])

    @staticmethod
    def constraintConversionTechnologyLinearCapexLBRule(model, tech, segment, node, time):
        """lower bound of the current segment"""

        # parameters
        lbSegment     = getattr(model, f'lbSegmentCapex{tech}')
        # variables
        capacityTechnologyAux = getattr(model, f'capacityAux{tech}')
        selectSegment         = getattr(model, f'selectSegmentCapex{tech}')

        return (selectSegment[segment, node, time] * lbSegment[segment]
                <= capacityTechnologyAux[segment, node, time])

    @staticmethod
    def constraintConversionTechnologyLinearCapexUBRule(model, tech, segment, node, time):
        """upper bound of the current segment"""

        # parameters
        ubSegment = getattr(model, f'ubSegmentCapex{tech}')
        # variables
        capacityTechnologyAux = getattr(model, f'capacityAux{tech}')
        selectSegment         = getattr(model, f'selectSegmentCapex{tech}')

        return(selectSegment[segment, node, time] * ubSegment[segment]
                >= capacityTechnologyAux[segment, node, time])

    @staticmethod
    def constraintConversionTechnologyLinearCapexAuxRule(model, tech, node, time):
        """link auxiliary variable and actual variable"""

        #sets
        setSegments = getattr(model, f'setSegmentsCapex{tech}')
        # variables
        capacityTechnologyAux = getattr(model, f'capacityAux{tech}')
        capacityTechnology    = getattr(model, f'capacity{tech}')

        return(sum(capacityTechnologyAux[segment, node, time] for segment in setSegments)
                == capacityTechnology[node, time])

    @staticmethod
    def constraintConversionTechnologyLinearCapexSegmentSelectionRule(model, tech, node, time):
        """only one segment can be selected at a time, and only if the technology is built (installTechnology =1)"""

        # sets
        setSegments = getattr(model, f'setSegmentsCapex{tech}')
        # variables
        installTechnology = getattr(model, f'install{tech}')
        selectSegment = getattr(model, f'selectSegmentCapex{tech}')

        return(sum(selectSegment[segment, node, time] for segment in setSegments)
                <= installTechnology[node, time])


    #%% TODO implement conditioning for e.g. hydrogen