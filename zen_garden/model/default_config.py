"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Default settings. Changes from the default values are specified in settings.py
==========================================================================================================================================================================="""


class Config(object):
    """
    A class that contains all relevant parameters for the ZEN-Garden
    """
    def __init__(self, analysis=None, solver=None, system=None, scenarios=None):
        """
        Initializes an instance of the parameters containing all defaults. If dictionaries are provided the defaults
        are overwritten.
        :param analysis: A dictionary used to update the default values in analysis
        :param solver: A dictionary used to update the default values in solver
        :param system: A dictionary used to update the default values in system
        :param scenarios: A dictionary used to update the default values in scenarios
        :return: A class instance of Config
        """

        ## Analysis - dictionary declaration
        # This dictionary contains all the settings related to the typology of analysis given a specific system configuration.
        # The dictionary also contains default settings related to the input data.
        self.analysis = dict()

        ## Solver - dictionary declaration
        # This dictionary contains all the settings related to the solver of the optimisation problem.
        self.solver = dict()

        ## System - dictionary declaration
        # This dictionary defines the configuration of the system by selecting the subset of technologies ot be included into the analysis.
        self.system = dict()

        ## Scenarios - dictionary declaration
        # This dictionary defines the set of scenarios that is evaluated.
        self.scenarios = dict()

        # set the defaults
        self._set_defaults()

        # update
        if analysis is not None:
            self.analysis.update(analysis)
        if solver is not None:
            self.solver.update(solver)
        if system is not None:
            self.system.update(system)
        if scenarios is not None:
            self.scenarios.update(scenarios)


    def _set_defaults(self):
        """
        Initializes all the default parameters
        """

        ## Analysis - Items assignment
        # objective function definition
        self.analysis["objective"] = "TotalCost"
        # typology of optimisation: minimize or maximize
        self.analysis["sense"]     = "minimize"
        # discount rate
        self.analysis["discount_rate"] = 0.06
        # transport distance (euclidean or actual)
        self.analysis["transportDistance"] = "Euclidean"
        # dictionary with subsets related to set
        self.analysis["subsets"] = {"setCarriers": [],
                               "setTechnologies": ["setConversionTechnologies", "setTransportTechnologies","setStorageTechnologies"],
                               "setConversionTechnologies": ["setConditioningTechnologies"]}
        # headers for the generation of input files
        self.analysis["headerDataInputs"] = {
                                            "setNodes": "node",
                                            "setEdges": "edge",
                                            "setLocation": "location",
                                            "setScenarios":"scenario",
                                            "set_time_steps":"time",
                                            "setTimeStepsOperation":"timeOperation",
                                            "setTimeStepsStorageLevel":"timeStorageLevel",
                                            "setTimeStepsYearly":"year",
                                            "set_time_steps_yearly_entire_horizon":"year",
                                            "setCarriers":"carrier",
                                            "setInputCarriers":"carrier",
                                            "setOutputCarriers":"carrier",
                                            "setDependentCarriers":"carrier",
                                            "setConditioningCarriers":"carrier",
                                            "setConditioningCarrierParents":"carrier",
                                            "set_elements":"element",
                                            "setConversionTechnologies":"technology",
                                            "setTransportTechnologies":"technology",
                                            "setStorageTechnologies":"technology",
                                            "setTechnologies":"technology",
                                            "setExistingTechnologies": "existingTechnology",
                                            "setCapacityTypes":"capacityType"}

        # file format of input data
        self.analysis["fileFormat"] = "csv"
        # time series aggregation
        self.analysis["timeSeriesAggregation"] = {
            "clusterMethod"         : "k_means",
            "solver"                : "gurobi",
            "hoursPerPeriod"        : 1,
            "extremePeriodMethod"   : "None",
            "rescaleClusterPeriods" : False,
            "representationMethod"  : "meanRepresentation",
            "resolution"            : 1,
            "segmentation"          : False,
            "noSegments"            : 12}

        self.analysis['postprocess'] = False
        self.analysis["folderOutput"] = "./outputs/"
        self.analysis["overwriteOutput"] = True
        self.analysis["compressOutput"] = True
        self.analysis["writeResultsYML"] = False
        self.analysis["maxOutputSizeMB"] = 500
        # name of data folder for energy system specification
        self.analysis["folderNameSystemSpecification"] = "systemSpecification"

        ## System - Items assignment
        # set of energy carriers
        self.system["setCarriers"] = []
        # set of conditioning carriers
        self.system["setConditioningCarriers"] = []
        # set of capacity types: power-rated or energy-rated
        self.system["setCapacityTypes"] = ["power","energy"]
        # set of conversion technologies
        self.system["setConversionTechnologies"] = []
        # set of conditioning technologies
        self.system["setConditioningTechnologies"] = []
        # set of storage technologies
        self.system["setStorageTechnologies"] = []
        # set of transport technologies
        self.system["setTransportTechnologies"] = []
        self.system['DoubleCapexTransport'] = False
        self.system["setBidirectionalTransportTechnologies"] = []
        # set of nodes
        self.system["setNodes"] = []
        # toggle to use timeSeriesAggregation
        self.system["conductTimeSeriesAggregation"] = False
        # toggle to perform analysis for multiple scenarios
        self.system["conductScenarioAnalysis"] = False
        # total hours per year
        self.system["totalHoursPerYear"] = 8760
        # unbounded market share for technology diffusion rate
        self.system["unboundedMarketShare"] = 0.01
        # rate at which the knowledge stock of existing capacities is depreciated annually
        self.system["knowledgeDepreciationRate"] = 0.1
        # spillover rate of knowledge stock to another
        self.system["knowledgeSpilloverRate"] = 0.05
        # social discount rate
        self.system["socialDiscountRate"] = 0
        # folder output



        ## Solver - Items assignment
        # solver selection (find more solver options for gurobi here: https://www.gurobi.com/documentation/9.1/refman/parameters.html)
        self.solver["name"]      = "glpk"
        # gurobi options
        self.solver["solverOptions"] = {
            "logfile":      ".//outputs//logs//pyomoLogFile.log",
            "MIPGap":       None,
            "TimeLimit":    None,
            "Method":       None
        }
        # use symbolic labels, only sensible for debugging infeasible problems. Adds overhead
        self.solver["useSymbolicLabels"] = False
        # analyze numerics
        self.solver["analyzeNumerics"]   = False
        self.solver["immutableUnit"]     = []
        self.solver["rangeUnitExponents"]    = {"min":-1,"max":1,"stepWidth":1}
        # round down to number of decimal points, for new capacity and unit multipliers
        self.solver["roundingDecimalPoints"]     = 5
        # round down to number of decimal points, for time series after TSA
        self.solver["roundingDecimalPointsTS"]   = 3
        # verbosity
        self.solver["verbosity"] = True
        # typology of model solved: MILP or MINLP
        self.solver["model"]      = "MILP"
        # parameters of meta-heuristic algorithm
        self.solver["parametersMetaheuristic"] = {
            "FEsMax":1e12, "kNumber":90, "mNumber":5, "q":0.05099, "xi":0.6795, "epsilon":1e-5, "MaxStagIter":650,
            "minVal":1e-6, "maxVal":1e6,"runsNumber":1
            }
        # evaluation of convergence in meta-heuristic. conditionDelta: (i) relative, (ii) absolute
        self.solver["convergenceCriterion"] = {"check": True, "conditionDelta":"relative", "restart":True}
        # settings for performance check
        self.solver["performanceCheck"] = {"printDeltaRun":1, "printDeltaIteration":1}
        # settings for selection of x-y relationships, which are modeled as PWA, and which are modeled linearly:
        # linear regression of x-y values: if relative intercept (intercept/slope) below threshold and rvalue above threshold, model linear with slope
        self.solver["linear_regression_check"] = {"eps_intercept":0.1,"epsRvalue":1-(1E-5)}

        ## Scenarios - dictionary declaration
        self.scenarios[""] = {}