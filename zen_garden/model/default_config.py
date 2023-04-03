"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Default configuration. Changes from the default values are specified in config.py (folders data/tests) and system.py (individual datasets)
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
        # This dictionary defines the configuration of the system by selecting the subset of technologies to be included into the analysis.
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
        self.analysis["objective"] = "total_cost"
        # typology of optimisation: minimize or maximize
        self.analysis["sense"]     = "minimize"
        # discount rate
        self.analysis["discount_rate"] = 0.06
        # transport distance (euclidean or actual)
        self.analysis["transport_distance"] = "Euclidean"
        # dictionary with subsets related to set
        self.analysis["subsets"] = {"set_carriers": [],
                               "set_technologies": ["set_conversion_technologies", "set_transport_technologies","set_storage_technologies"],
                               "set_conversion_technologies": ["set_conditioning_technologies"]}
        # headers for the generation of input files
        self.analysis["header_data_inputs"] = {
                "set_nodes": "node",
                "set_edges": "edge",
                "set_location": "location",
                "set_time_steps":"time", # IMPORTANT: time must be unique
                "set_time_steps_operation":"time_operation",
                "set_time_steps_storage_level":"time_storage_level",
                "set_time_steps_yearly":"year", # IMPORTANT: year must be unique
                "set_time_steps_yearly_entire_horizon":"year_entire_horizon",
                "set_carriers":"carrier",
                "set_input_carriers":"carrier",
                "set_output_carriers":"carrier",
                "set_dependent_carriers":"carrier",
                "set_conditioning_carriers":"carrier",
                "set_conditioning_carrier_parents":"carrier",
                "set_elements":"element",
                "set_conversion_technologies":"technology",
                "set_transport_technologies":"technology",
                "set_storage_technologies":"technology",
                "set_technologies":"technology",
                "set_existing_technologies": "existing_technology",
                "set_capacity_types":"capacity_type"}
        # time series aggregation
        self.analysis["time_series_aggregation"] = {
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
        self.analysis["folder_output"] = "./outputs/"
        self.analysis["overwrite_output"] = True
        # output format, can be h5, json or gzip
        self.analysis["output_format"] = "h5"
        self.analysis["write_results_yml"] = False
        self.analysis["max_output_size_mb"] = 500
        # name of data folder for energy system specification
        self.analysis["folder_name_system_specification"] = "system_specification"
        #earliest possible year of in input data, needed to differentiate between yearly and generic time indices
        self.analysis["earliest_year_of_data"] = 1900
        ## System - Items assignment
        # set of energy carriers
        self.system["set_carriers"] = []
        # set of conditioning carriers
        self.system["set_conditioning_carriers"] = []
        # set of capacity types: power-rated or energy-rated
        self.system["set_capacity_types"] = ["power","energy"]
        # set of conversion technologies
        self.system["set_conversion_technologies"] = []
        # set of conditioning technologies
        self.system["set_conditioning_technologies"] = []
        # set of storage technologies
        self.system["set_storage_technologies"] = []
        self.system["storage_periodicity"] = True
        # set of transport technologies
        self.system["set_transport_technologies"] = []
        self.system['double_capex_transport'] = False
        self.system["set_bidirectional_transport_technologies"] = []
        # set of nodes
        self.system["set_nodes"] = []
        # toggle to use time_series_aggregation
        self.system["conduct_time_series_aggregation"] = False
        # toggle to perform analysis for multiple scenarios
        self.system["conduct_scenario_analysis"] = False
        # total hours per year
        self.system["total_hours_per_year"] = 8760
        # unbounded market share for technology diffusion rate
        self.system["unbounded_market_share"] = 0.01
        # rate at which the knowledge stock of existing capacities is depreciated annually
        self.system["knowledge_depreciation_rate"] = 0.1
        # spillover rate of knowledge stock to another
        self.system["knowledge_spillover_rate"] = 0.05
        # social discount rate
        self.system["social_discount_rate"] = 0
        # enforce selfish behavior
        self.system["enforce_selfish_behavior"] = False

        ## Solver - Items assignment
        # solver selection (find more solver options for gurobi here: https://www.gurobi.com/documentation/9.1/refman/parameters.html)
        self.solver["name"]      = "glpk"
        # gurobi options
        self.solver["solver_options"] = {
            "logfile":      ".//outputs//logs//pyomoLogFile.log",
            "MIPGap":       None,
            "TimeLimit":    None,
            "Method":       None
        }
        # use symbolic labels, only sensible for debugging infeasible problems. Adds overhead
        self.solver["use_symbolic_labels"] = False
        # analyze numerics
        self.solver["analyze_numerics"]   = False
        self.solver["immutable_unit"]     = []
        self.solver["range_unit_exponents"]    = {"min":-1,"max":1,"step_width":1}
        # round down to number of decimal points, for new capacity and unit multipliers
        self.solver["rounding_decimal_points"]     = 5
        # round down to number of decimal points, for time series after TSA
        self.solver["rounding_decimal_points_ts"]   = 3
        # verbosity
        self.solver["verbosity"] = True
        # add duals 
        self.solver["add_duals"] = False
        # settings for selection of x-y relationships, which are modeled as PWA, and which are modeled linearly:
        # linear regression of x-y values: if relative intercept (intercept/slope) below threshold and rvalue above threshold, model linear with slope
        self.solver["linear_regression_check"] = {"eps_intercept":0.1,"epsRvalue":1-(1E-5)}

        ## Scenarios - dictionary declaration
        self.scenarios[""] = {}