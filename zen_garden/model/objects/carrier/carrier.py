"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class defining a generic energy carrier.
                The class takes as inputs the abstract optimization model. The class adds parameters, variables and
                constraints of a generic carrier and returns the abstract optimization model.
==========================================================================================================================================================================="""
import logging
import time

import linopy as lp
import numpy as np
import xarray as xr

from zen_garden.utils import lp_sum
from ..component import ZenIndex
from ..element import Element


class Carrier(Element):
    # set label
    label = "set_carriers"
    # empty list of elements
    list_of_elements = []

    def __init__(self, carrier: str, optimization_setup):
        """initialization of a generic carrier object
        :param carrier: carrier that is added to the model
        :param optimization_setup: The OptimizationSetup the element is part of """

        logging.info(f'Initialize carrier {carrier}')
        super().__init__(carrier, optimization_setup)
        # store input data
        self.store_input_data()

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        set_base_time_steps_yearly = self.energy_system.set_base_time_steps_yearly
        set_time_steps_yearly = self.energy_system.set_time_steps_yearly
        # set attributes of carrier
        # raw import
        self.raw_time_series = {}
        self.raw_time_series["demand_carrier"] = self.data_input.extract_input_data("demand_carrier", index_sets=["set_nodes", "set_time_steps"], time_steps=set_base_time_steps_yearly)
        self.raw_time_series["availability_carrier_import"] = self.data_input.extract_input_data("availability_carrier_import", index_sets=["set_nodes", "set_time_steps"], time_steps=set_base_time_steps_yearly)
        self.raw_time_series["availability_carrier_export"] = self.data_input.extract_input_data("availability_carrier_export", index_sets=["set_nodes", "set_time_steps"], time_steps=set_base_time_steps_yearly)
        self.raw_time_series["export_price_carrier"] = self.data_input.extract_input_data("export_price_carrier", index_sets=["set_nodes", "set_time_steps"], time_steps=set_base_time_steps_yearly)
        self.raw_time_series["import_price_carrier"] = self.data_input.extract_input_data("import_price_carrier", index_sets=["set_nodes", "set_time_steps"], time_steps=set_base_time_steps_yearly)
        # non-time series input data
        self.availability_carrier_import_yearly = self.data_input.extract_input_data("availability_carrier_import_yearly", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        self.availability_carrier_export_yearly = self.data_input.extract_input_data("availability_carrier_export_yearly", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        self.carbon_intensity_carrier = self.data_input.extract_input_data("carbon_intensity", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        self.shed_demand_price = self.data_input.extract_input_data("shed_demand_price", index_sets=[])

    def overwrite_time_steps(self, base_time_steps):
        """ overwrites set_time_steps_operation"""
        set_time_steps_operation = self.energy_system.time_steps.encode_time_step(self.name, base_time_steps=base_time_steps, time_step_type="operation", yearly=True)
        setattr(self, "set_time_steps_operation", set_time_steps_operation.squeeze().tolist())

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Carrier --- ###
    @classmethod
    def construct_sets(cls, optimization_setup):
        """ constructs the pe.Sets of the class <Carrier>
        :param optimization_setup: The OptimizationSetup the element is part of """
        pass

    @classmethod
    def construct_params(cls, optimization_setup):
        """ constructs the pe.Params of the class <Carrier>
        :param optimization_setup: The OptimizationSetup the element is part of """
        # demand of carrier
        optimization_setup.parameters.add_parameter(name="demand_carrier", data=optimization_setup.initialize_component(cls, "demand_carrier", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"]),
            doc='Parameter which specifies the carrier demand')
        # availability of carrier
        optimization_setup.parameters.add_parameter(name="availability_carrier_import",
            data=optimization_setup.initialize_component(cls, "availability_carrier_import", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"]),
            doc='Parameter which specifies the maximum energy that can be imported from outside the system boundaries')
        # availability of carrier
        optimization_setup.parameters.add_parameter(name="availability_carrier_export",
            data=optimization_setup.initialize_component(cls, "availability_carrier_export", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"]),
            doc='Parameter which specifies the maximum energy that can be exported to outside the system boundaries')
        # availability of carrier
        optimization_setup.parameters.add_parameter(name="availability_carrier_import_yearly",
            data=optimization_setup.initialize_component(cls, "availability_carrier_import_yearly", index_names=["set_carriers", "set_nodes", "set_time_steps_yearly"]),
            doc='Parameter which specifies the maximum energy that can be imported from outside the system boundaries for the entire year')
        # availability of carrier
        optimization_setup.parameters.add_parameter(name="availability_carrier_export_yearly",
            data=optimization_setup.initialize_component(cls, "availability_carrier_export_yearly", index_names=["set_carriers", "set_nodes", "set_time_steps_yearly"]),
            doc='Parameter which specifies the maximum energy that can be exported to outside the system boundaries for the entire year')
        # import price
        optimization_setup.parameters.add_parameter(name="import_price_carrier", data=optimization_setup.initialize_component(cls, "import_price_carrier", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"]),
            doc='Parameter which specifies the import carrier price')
        # export price
        optimization_setup.parameters.add_parameter(name="export_price_carrier", data=optimization_setup.initialize_component(cls, "export_price_carrier", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"]),
            doc='Parameter which specifies the export carrier price')
        # demand shedding price
        optimization_setup.parameters.add_parameter(name="shed_demand_price", data=optimization_setup.initialize_component(cls, "shed_demand_price", index_names=["set_carriers"]),
            doc='Parameter which specifies the price to shed demand')
        # carbon intensity
        optimization_setup.parameters.add_parameter(name="carbon_intensity_carrier",
            data=optimization_setup.initialize_component(cls, "carbon_intensity_carrier", index_names=["set_carriers", "set_nodes", "set_time_steps_yearly"]),
            doc='Parameter which specifies the carbon intensity of carrier')

    @classmethod
    def construct_vars(cls, optimization_setup):
        """ constructs the pe.Vars of the class <Carrier>
        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        variables = optimization_setup.variables
        sets = optimization_setup.sets

        # flow of imported carrier
        variables.add_variable(model, name="import_carrier_flow", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup), bounds=(0,np.inf),
                               doc="node- and time-dependent carrier import from the grid")
        # flow of exported carrier
        variables.add_variable(model, name="export_carrier_flow", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup), bounds=(0,np.inf),
                               doc="node- and time-dependent carrier export from the grid")
        # carrier import/export cost
        variables.add_variable(model, name="cost_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup),
                               doc="node- and time-dependent carrier cost due to import and export")
        # total carrier import/export cost
        variables.add_variable(model, name="cost_carrier_total", index_sets=sets["set_time_steps_yearly"],
                               doc="total carrier cost due to import and export")
        # carbon emissions
        variables.add_variable(model, name="carbon_emissions_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup),
                               doc="carbon emissions of importing and exporting carrier")
        # carbon emissions carrier
        variables.add_variable(model, name="carbon_emissions_carrier_total", index_sets=sets["set_time_steps_yearly"],
                               doc="total carbon emissions of importing and exporting carrier")
        # shed demand
        variables.add_variable(model, name="shed_demand_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup), bounds=(0,np.inf),
                               doc="shed demand of carrier")
        # cost of shed demand
        variables.add_variable(model, name="cost_shed_demand_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup), bounds=(0,np.inf),
                               doc="shed demand of carrier")

        # add pe.Sets of the child classes
        for subclass in cls.__subclasses__():
            if np.size(optimization_setup.system[subclass.label]):
                subclass.construct_vars(optimization_setup)

    @classmethod
    def construct_constraints(cls, optimization_setup):
        """ constructs the pe.Constraints of the class <Carrier>
        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        constraints = optimization_setup.constraints
        sets = optimization_setup.sets
        rules = CarrierRules(optimization_setup)
        # limit import flow by availability
        t0 = time.perf_counter()
        constraints.add_constraint_block(model, name="constraint_availability_carrier_import",
                                         constraint=rules.get_constraint_availability_carrier_import(),
                                         doc='node- and time-dependent carrier availability to import from outside the system boundaries', )
        t1 = time.perf_counter()
        logging.debug(f"Carrier: constraint_availability_carrier_import took {t1 - t0:.4f} seconds")
        # limit export flow by availability
        constraints.add_constraint_block(model, name="constraint_availability_carrier_export",
                                         constraint=rules.get_constraint_availability_carrier_export(),
                                         doc='node- and time-dependent carrier availability to export to outside the system boundaries')
        t2 = time.perf_counter()
        logging.debug(f"Carrier: constraint_availability_carrier_export took {t2 - t1:.4f} seconds")
        # limit import flow by availability for each year
        constraints.add_constraint_rule(model, name="constraint_availability_carrier_import_yearly",
                                        index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_yearly"], optimization_setup),
                                        rule=rules.constraint_availability_carrier_import_yearly_rule,
                                        doc='node- and time-dependent carrier availability to import from outside the system boundaries summed over entire year', )
        t3 = time.perf_counter()
        logging.debug(f"Carrier: constraint_availability_carrier_import_yearly took {t3 - t2:.4f} seconds")
        # limit export flow by availability for each year
        constraints.add_constraint_rule(model, name="constraint_availability_carrier_export_yearly",
                                        index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_yearly"], optimization_setup),
                                        rule=rules.constraint_availability_carrier_export_yearly_rule,
                                        doc='node- and time-dependent carrier availability to export to outside the system boundaries summed over entire year', )
        t4 = time.perf_counter()
        logging.debug(f"Carrier: constraint_availability_carrier_export_yearly took {t4 - t3:.4f} seconds")
        # cost for carrier
        constraints.add_constraint_block(model, name="constraint_cost_carrier",
                                         constraint=rules.get_constraint_cost_carrier(),
                                         doc="cost of importing and exporting carrier")
        t5 = time.perf_counter()
        logging.debug(f"Carrier: constraint_cost_carrier took {t5 - t4:.4f} seconds")
        # cost for shed demand
        constraints.add_constraint_block(model, name="constraint_cost_shed_demand",
                                         constraint=rules.get_constraint_cost_shed_demand(),
                                         doc="cost of shedding carrier demand")
        t6 = time.perf_counter()
        logging.debug(f"Carrier: constraint_cost_shed_demand took {t6 - t5:.4f} seconds")
        # limit of shed demand
        constraints.add_constraint_block(model, name="constraint_limit_shed_demand",
                                         constraint=rules.get_constraint_limit_shed_demand(),
                                         doc="limit of shedding carrier demand")
        t7 = time.perf_counter()
        logging.debug(f"Carrier: constraint_limit_shed_demand took {t7 - t6:.4f} seconds")
        # total cost for carriers
        constraints.add_constraint_rule(model, name="constraint_cost_carrier_total",
                                        index_sets=sets["set_time_steps_yearly"], rule=rules.constraint_cost_carrier_total_rule,
                                        doc="total cost of importing and exporting carriers")
        t8 = time.perf_counter()
        logging.debug(f"Carrier: constraint_cost_carrier_total took {t8 - t7:.4f} seconds")
        # carbon emissions
        constraints.add_constraint_block(model, name="constraint_carbon_emissions_carrier",
                                         constraint=rules.get_constraint_carbon_emissions_carrier(),
                                         doc="carbon emissions of importing and exporting carrier")
        t9 = time.perf_counter()
        logging.debug(f"Carrier: constraint_carbon_emissions_carrier took {t9 - t8:.4f} seconds")
        # carbon emissions carrier
        constraints.add_constraint_rule(model, name="constraint_carbon_emissions_carrier_total",
                                        index_sets=sets["set_time_steps_yearly"], rule=rules.constraint_carbon_emissions_carrier_total_rule,
            doc="total carbon emissions of importing and exporting carriers")
        t10 = time.perf_counter()
        logging.debug(f"Carrier: constraint_carbon_emissions_carrier_total took {t10 - t9:.4f} seconds")
        # energy balance
        constraints.add_constraint_block(model, name="constraint_nodal_energy_balance",
                                         constraint=rules.get_constraint_nodal_energy_balance(),
                                         doc='node- and time-dependent energy balance for each carrier', )
        t11 = time.perf_counter()
        logging.debug(f"Carrier: constraint_nodal_energy_balance took {t11 - t10:.4f} seconds")
        # add pe.Sets of the child classes
        for subclass in cls.__subclasses__():
            if len(optimization_setup.system[subclass.label]) > 0:
                subclass.construct_constraints(optimization_setup)


class CarrierRules:
    """
    Rules for the Carrier class
    """

    def __init__(self, optimization_setup):
        """
        Inits the rules for a given EnergySystem
        :param optimization_setup: The OptimizationSetup the element is part of
        """

        self.optimization_setup = optimization_setup
        self.energy_system = optimization_setup.energy_system

    # %% Constraint rules defined in current class
    def get_constraint_availability_carrier_import(self):
        """node- and time-dependent carrier availability to import from outside the system boundaries"""
        # get parameter object
        params = self.optimization_setup.parameters
        model = self.optimization_setup.model

        return (model.variables["import_carrier_flow"]
                <= params.availability_carrier_import)

    def get_constraint_availability_carrier_export(self):
        """node- and time-dependent carrier availability to export to outside the system boundaries"""
        # get parameter object
        params = self.optimization_setup.parameters
        model = self.optimization_setup.model

        return (model.variables["export_carrier_flow"]
                <= params.availability_carrier_export)

    def constraint_availability_carrier_import_yearly_rule(self, carrier, node, year):
        """node- and year-dependent carrier availability to import from outside the system boundaries"""
        # get parameter object
        params = self.optimization_setup.parameters
        model = self.optimization_setup.model

        if params.availability_carrier_import_yearly.loc[carrier, node, year] != np.inf:
            operational_time_steps = self.energy_system.time_steps.get_time_steps_year2operation(carrier, year)
            return ((model.variables["import_carrier_flow"][carrier, node, operational_time_steps] * params.time_steps_operation_duration.loc[carrier, operational_time_steps]).sum()
                    <= params.availability_carrier_import_yearly.loc[carrier, node, year].item())
        else:
            return None

    def constraint_availability_carrier_export_yearly_rule(self, carrier, node, year):
        """node- and year-dependent carrier availability to export to outside the system boundaries"""
        # get parameter object
        params = self.optimization_setup.parameters
        model = self.optimization_setup.model
        operational_time_steps = self.energy_system.time_steps.get_time_steps_year2operation(carrier, year)
        if params.availability_carrier_export_yearly.loc[carrier, node, year] != np.inf:
            return ((model.variables["export_carrier_flow"][carrier, node, operational_time_steps] * params.time_steps_operation_duration.loc[carrier, operational_time_steps]).sum()
                    <= params.availability_carrier_export_yearly.loc[carrier, node, year].item())
        else:
            return None

    def get_constraint_cost_carrier(self):
        """ cost of importing and exporting carrier"""
        # get parameter object
        params = self.optimization_setup.parameters
        model = self.optimization_setup.model

        # normal tuple constraints
        mask = ((params.availability_carrier_import != 0) | (params.availability_carrier_export != 0))
        mask = xr.align(model.variables.labels, mask)[1]
        tuples = [(1.0, model.variables["cost_carrier"]),
                  (-params.import_price_carrier.where(mask), model.variables["import_carrier_flow"].where(mask)),
                  (params.export_price_carrier.where(mask), model.variables["export_carrier_flow"].where(mask))]
        return model.linexpr(*tuples) == 0

    def get_constraint_cost_shed_demand(self):
        """ cost of shedding demand of carrier """
        # get parameter object
        params = self.optimization_setup.parameters
        model = self.optimization_setup.model

        # get all the constraints
        constraints = []
        index_values, index_names = Element.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)
        for carrier in index.get_unique([0]):
            if params.shed_demand_price.loc[carrier] != np.inf:
                constraints.append((model.variables["cost_shed_demand_carrier"].loc[carrier]
                                    - model.variables["shed_demand_carrier"].loc[carrier] * params.shed_demand_price.loc[carrier]
                                    == 0))
            else:
                constraints.append((model.variables["shed_demand_carrier"].loc[carrier]
                                    == 0))
        return self.optimization_setup.constraints.reorder_list(constraints, index.get_unique([0]), index_names[:1], model)

    def get_constraint_limit_shed_demand(self):
        """ limit demand shedding at low price """
        # get parameter object
        params = self.optimization_setup.parameters
        model = self.optimization_setup.model

        return (model.variables["shed_demand_carrier"]
                <= params.demand_carrier)

    def constraint_cost_carrier_total_rule(self, year):
        """ total cost of importing and exporting carrier"""
        # get parameter object
        params = self.optimization_setup.parameters
        sets = self.optimization_setup.sets
        model = self.optimization_setup.model

        terms = []
        # This vectorizes over times and locations
        for carrier in sets["set_carriers"]:
            times = self.energy_system.time_steps.get_time_steps_year2operation(carrier, year)
            expr = (model.variables["cost_carrier"].loc[carrier, :, times] + model.variables["cost_shed_demand_carrier"].loc[carrier, :, times]) * params.time_steps_operation_duration.loc[carrier, times]
            terms.append(expr.sum())

        return (model.variables["cost_carrier_total"].loc[year]
                - lp_sum(terms)
                == 0)

    def get_constraint_carbon_emissions_carrier(self):
        """ carbon emissions of importing and exporting carrier"""
        # get parameter object
        params = self.optimization_setup.parameters
        model = self.optimization_setup.model

        # get all the constraints
        constraints = []
        index_values, index_names = Element.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)
        times = index.get_unique([2])
        for carrier in index.get_unique([0]):
            yearly_time_steps = [self.energy_system.time_steps.convert_time_step_operation2year(carrier, t) for t in times]

            # get the time-dependent factor
            mask = (params.availability_carrier_import.loc[carrier, :, times] != 0) | (params.availability_carrier_export.loc[carrier, :, times] != 0)
            fac = np.where(mask, params.carbon_intensity_carrier.loc[carrier, :, yearly_time_steps].data, 0.0)
            fac = xr.DataArray(fac, coords=[model.variables.coords["set_nodes"], model.variables.coords["set_time_steps_operation"]])
            constraints.append(model.variables["carbon_emissions_carrier"].loc[carrier, :]
                               - fac * (model.variables["import_carrier_flow"].loc[carrier, :] - model.variables["export_carrier_flow"].loc[carrier, :])
                               == 0)

        return self.optimization_setup.constraints.reorder_list(constraints, index.get_unique([0]), index_names[:1], model)

    def constraint_carbon_emissions_carrier_total_rule(self, year):
        """ total carbon emissions of importing and exporting carrier"""
        # get parameter object
        params = self.optimization_setup.parameters
        sets = self.optimization_setup.sets
        model = self.optimization_setup.model

        # calculate the sums via loop
        terms = []
        for carrier in sets["set_carriers"]:
            time_arr = xr.DataArray(self.energy_system.time_steps.get_time_steps_year2operation(carrier, year),
                                    dims=["set_time_steps_operation"])
            expr = model.variables["carbon_emissions_carrier"].loc[carrier, :, time_arr] * params.time_steps_operation_duration.loc[carrier, time_arr]
            terms.append(expr.sum())

        return (model.variables["carbon_emissions_carrier_total"].loc[year]
                - lp_sum(terms)
                == 0)

    def get_constraint_nodal_energy_balance(self):
        """
        nodal energy balance for each time step.
        """
        # get parameter object
        params = self.optimization_setup.parameters
        sets = self.optimization_setup.sets
        model = self.optimization_setup.model

        # get the index
        index_values, index_names = Carrier.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        # carrier flow transport technologies
        if model.variables["carrier_flow"].size > 0:
            carrier_flow_in = []
            carrier_flow_out = []
            # precalculate the edges
            edges_in = {node: self.energy_system.calculate_connected_edges(node, "in") for node in index.get_unique([1])}
            edges_out = {node: self.energy_system.calculate_connected_edges(node, "out") for node in index.get_unique([1])}
            # loop over all nodes and carriers
            for carrier, node in index.get_unique([0, 1]):
                techs = []
                for tech in sets["set_transport_technologies"]:
                    if carrier in sets["set_reference_carriers"][tech]:
                        techs.append(tech)
                carrier_flow_in.append((model.variables["carrier_flow"].loc[techs, edges_in[node]] - model.variables["carrier_loss"].loc[techs, edges_in[node]]).sum(["set_transport_technologies", "set_edges"]))
                carrier_flow_out.append(model.variables["carrier_flow"].loc[techs, edges_out[node]].sum(["set_transport_technologies", "set_edges"]))
            # merge and regroup
            carrier_flow_in = lp.merge(*carrier_flow_in, dim="group")
            carrier_flow_in = self.optimization_setup.constraints.reorder_group(carrier_flow_in, None, None, index.get_unique([0, 1]), index_names[:-1], model)
            carrier_flow_out = lp.merge(*carrier_flow_out, dim="group")
            carrier_flow_out = self.optimization_setup.constraints.reorder_group(carrier_flow_out, None, None, index.get_unique([0, 1]), index_names[:-1], model)
        else:
            # if there is no carrier flow we just create empty arrays
            carrier_flow_in = model.variables["import_carrier_flow"].where(False).to_linexpr()
            carrier_flow_out = model.variables["import_carrier_flow"].where(False).to_linexpr()

        # carrier flow transport technologies
        carrier_conversion_in = []
        carrier_conversion_out = []
        nodes = list(sets["set_nodes"])
        for carrier in index.get_unique([0]):
            techs_in = [tech for tech in sets["set_conversion_technologies"] if carrier in sets["set_input_carriers"][tech]]
            # we need to catch emtpy lookups
            if len(techs_in) == 0:
                carrier_in = []
            else:
                carrier_in = [carrier]
            techs_out = [tech for tech in sets["set_conversion_technologies"] if carrier in sets["set_output_carriers"][tech]]
            # we need to catch emtpy lookups
            if len(techs_out) == 0:
                carrier_out = []
            else:
                carrier_out = [carrier]
            carrier_conversion_in.append(model.variables["input_flow"].loc[techs_in, carrier_in, nodes].sum(model.variables["input_flow"].dims[:2]))
            carrier_conversion_out.append(model.variables["output_flow"].loc[techs_out, carrier_out, nodes].sum(model.variables["output_flow"].dims[:2]))
        # merge and regroup
        carrier_conversion_in = lp.merge(*carrier_conversion_in, dim="group")
        carrier_conversion_in = self.optimization_setup.constraints.reorder_group(carrier_conversion_in, None, None, index.get_unique([0]), index_names[:1], model)
        carrier_conversion_out = lp.merge(*carrier_conversion_out, dim="group")
        carrier_conversion_out = self.optimization_setup.constraints.reorder_group(carrier_conversion_out, None, None, index.get_unique([0]), index_names[:1], model)

        # carrier flow storage technologies
        if model.variables["carrier_flow_charge"].size > 0:
            carrier_flow_charge = []
            carrier_flow_discharge = []
            for carrier in index.get_unique([0]):
                storage_techs = [tech for tech in sets["set_storage_technologies"] if carrier in sets["set_reference_carriers"][tech]]
                carrier_flow_charge.append(model.variables["carrier_flow_charge"].loc[storage_techs].sum("set_storage_technologies"))
                carrier_flow_discharge.append(model.variables["carrier_flow_discharge"].loc[storage_techs].sum("set_storage_technologies"))
            # merge and regroup
            carrier_flow_charge = lp.merge(*carrier_flow_charge, dim="group")
            carrier_flow_charge = self.optimization_setup.constraints.reorder_group(carrier_flow_charge, None, None, index.get_unique([0]), index_names[:1], model)
            carrier_flow_discharge = lp.merge(*carrier_flow_discharge, dim="group")
            carrier_flow_discharge = self.optimization_setup.constraints.reorder_group(carrier_flow_discharge, None, None, index.get_unique([0]), index_names[:1], model)
        else:
            # if there is no carrier flow we just create empty arrays
            carrier_flow_charge = model.variables["import_carrier_flow"].where(False).to_linexpr()
            carrier_flow_discharge = model.variables["import_carrier_flow"].where(False).to_linexpr()

        # carrier import, demand and export
        carrier_import = model.variables["import_carrier_flow"].to_linexpr()
        carrier_export = model.variables["export_carrier_flow"].to_linexpr()
        carrier_demand = params.demand_carrier
        # shed demand
        carrier_shed_demand = model.variables["shed_demand_carrier"].to_linexpr()

        # Add everything
        lhs = lp.merge(carrier_conversion_out,
                       -carrier_conversion_in,
                       carrier_flow_in,
                       -carrier_flow_out,
                       carrier_flow_discharge,
                       -carrier_flow_charge,
                       carrier_import,
                       -carrier_export,
                       carrier_shed_demand)

        return lhs == carrier_demand
