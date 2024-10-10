"""
Functions to apply time series aggregation to time series
"""
import copy

import pandas as pd
import numpy as np
import logging
import tsam.timeseriesaggregation as tsam
from zen_garden.model.objects.energy_system import EnergySystem
from zen_garden.model.objects.element import Element
from zen_garden.model.objects.technology.technology import Technology
from zen_garden.model.objects.technology.storage_technology import StorageTechnology


class TimeSeriesAggregation(object):
    """
    Class containing methods to apply time series aggregation
    """
    def __init__(self, energy_system: EnergySystem):
        """ initializes the time series aggregation. The data is aggregated for a single year and then concatenated

        :param energy_system: The energy system to use"""
        logging.info("\n--- Time series aggregation ---")
        self.energy_system = energy_system
        self.time_steps = self.energy_system.time_steps
        self.optimization_setup = energy_system.optimization_setup
        self.system = self.optimization_setup.system
        self.analysis = self.optimization_setup.analysis
        self.header_set_time_steps = self.analysis['header_data_inputs']["set_time_steps"]
        # if set_time_steps as input (because already aggregated), use this as base time step, otherwise self.set_base_time_steps
        self.set_base_time_steps = self.energy_system.set_base_time_steps_yearly
        self.number_typical_periods = min(self.system["unaggregated_time_steps_per_year"], self.system["aggregated_time_steps_per_year"])
        self.conducted_tsa = False
        self.get_excluded_ts()
        # if number of time steps >= number of base time steps, skip aggregation
        if self.number_typical_periods < np.size(self.set_base_time_steps) and self.system["conduct_time_series_aggregation"]:
            # select time series
            self.select_ts_of_all_elements()
            if not self.df_ts_raw.empty:
                # run time series aggregation to create typical periods
                self.run_tsa()
            # nothing to aggregate
            else:
                assert len(self.excluded_ts) == 0, "Do not exclude any time series from aggregation, if there is then nothing else to aggregate!"
                # aggregate to single time step
                self.single_ts_tsa()
        else:
            self.typical_periods = pd.DataFrame()
            set_time_steps = self.set_base_time_steps
            time_step_duration = self.energy_system.time_steps.calculate_time_step_duration(set_time_steps, self.set_base_time_steps)
            sequence_time_steps = np.concatenate([[time_step] * time_step_duration[time_step] for time_step in time_step_duration])
            self.set_time_attributes(set_time_steps, time_step_duration, sequence_time_steps)
            # set aggregated time series
            self.set_aggregated_ts_all_elements()
        # set aggregated time steps to time step object
        self.time_steps.set_aggregated_time_steps(self)
        # repeat order of operational time steps and link with investment and yearly time steps
        self.repeat_sequence_time_steps_for_all_years()
        logging.info("Calculate operational time steps for storage levels")
        self.calculate_time_steps_storage_level()
        # overwrite duration of operational and storage time steps in energy system
        self.energy_system.time_steps_operation_duration = pd.Series(self.time_steps.time_steps_operation_duration)
        self.energy_system.time_steps_storage_duration = pd.Series(self.time_steps.time_steps_storage_duration)

    def select_ts_of_all_elements(self):
        """ this method retrieves the raw time series for the aggregation of all input data sets. """
        dict_raw_ts = {}
        for element in self.optimization_setup.get_all_elements(Element):
            df_ts_raw = self.extract_raw_ts(element, self.header_set_time_steps)
            if not df_ts_raw.empty:
                dict_raw_ts[element.name] = df_ts_raw
        if dict_raw_ts:
            self.df_ts_raw = pd.concat(dict_raw_ts.values(), axis=1, keys=dict_raw_ts.keys())
        else:
            self.df_ts_raw = pd.Series()

    def substitute_column_names(self, direction="flatten"):
        """ this method substitutes the column names to have flat columns names (otherwise sklearn warning)

        :param direction: flatten or raise
        """
        if direction == "flatten":
            if not hasattr(self, "column_names_original"):
                self.column_names_original = self.df_ts_raw.columns
                self.column_names_flat = [str(index) for index in self.column_names_original]
                self.df_ts_raw.columns = self.column_names_flat
        elif direction == "raise":
            self.typical_periods = self.typical_periods[self.column_names_flat]
            self.typical_periods.columns = self.column_names_original

    def run_tsa(self):
        """ this method runs the time series aggregation """
        # substitute column names
        self.substitute_column_names(direction="flatten")
        # create aggregation object
        self.aggregation = tsam.TimeSeriesAggregation(timeSeries=self.df_ts_raw, noTypicalPeriods=self.number_typical_periods,
            hoursPerPeriod=self.analysis["time_series_aggregation"]["hoursPerPeriod"], resolution=self.analysis["time_series_aggregation"]["resolution"],
            clusterMethod=self.analysis["time_series_aggregation"]["clusterMethod"], solver=self.analysis["time_series_aggregation"]["solver"],
            extremePeriodMethod=self.analysis["time_series_aggregation"]["extremePeriodMethod"], rescaleClusterPeriods=self.analysis["time_series_aggregation"]["rescaleClusterPeriods"],
            representationMethod=self.analysis["time_series_aggregation"]["representationMethod"])
        # create typical periods
        self.typical_periods = self.aggregation.createTypicalPeriods().reset_index(drop=True)
        self.set_time_attributes(self.aggregation.clusterPeriodIdx, self.aggregation.clusterPeriodNoOccur, self.aggregation.clusterOrder)
        # resubstitute column names
        self.substitute_column_names(direction="raise")
        # set aggregated time series
        self.set_aggregated_ts_all_elements()
        self.conducted_tsa = True

    def set_aggregated_ts_all_elements(self):
        """ this method sets the aggregated time series and sets the necessary attributes after the aggregation """
        # sort typical periods to avoid indexing past lexsort depth
        self.typical_periods = self.typical_periods.sort_index(axis=1)
        # sets the aggregated time series of each element
        for element in self.optimization_setup.get_all_elements(Element):
            raw_ts = getattr(element, "raw_time_series")
            # iterate through raw time series
            for ts in raw_ts:
                if raw_ts[ts] is None:
                    continue
                index_names = list(raw_ts[ts].index.names)
                index_names.remove(self.header_set_time_steps)
                df_ts = raw_ts[ts].unstack(level=index_names)

                df_aggregated_ts = pd.DataFrame(index=self.set_time_steps, columns=df_ts.columns)
                # columns which are in aggregated time series and which are not
                if element.name in self.typical_periods and ts in self.typical_periods[element.name]:
                    df_typical_periods = self.typical_periods[element.name, ts]
                    aggregated_columns = df_ts.columns.intersection(df_typical_periods.columns)
                    not_aggregated_columns = df_ts.columns.difference(df_typical_periods.columns)
                    # aggregated columns
                    df_aggregated_ts[aggregated_columns] = self.typical_periods[element.name, ts][aggregated_columns]
                else:
                    not_aggregated_columns = df_ts.columns
                # not aggregated columns because excluded
                if (element.name, ts) in self.excluded_ts:
                    df_aggregated_ts = self.manually_aggregate_ts(df_ts)
                # not aggregated columns because constant
                else:
                    df_aggregated_ts[not_aggregated_columns] = df_ts.loc[df_aggregated_ts.index, not_aggregated_columns]
                # reorder
                df_aggregated_ts.index.names = [self.header_set_time_steps]
                df_aggregated_ts.columns.names = index_names
                df_aggregated_ts = df_aggregated_ts.stack(index_names,future_stack=True)
                df_aggregated_ts.index = df_aggregated_ts.index.reorder_levels(index_names + [self.header_set_time_steps])
                setattr(element, ts, df_aggregated_ts)
                element.aggregated = True
                # self.set_aggregation_indicators(element)

    def get_excluded_ts(self):
        """ gets the names of all elements and parameter ts that shall be excluded from the time series aggregation """
        self.excluded_ts = []
        if self.system["exclude_parameters_from_TSA"]:
            excluded_parameters = self.optimization_setup.energy_system.data_input.read_input_csv("exclude_parameter_from_TSA")
            # exclude file exists
            if excluded_parameters is not None:
                for _,vals in excluded_parameters.iterrows():
                    element_name = vals[0]
                    parameter = vals[1]
                    element = self.optimization_setup.get_element(cls=Element, name=element_name)
                    # specific element
                    if element is not None:
                        if parameter is np.nan:
                            logging.warning(f"Excluding all parameters {', '.join(element.raw_time_series.keys())} of {element_name} from time series aggregation")
                            for parameter_name in element.raw_time_series:
                                self.excluded_ts.append((element_name,parameter_name))
                        elif parameter in element.raw_time_series:
                            self.excluded_ts.append((element_name,parameter))
                    # for an entire set of elements
                    else:
                        if parameter is np.nan:
                            logging.warning("Please specify a specific parameter to exclude from time series aggregation when not providing a specific element")
                        else:
                            element_class = self.optimization_setup.get_element_class(name=element_name)
                            if element_class is not None:
                                logging.info(f"Parameter {parameter} is excluded from time series aggregation for all elements in {element_name}")
                                class_elements = self.optimization_setup.get_all_elements(cls=element_class)
                                for class_element in class_elements:
                                    if parameter in class_element.raw_time_series:
                                        self.excluded_ts.append((class_element.name, parameter))
                            else:
                                logging.warning(f"Exclusion from time series aggregation: {element_name} is neither a specific element nor an element class.")
            # remove duplicates
            self.excluded_ts = [*set(self.excluded_ts)]
            self.excluded_ts.sort()

    def manually_aggregate_ts(self,df):
        """ manually aggregates time series for excluded parameters.

        :param df: dataframe that is manually aggregated
        :return agg_df: aggregated dataframe """
        agg_df = pd.DataFrame(index=self.set_time_steps,columns=df.columns)
        tsa_options = self.analysis["time_series_aggregation"]
        if tsa_options["representationMethod"] == "meanRepresentation":
            representation_method = "mean"
        elif tsa_options["representationMethod"] == "mediodRepresentation":
            representation_method = "median"
        elif tsa_options["representationMethod"] is None:
            if tsa_options["clusterMethod"] == "k_means":
                representation_method = "mean"
            elif tsa_options["clusterMethod"] == "k_medoids" or tsa_options["clusterMethod"] == "hierarchical":
                representation_method = "median"
            else:
                raise NotImplementedError(
                    f"Default representation method not yet implemented for cluster method {tsa_options['clusterMethod']} manually aggregating excluded time series")
        else:
            raise NotImplementedError(
                f"Representation method {self.analysis['time_series_aggregation']['representationMethod']} not yet implemented for manually aggregating excluded time series")

        for time_step in self.set_time_steps:
            df_slice = df.loc[self.sequence_time_steps == time_step]
            if representation_method == "mean":
                agg_df.loc[time_step] = df_slice.mean(axis=0)
            else:
                agg_df.loc[time_step] = df_slice.median(axis=0)
        return agg_df.astype(float)

    def extract_raw_ts(self, element, header_set_time_steps):
        """ extract the time series from an element and concatenates the non-constant time series to a pd.DataFrame

        :param element: element of the optimization
        :param header_set_time_steps: name of set_time_steps
        :return df_ts_raw: pd.DataFrame with non-constant time series"""
        dict_raw_ts = {}
        raw_ts = getattr(element, "raw_time_series")
        for ts in raw_ts:
            if raw_ts[ts] is None:
                continue
            raw_ts[ts].name = ts
            df_ts = raw_ts[ts].unstack(level=header_set_time_steps).T
            # select time series that are not constant (rows have more than 1 unique entries)
            df_ts_non_constant = df_ts[df_ts.columns[df_ts.apply(lambda column: len(np.unique(column)) != 1)]]
            if (element.name,ts) in self.excluded_ts:
                df_empty = pd.DataFrame(index=df_ts_non_constant.index)
                dict_raw_ts[ts] = df_empty
            else:
                if isinstance(df_ts_non_constant.columns,pd.MultiIndex):
                    df_ts_non_constant.columns = df_ts_non_constant.columns.to_flat_index()
                dict_raw_ts[ts] = df_ts_non_constant
        df_ts_raw = pd.concat(dict_raw_ts.values(), axis=1, keys=dict_raw_ts.keys())
        return df_ts_raw

    def link_time_steps(self):
        """ calculates the necessary overlapping time steps of the investment and operation of a technology for all years.
        It sets the union of the time steps for investment, operation and years """
        list_sequence_time_steps = [self.time_steps.sequence_time_steps_yearly,
                                    self.time_steps.sequence_time_steps_operation]
        old_sequence_time_steps = copy.copy(self.time_steps.sequence_time_steps_operation)
        unique_time_steps_sequences = self.unique_time_steps_multiple_indices(list_sequence_time_steps)
        if unique_time_steps_sequences:
            set_time_steps, time_steps_duration, sequence_time_steps = unique_time_steps_sequences
            # set sequence time steps
            self.time_steps.time_steps_operation = set_time_steps
            self.time_steps.time_steps_operation_duration = time_steps_duration
            self.time_steps.sequence_time_steps_operation = sequence_time_steps
            # time series parameters
            for element in self.optimization_setup.get_all_elements(Element):
                self.overwrite_ts_with_expanded_timeindex(element, old_sequence_time_steps)

        else:
            for element in self.optimization_setup.get_all_elements(Element):
                # check to multiply the time series with the yearly variation
                self.yearly_variation_nonaggregated_ts(element)

    def overwrite_ts_with_expanded_timeindex(self, element, old_sequence_time_steps):
        """ this method expands the aggregated time series to match the extended operational time steps because of matching the investment and operational time sequences.

        :param element: element of the optimization
        :param old_sequence_time_steps: old order of operational time steps """
        header_set_time_steps = self.analysis['header_data_inputs']["set_time_steps"]
        idx_old2new = np.array([np.unique(old_sequence_time_steps[np.argwhere(idx == self.time_steps.sequence_time_steps_operation)]) for idx in self.time_steps.time_steps_operation]).squeeze()
        for ts in element.raw_time_series:
            if element.raw_time_series[ts] is not None:
                old_ts = getattr(element, ts).unstack(header_set_time_steps)
                new_ts = pd.DataFrame(index=old_ts.index, columns=self.time_steps.time_steps_operation)
                new_ts = old_ts.loc[:, idx_old2new[new_ts.columns]].T.reset_index(drop=True).T
                new_ts.columns.names = [header_set_time_steps]
                new_ts = new_ts.stack()
                # multiply with yearly variation
                new_ts = self.multiply_yearly_variation(element, ts, new_ts)
                # overwrite time series
                setattr(element, ts, new_ts)

    def yearly_variation_nonaggregated_ts(self, element):
        """ multiply the time series with the yearly variation if the element's time series are not aggregated

        :param element: element of the optimization """
        for ts in element.raw_time_series:
            if element.raw_time_series[ts] is None:
                continue
            # multiply with yearly variation
            new_ts = self.multiply_yearly_variation(element, ts, getattr(element, ts))
            # overwrite time series
            setattr(element, ts, new_ts)

    def multiply_yearly_variation(self, element, ts_name, ts):
        """ this method multiplies time series with the yearly variation of the time series
        The index of the variation is the same as the original time series, just time and year substituted

        :param element: technology of the optimization
        :param ts_name: name of time series
        :param ts: time series
        :return multipliedTimeSeries: ts multiplied with yearly variation """
        if hasattr(element.data_input, ts_name + "_yearly_variation"):
            yearly_variation = getattr(element.data_input, ts_name + "_yearly_variation")
            header_set_time_steps = self.analysis['header_data_inputs']["set_time_steps"]
            header_set_time_steps_yearly = self.analysis['header_data_inputs']["set_time_steps_yearly"]
            ts_df = ts.unstack(header_set_time_steps)
            yearly_variation = yearly_variation.unstack(header_set_time_steps_yearly)
            # if only one unique value
            if len(np.unique(yearly_variation)) == 1:
                ts = ts_df.stack() * np.unique(yearly_variation)[0]
            else:
                for year in self.energy_system.set_time_steps_yearly:
                    if not all(yearly_variation[year] == 1):
                        base_time_steps = self.energy_system.time_steps.decode_time_step(year, "yearly")
                        element_time_steps = self.energy_system.time_steps.encode_time_step(base_time_steps,time_step_type="operation")
                        ts_df.loc[:, element_time_steps] = ts_df[element_time_steps].multiply(yearly_variation[year], axis=0).fillna(0)
                ts = ts_df.stack()
        # round down if lower than decimal points
        if self.optimization_setup.solver["round_parameters"]:
            rounding_value = 10 ** (-self.optimization_setup.solver["rounding_decimal_points_tsa"])
            ts[ts.abs() < rounding_value] = 0
        return ts

    def repeat_sequence_time_steps_for_all_years(self):
        """ this method repeats the operational time series for all years."""
        logging.info("Repeat the time series sequences for all years")
        optimized_years = len(self.energy_system.set_time_steps_yearly)
        # concatenate the order of time steps and link with investment and yearly time steps
        old_sequence_time_steps = self.time_steps.sequence_time_steps_operation
        new_sequence_time_steps = np.hstack([old_sequence_time_steps] * optimized_years)
        self.time_steps.sequence_time_steps_operation = new_sequence_time_steps
        # calculate the time steps in operation to link with investment and yearly time steps
        self.link_time_steps()
        # set operation2year and year2operation time steps
        self.time_steps.set_time_steps_operation2year_both_dir()

    def calculate_time_steps_storage_level(self):
        """ this method calculates the number of time steps on the storage level, and the sequence in which the storage levels are connected
        """
        sequence_time_steps = self.time_steps.sequence_time_steps_operation
        # if time series aggregation was conducted
        if self.conducted_tsa:
            # calculate connected storage levels, i.e., time steps that are constant for
            idx_last_connected_storage_level = np.append(np.flatnonzero(np.diff(sequence_time_steps)), len(sequence_time_steps) - 1)
            time_steps_storage = []
            time_steps_storage_duration = {}
            time_steps_energy2power = {}
            sequence_time_steps_storage = np.zeros(np.size(sequence_time_steps)).astype(int)
            counter_time_step = 0
            for idx_time_step, idx_storage_level in enumerate(idx_last_connected_storage_level):
                time_steps_storage.append(idx_time_step)
                time_steps_storage_duration[idx_time_step] = len(range(counter_time_step, idx_storage_level + 1))
                sequence_time_steps_storage[counter_time_step:idx_storage_level + 1] = idx_time_step
                time_steps_energy2power[idx_time_step] = sequence_time_steps[idx_storage_level]
                counter_time_step = idx_storage_level + 1
        else:
            time_steps_storage = self.time_steps.time_steps_operation
            time_steps_storage_duration = self.time_steps.time_steps_operation_duration
            sequence_time_steps_storage = sequence_time_steps
            time_steps_energy2power = {idx: idx for idx in self.time_steps.time_steps_operation}
        # overwrite in time steps object
        self.time_steps.time_steps_storage = time_steps_storage
        self.time_steps.time_steps_storage_duration = time_steps_storage_duration
        self.time_steps.sequence_time_steps_storage = sequence_time_steps_storage
        # set the storage2year
        self.time_steps.set_time_steps_storage2year_both_dir()
        # set the dict time_steps_energy2power
        self.time_steps.time_steps_energy2power = time_steps_energy2power
        # set the first and last time step of each year
        self.time_steps.set_time_steps_storage_startend(self.optimization_setup.system)

    def unique_time_steps_multiple_indices(self, list_sequence_time_steps):
        """ this method returns the unique time steps of multiple time grids

        :param list_sequence_time_steps: list of operational and investment time steps
        :return (set_time_steps, time_steps_duration, sequence_time_steps): time steps, duration and sequence
        """
        sequence_time_steps = np.zeros(np.size(list_sequence_time_steps, axis=1)).astype(int)
        combined_sequence_time_steps = np.vstack(list_sequence_time_steps)
        unique_combined_time_steps, unique_indices, count_combined_time_steps = np.unique(combined_sequence_time_steps, axis=1, return_counts=True, return_index=True)
        # if unique yearly time steps (row 1) are the same as original, or if the operational time series (row 0) only has one unique time step
        if len(np.unique(combined_sequence_time_steps[1, :])) == len(combined_sequence_time_steps[1, :]) or len(np.unique(combined_sequence_time_steps[0, :])) == 1:
            return None
        set_time_steps = []
        time_steps_duration = {}
        for idx_unique_time_steps, count_unique_time_steps in enumerate(count_combined_time_steps):
            set_time_steps.append(idx_unique_time_steps)
            time_steps_duration[idx_unique_time_steps] = count_unique_time_steps
            unique_time_step = unique_combined_time_steps[:, idx_unique_time_steps]
            idx_input = np.argwhere(np.all(combined_sequence_time_steps.T == unique_time_step, axis=1))
            # fill new order time steps 
            sequence_time_steps[idx_input] = idx_unique_time_steps
        return (set_time_steps, time_steps_duration, sequence_time_steps)

    def single_ts_tsa(self):
        """ manually aggregates the constant time series to single ts """
        if self.number_typical_periods > 1:
            logging.warning("You are trying to aggregate constant time series to more than one representative time step. This setting is overwritten to one representative time step.")
            self.number_typical_periods = 1
        unaggregated_time_steps = self.system["unaggregated_time_steps_per_year"]
        set_time_steps = [0]
        time_steps_duration = {0:unaggregated_time_steps}
        sequence_time_steps = np.hstack(set_time_steps*unaggregated_time_steps)
        self.set_time_attributes(set_time_steps, time_steps_duration, sequence_time_steps)
        # create empty typical_period df
        self.typical_periods = pd.DataFrame(index=set_time_steps)
        # set aggregated time series
        self.set_aggregated_ts_all_elements()
        self.conducted_tsa = True

    def set_time_attributes(self,set_time_steps, time_steps_duration, sequence_time_steps):
        """ this method sets the operational time attributes of an element.

        :param set_time_steps: set_time_steps of operation
        :param time_steps_duration: time_steps_duration of operation
        :param sequence_time_steps: sequence of operation """
        self.set_time_steps = set_time_steps
        self.time_steps_duration = time_steps_duration
        self.sequence_time_steps = sequence_time_steps
