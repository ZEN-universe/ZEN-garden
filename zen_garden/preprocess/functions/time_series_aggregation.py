"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      January-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Functions to apply time series aggregation to time series
==========================================================================================================================================================================="""
import pandas as pd
import numpy as np
import logging
import tsam.timeseriesaggregation as tsam
from zen_garden.model.objects.energy_system import EnergySystem
from zen_garden.model.objects.element import Element
from zen_garden.model.objects.technology.technology import Technology
from zen_garden.model.objects.technology.storage_technology import StorageTechnology


class TimeSeriesAggregation(object):

    def __init__(self, energy_system: EnergySystem):
        """ initializes the time series aggregation. The data is aggregated for a single year and then concatenated
        :param energy_system: The energy system to use"""
        self.energy_system = energy_system
        self.system = energy_system.system
        self.analysis = energy_system.analysis
        self.header_set_time_steps = self.analysis['header_data_inputs']["set_time_steps"]
        # if set_time_steps as input (because already aggregated), use this as base time step, otherwise self.set_base_time_steps
        self.set_base_time_steps = self.energy_system.set_base_time_steps_yearly
        self.number_typical_periods = min(self.system["unaggregated_time_steps_per_year"], self.system["aggregated_time_steps_per_year"])
        self.conducted_tsa = False
        # if number of time steps >= number of base time steps, skip aggregation
        if self.number_typical_periods < np.size(self.set_base_time_steps) and self.system["conduct_time_series_aggregation"]:
            # select time series
            self.select_ts_of_all_elements()
            if not self.df_ts_raw.empty:
                # run time series aggregation to create typical periods
                self.run_tsa()
        else:
            self.typical_periods = pd.DataFrame()
            _set_time_steps = self.set_base_time_steps
            _time_step_duration = self.energy_system.calculate_time_step_duration(_set_time_steps, self.set_base_time_steps)
            _sequence_time_steps = np.concatenate([[time_step] * _time_step_duration[time_step] for time_step in _time_step_duration])
            self.set_time_attributes(self, _set_time_steps, _time_step_duration, _sequence_time_steps)
            # set aggregated time series
            self.set_aggregated_ts_all_elements()

    def select_ts_of_all_elements(self):
        """ this method retrieves the raw time series for the aggregation of all input data sets.
        Only in aligned time grid approach! """

        _dict_raw_ts = {}
        for element in self.energy_system.get_all_elements(Element):
            df_ts_raw = self.extract_raw_ts(element, self.header_set_time_steps)
            if not df_ts_raw.empty:
                _dict_raw_ts[element.name] = df_ts_raw
        if _dict_raw_ts:
            self.df_ts_raw = pd.concat(_dict_raw_ts.values(), axis=1, keys=_dict_raw_ts.keys())
        else:
            self.df_ts_raw = pd.Series()

    def substitute_column_names(self, direction="flatten"):
        """ this method substitutes the column names to have flat columns names (otherwise sklearn warning) """
        if direction == "flatten":
            if not hasattr(self, "columnNamesOriginal"):
                self.columnNamesOriginal = self.df_ts_raw.columns
                self.columnNamesFlat = [str(index) for index in self.columnNamesOriginal]
                self.df_ts_raw.columns = self.columnNamesFlat
        elif direction == "raise":
            self.typical_periods = self.typical_periods[self.columnNamesFlat]
            self.typical_periods.columns = self.columnNamesOriginal

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
        self.set_time_attributes(self, self.aggregation.clusterPeriodIdx, self.aggregation.clusterPeriodNoOccur, self.aggregation.clusterOrder)
        # resubstitute column names
        self.substitute_column_names(direction="raise")
        # set aggregated time series
        self.set_aggregated_ts_all_elements()
        self.conducted_tsa = True

    def set_aggregated_ts_all_elements(self):
        """ this method sets the aggregated time series and sets the necessary attributes after the aggregation to a single time grid """
        for element in self.energy_system.get_all_elements(Element):
            raw_ts = getattr(element, "raw_time_series")
            # set_time_steps, duration and sequence time steps
            element.set_time_steps_operation = list(self.set_time_steps)
            element.time_steps_operation_duration = self.time_steps_duration
            element.sequence_time_steps = self.sequence_time_steps

            # iterate through raw time series
            for ts in raw_ts:
                _index_names = list(raw_ts[ts].index.names)
                _index_names.remove(self.header_set_time_steps)
                df_ts = raw_ts[ts].unstack(level=_index_names)

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
                # not aggregated columns
                df_aggregated_ts[not_aggregated_columns] = df_ts.loc[df_aggregated_ts.index, not_aggregated_columns]
                # reorder
                df_aggregated_ts.index.names = [self.header_set_time_steps]
                df_aggregated_ts.columns.names = _index_names
                df_aggregated_ts = df_aggregated_ts.stack(_index_names)
                df_aggregated_ts.index = df_aggregated_ts.index.reorder_levels(_index_names + [self.header_set_time_steps])
                setattr(element, ts, df_aggregated_ts)
                TimeSeriesAggregation.set_aggregation_indicators(element)

    def extract_raw_ts(self, element, header_set_time_steps):
        """ extract the time series from an element and concatenates the non-constant time series to a pd.DataFrame
        :param element: element of the optimization 
        :param header_set_time_steps: name of set_time_steps
        :return df_time_series_raw: pd.DataFrame with non-constant time series"""
        _dict_raw_ts = {}
        raw_ts = getattr(element, "raw_time_series")
        for ts in raw_ts:
            raw_ts[ts].name = ts
            _index_names = list(raw_ts[ts].index.names)
            _index_names.remove(header_set_time_steps)
            df_ts = raw_ts[ts].unstack(level=_index_names)
            # select time series that are not constant (rows have more than 1 unique entries)
            df_ts_non_constant = df_ts[df_ts.columns[df_ts.apply(lambda column: len(np.unique(column)) != 1)]]
            _dict_raw_ts[ts] = df_ts_non_constant
        df_ts_raw = pd.concat(_dict_raw_ts.values(), axis=1, keys=_dict_raw_ts.keys())
        return df_ts_raw

    def link_time_steps(self, element):
        """ calculates the necessary overlapping time steps of the investment and operation of a technology for all years.
        It sets the union of the time steps for investment, operation and years.
        :param element: technology of the optimization """
        list_sequence_time_steps = [self.energy_system.get_sequence_time_steps(element.name, "operation"),
                                    self.energy_system.get_sequence_time_steps(None, "yearly")]

        unique_time_steps_sequences = self.unique_time_steps_multiple_indices(list_sequence_time_steps)
        if unique_time_steps_sequences:
            set_time_steps, time_steps_duration, sequence_time_steps = unique_time_steps_sequences
            # set sequence time steps
            self.energy_system.set_sequence_time_steps(element.name, sequence_time_steps)
            # time series parameters
            self.overwrite_ts_with_expanded_timeindex(element, set_time_steps, sequence_time_steps)
            # set attributes
            self.set_time_attributes(element, set_time_steps, time_steps_duration, sequence_time_steps)
        else:
            # check to multiply the time series with the yearly variation
            self.yearly_variation_nonaggregated_ts(element)

    def calculate_time_steps_operation2invest(self, element):
        """ calculates the conversion of operational time steps to invest time steps """
        _sequence_time_steps_operation = getattr(element, "sequence_time_steps")
        _sequence_time_steps_yearly = getattr(self.energy_system, "sequence_time_steps_yearly")
        time_steps_operation2invest = np.unique(np.vstack([_sequence_time_steps_operation, _sequence_time_steps_yearly]), axis=1)
        time_steps_operation2invest = {key: val for key, val in zip(time_steps_operation2invest[0, :], time_steps_operation2invest[1, :])}
        self.energy_system.set_time_steps_operation2invest(element.name, time_steps_operation2invest)

    def overwrite_ts_with_expanded_timeindex(self, element, set_time_steps_operation, sequence_time_steps):
        """ this method expands the aggregated time series to match the extended operational time steps because of matching the investment and operational time sequences.
        :param element: element of the optimization
        :param set_time_steps_operation: new time steps operation
        :param sequence_time_steps: new order of operational time steps """
        header_set_time_steps = self.analysis['header_data_inputs']["set_time_steps"]
        old_sequence_time_steps = element.sequence_time_steps
        _idx_old2new = np.array([np.unique(old_sequence_time_steps[np.argwhere(idx == sequence_time_steps)]) for idx in set_time_steps_operation]).squeeze()
        for ts in element.raw_time_series:
            _old_ts = getattr(element, ts).unstack(header_set_time_steps)
            _new_ts = pd.DataFrame(index=_old_ts.index, columns=set_time_steps_operation)
            _new_ts = _old_ts.loc[:, _idx_old2new[_new_ts.columns]].T.reset_index(drop=True).T
            _new_ts.columns.names = [header_set_time_steps]
            _new_ts = _new_ts.stack()
            # multiply with yearly variation
            _new_ts = self.multiply_yearly_variation(element, ts, _new_ts)
            # overwrite time series
            setattr(element, ts, _new_ts)

    def yearly_variation_nonaggregated_ts(self, element):
        """ multiply the time series with the yearly variation if the element's time series are not aggregated
        :param element: element of the optimization """
        for ts in element.raw_time_series:
            # multiply with yearly variation
            _new_ts = self.multiply_yearly_variation(element, ts, getattr(element, ts))
            # overwrite time series
            setattr(element, ts, _new_ts)

    def multiply_yearly_variation(self, element, ts_name, ts):
        """ this method multiplies time series with the yearly variation of the time series
        The index of the variation is the same as the original time series, just time and year substituted
        :param element: technology of the optimization
        :param ts_name: name of time series
        :param ts: time series
        :return multipliedTimeSeries: ts multiplied with yearly variation """
        if hasattr(element.data_input, ts_name + "_yearly_variation"):
            _yearly_variation = getattr(element.data_input, ts_name + "_yearly_variation")
            header_set_time_steps = self.analysis['header_data_inputs']["set_time_steps"]
            header_set_time_steps_yearly = self.analysis['header_data_inputs']["set_time_steps_yearly"]
            _ts = ts.unstack(header_set_time_steps)
            _yearly_variation = _yearly_variation.unstack(header_set_time_steps_yearly)
            # if only one unique value
            if len(np.unique(_yearly_variation)) == 1:
                ts = _ts.stack() * np.unique(_yearly_variation)[0]
            else:
                for year in self.energy_system.set_time_steps_yearly:
                    if not all(_yearly_variation[year] == 1):
                        _base_time_steps = self.energy_system.decode_time_step(None, year, "yearly")
                        _element_time_steps = self.energy_system.encode_time_step(element.name, _base_time_steps, yearly=True)
                        _ts.loc[:, _element_time_steps] = _ts[_element_time_steps].multiply(_yearly_variation[year], axis=0).fillna(0)
                ts = _ts.stack()
        # round down if lower than decimal points
        _rounding_value = 10 ** (-self.energy_system.solver["rounding_decimal_points_ts"])
        ts[ts.abs() < _rounding_value] = 0
        return ts

    def repeat_sequence_time_steps_for_all_years(self):
        """ this method repeats the operational time series for all years."""
        logging.info("Repeat the time series sequences for all years")
        optimized_years = len(self.energy_system.set_time_steps_yearly)
        # concatenate the order of time steps for all elements and link with investment and yearly time steps
        for element in self.energy_system.get_all_elements(Element):
            # optimized_years = EnergySystem.get_system()["optimized_years"]
            old_sequence_time_steps = self.energy_system.get_attribute_of_specific_element(Element, element.name, "sequence_time_steps")
            new_sequence_time_steps = np.hstack([old_sequence_time_steps] * optimized_years)
            element.sequence_time_steps = new_sequence_time_steps
            self.energy_system.set_sequence_time_steps(element.name, element.sequence_time_steps)
            # calculate the time steps in operation to link with investment and yearly time steps
            self.link_time_steps(element)
            # set operation2invest time step dict
            if element in self.energy_system.get_all_elements(Technology):
                self.calculate_time_steps_operation2invest(element)

    def set_aggregation_indicators(self, element):
        """ this method sets the indicators that element is aggregated """
        # add order of time steps to Energy System
        self.energy_system.set_sequence_time_steps(element.name, element.sequence_time_steps, time_step_type="operation")
        element.aggregated = True

    def unique_time_steps_multiple_indices(self, list_sequence_time_steps):
        """ this method returns the unique time steps of multiple time grids """
        sequence_time_steps = np.zeros(np.size(list_sequence_time_steps, axis=1)).astype(int)
        combined_sequence_time_steps = np.vstack(list_sequence_time_steps)
        unique_combined_time_steps, unique_indices, count_combined_time_steps = np.unique(combined_sequence_time_steps, axis=1, return_counts=True, return_index=True)
        # if unique time steps are the same as original, or if the second until last only have a single unique value
        if len(np.unique(combined_sequence_time_steps[0, :])) == len(combined_sequence_time_steps[0, :]) or len(np.unique(combined_sequence_time_steps[1:, :], axis=1)[0]) == 1:
            return None
        set_time_steps = []
        time_steps_duration = {}
        for _idx_unique_time_steps, _count_unique_time_steps in enumerate(count_combined_time_steps):
            set_time_steps.append(_idx_unique_time_steps)
            time_steps_duration[_idx_unique_time_steps] = _count_unique_time_steps
            _unique_time_step = unique_combined_time_steps[:, _idx_unique_time_steps]
            _idx_input = np.argwhere(np.all(combined_sequence_time_steps.T == _unique_time_step, axis=1))
            # fill new order time steps 
            sequence_time_steps[_idx_input] = _idx_unique_time_steps
        return (set_time_steps, time_steps_duration, sequence_time_steps)

    def overwrite_raw_ts(self, element):
        """ this method overwrites the raw time series to the already once aggregated time series
        :param element: technology of the optimization """
        for ts in element.raw_time_series:
            element.raw_time_series[ts] = getattr(element, ts)

    @staticmethod
    def set_time_attributes(element, set_time_steps, time_steps_duration, sequence_time_steps):
        """ this method sets the operational time attributes of an element.
        :param element: element of the optimization
        :param set_time_steps: set_time_steps of operation
        :param time_steps_duration: time_steps_duration of operation
        :param sequence_time_steps: sequence of operation """
        if isinstance(element, TimeSeriesAggregation):
            element.set_time_steps = set_time_steps
            element.time_steps_duration = time_steps_duration
            element.sequence_time_steps = sequence_time_steps
        else:
            element.set_time_steps_operation = set_time_steps
            element.time_steps_operation_duration = time_steps_duration
            element.sequence_time_steps = sequence_time_steps

    def conduct_tsa(self):
        """ this method conducts time series aggregation """
        logging.info("\n--- Time series aggregation ---")
        # repeat order of operational time steps and link with investment and yearly time steps
        self.repeat_sequence_time_steps_for_all_years()
        logging.info("Calculate operational time steps for storage levels and energy balances")
        for element in self.energy_system.get_all_elements(StorageTechnology):
            # calculate time steps of storage levels
            element.calculate_time_steps_storage_level(conducted_tsa=self.conducted_tsa)
