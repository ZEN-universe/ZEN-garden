"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Scenario settings settings.
==========================================================================================================================================================================="""
scenarios = dict()
scenarios["1"] = {"system": {'reference_year': 2023,
                             'set_nodes': ['CH'],
                             'set_transport_technologies': [],
                             'conduct_time_series_aggregation': True,
                             'unaggregated_time_steps_per_year': 8760,
                             'aggregated_time_steps_per_year': 1},
                  'analysis': {"discount_rate": 0.03}}
