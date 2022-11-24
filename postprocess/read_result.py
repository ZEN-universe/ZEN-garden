import pickle
import os
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import operator
import numpy as np
from postprocess.results_JM import Postprocess
from outputs.meijun_func.color_scheme import gen_type_color
from outputs.meijun_func.life_easymaker import extractFirstEntry

# The raw output pickle

# filePath = os.getcwd() + '\\outputs\\NUTS0_electricity_short0\\varDict.pickle'
# data1 = pd.read_pickle(filePath)
# print(data1.keys())
# print(data1['importCarrierFlow'])






# Disaggregated output

if __name__ == "__main__":

    dataSetName = "NUTS0_electricity"

    scenarioNames = {
        "short0"    :"PF",
    }
    scenarios = {}

    # reload results
    reloadResults = False
    # folder specification
    folderPath = "outputs/"
    if not os.path.exists(folderPath + "resultsScenario.pickle") or reloadResults:
        for scenarioName in scenarioNames:
            scenarios[scenarioName] = Postprocess(
                dataSetName     = dataSetName,
                scenarioName    = scenarioName,
                scenarioType    = scenarioNames[scenarioName],
                folderPath      = folderPath,
            )
        with open(folderPath + "resultsScenario.pickle", "wb") as inputFile:
            pickle.dump(scenarios, inputFile)
        reloaded = True
    else:
        with open(folderPath + "resultsScenario.pickle", "rb") as inputFile:
            scenarios = pickle.load(inputFile)
        reloaded = False

    # Trying around with the preprocessing data #

    # Finding out the existing countries
    # countries_list = extractFirstEntry(list(scenarios["short0"].calculateFullTimeSeries(
    #     scenarios["short0"].extractDataFromRaw("outputFlow")["nuclear", "electricity"],
    #     elementName="electricity").keys()))
    # print(len(countries_list))

    # Finding out the existing generation types. Since each country has all the technologies, it does not need to
    # filter the country in the MultiIndex structure





    # Getting data for the plot #
    country = "AT"
    hours = len(scenarios["short0"].calculateFullTimeSeries(
        scenarios["short0"].extractDataFromRaw("demandCarrier", 'par')["electricity"],
        elementName="electricity")[country])
#
    tech_list = extractFirstEntry(list(scenarios["short0"].extractDataFromRaw("outputFlow").keys()))
    gen = {}
    for tech in tech_list:
        try:
            gen[tech] = scenarios["short0"].calculateFullTimeSeries(
                scenarios["short0"].extractDataFromRaw("outputFlow")[tech, "electricity"],
                elementName="natural_gas_turbine")[country]
        except:
            continue
    # print(gen.keys())
    demand = scenarios["short0"].calculateFullTimeSeries(
        scenarios["short0"].extractDataFromRaw("demandCarrier", 'par')["electricity"],
        elementName="electricity")[country]
    pos_phs = scenarios["short0"].calculateFullTimeSeries(
        scenarios["short0"].extractDataFromRaw("carrierFlowDischarge")['pumped_hydro'],
        elementName="electricity")[country]
    neg_phs = scenarios["short0"].calculateFullTimeSeries(
        scenarios["short0"].extractDataFromRaw("carrierFlowCharge")['pumped_hydro'],
        elementName="electricity")[country]
    # DE_gas_stor = scenarios["short0"].fullStorageLevel["natural_gas_storage", "DE"]

    lines_list = extractFirstEntry(list(scenarios["short0"].calculateFullTimeSeries(
        scenarios["short0"].extractDataFromRaw("carrierFlow")["power_line"], elementName="electricity").keys()))
    import_lines = []
    export_lines = []
    for item in lines_list:
        start_point, end_point = item.split("-")
        if start_point == country:
            export_lines.append(item)
        if end_point == country:
            import_lines.append(item)
    elecImport = np.zeros(hours)
    elecExport = np.zeros(hours)
    for item in import_lines:
        raw_import = scenarios["short0"].calculateFullTimeSeries(
            scenarios["short0"].extractDataFromRaw("carrierFlow")["power_line"], elementName="electricity")[item]
        loss_import = scenarios["short0"].calculateFullTimeSeries(
            scenarios["short0"].extractDataFromRaw("carrierLoss")["power_line"], elementName="electricity")[item]
        elecImport = list(map(operator.sub, list(map(operator.add, elecImport, raw_import)), loss_import))
    for item in export_lines:
        elecExport = list(map(operator.add, elecExport, scenarios["short0"].calculateFullTimeSeries(
            scenarios["short0"].extractDataFromRaw("carrierFlow")["power_line"], elementName="electricity")[item]))
    # shed_demand = list(scenarios["short0"].calculateFullTimeSeries(
    #     scenarios["short0"].extractDataFromRaw("shedDemandCarrier")["electricity"], elementName="electricity")[country])
    # netDemand = list(map(operator.add, list(map(operator.add, demand, neg_phs)), elecExport))
    # netDemand = list(map(operator.sub, netDemand, shed_demand))



    # Start to plot! #
    fig = go.Figure()

    for tech in tech_list:
        if tech in gen:
            if max(gen[tech]) > 0:
                color_tech = gen_type_color(tech)
                fig.add_trace(go.Scatter(
                    x=[*range(1, len(gen[tech]))], y=gen[tech],
                    hoverinfo='x+y+name',
                    name=tech,
                    mode='lines',
                    #             line=dict(width=0.5, color=colorf,shape='spline',smoothing=0.4),
                    line=dict(width=0.5, color=color_tech),
                    stackgroup='one'  # define stack group
            ))

    fig.add_trace(go.Scatter(
        x=[*range(1, len(pos_phs))], y=pos_phs,
        hoverinfo='x+y+name',
        name='PHS Discharge',
        mode='lines',
        line=dict(width=0.5, color=gen_type_color("pumped_hydro")),
        stackgroup='one'
    ))

    fig.add_trace(go.Scatter(
        x=[*range(1, len(neg_phs))], y=list(map(operator.neg, neg_phs)),
        hoverinfo='x+y+name',
        name='PHS Charge',
        mode='lines',
        line=dict(width=0.5, color=gen_type_color("pumped_hydro")),
        stackgroup='two'
    ))

    fig.add_trace(go.Scatter(
        x=[*range(1, len(elecImport))], y=elecImport,
        hoverinfo='x+y+name',
        name='Power Import',
        mode='lines',
        line=dict(width=0.5, color=gen_type_color("powerExchange")),
        stackgroup='one'
    ))

    fig.add_trace(go.Scatter(
        x=[*range(1, len(elecExport))], y=list(map(operator.neg, elecExport)),
        hoverinfo='x+y+name',
        name='Power Export',
        mode='lines',
        line=dict(width=0.5, color=gen_type_color("powerExchange")),
        stackgroup='two'
    ))

    fig.add_trace(go.Scatter(
        x=[*range(1, len(netDemand))], y=netDemand,
        hoverinfo='x+y+name',
        name='Net Demand',
        mode='lines',
        line=dict(width=1.5, color='black'),
    ))

    fig.add_trace(go.Scatter(
        x=[*range(1, len(demand))], y=demand,
        hoverinfo='x+y+name',
        name='Demand',
        mode='lines',
        line=dict(width=1.5, color='black', dash='dash'),
    ))
#
    fig.update_layout(title_text=country + ' hourly profile',
                      width=1400,
                      height=900)
    fig.update_xaxes(title_text='Hour')
    fig.update_yaxes(title_text='Power [GW]')

    fig.show()




    # Raw data tryout section
    # To get the data from storage, put "carrierFlowCharge" or "carrierFlowDischarge"
    # print(scenarios["short0"].calculateFullTimeSeries(
    #     scenarios["short0"].extractDataFromRaw("carrierFlowCharge")['pumped_hydro'], elementName="electricity")["DE"])

    # To get the data from electricity transmission
    # print(scenarios["short0"].calculateFullTimeSeries(
    #     scenarios["short0"].extractDataFromRaw("carrierFlow")['power_line'], elementName="electricity"))

    # print(max(scenarios["short0"].calculateFullTimeSeries(
    #     scenarios["short0"].extractDataFromRaw("shedDemandCarrier")["electricity"], elementName="electricity")["DE"]))

    # Toy code to check if there is power flow from two directions on the same time. Tried several connections and there
    # is none
    # line_one = list(scenarios["short0"].calculateFullTimeSeries(
    #     scenarios["short0"].extractDataFromRaw("carrierFlow")["power_line"], elementName="electricity")["IT-CH"])
    # line_two = list(scenarios["short0"].calculateFullTimeSeries(
    #     scenarios["short0"].extractDataFromRaw("carrierFlow")["power_line"], elementName="electricity")["CH-IT"])
    # count = 0
    # for i in range(len(line_one)):
    #     if (line_one[i] > 0) & (line_two[i] > 0):
    #         count += 1
    # print(count)