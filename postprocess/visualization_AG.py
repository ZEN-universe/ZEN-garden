"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich
Description:  Class is defining the postprocessing of the results.
              The class takes as inputs the optimization problem (model) and the system configurations (system).
              The class contains methods to read the results and save them in a result dictionary (resultDict).
==========================================================================================================================================================================="""
import os
import pickle
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot    as plt
from   datetime             import datetime


class VisualizeResults:

    def __init__(self, dataset, scenario = "", pltShow = True):
        """postprocessing of the results of the optimization
        :param model:     optimization model
        :param pyoDict:   input data dictionary
        :param modelName: model name used for the directory to save the results in"""
        # set modelName
        self.dataset  = dataset
        self.scenario = scenario
        self.setModelName()
        self.nameDir  = f"outputs/{self.name}"
        # plot settings
        self.pltShow  = pltShow
        # init directories
        self.initDirectory("plots")
        self.initDirectory("files")
        # load results
        self.paramDict = self.loadResults("paramDict")
        self.varDict   = self.loadResults("varDict")
        self.analysis  = self.loadResults("Analysis")
        self.system    = self.loadResults("System")
        # get sets and set time-step duration
        self.getSets()
        self.setTimeStepsDuration()

    ## general methods
    def initDirectory(self,folder):
        """init directories to store plots and files"""
        if not os.path.exists(f"{self.nameDir}/{folder}"):
            os.makedirs(f"{self.nameDir}/{folder}")

    def setModelName(self):
        """set model name"""
        if self.scenario == str():
            self.name = self.dataset
        else:
            self.name = self.dataset + "_" + self.scenario

    def setTimeStepsDuration(self):
        """ set timesteps duration"""
        self.timeStepsCarrierDuration   = self.getDataframe("timeStepsCarrierDuration", indexNames=["carrier", "time"], type="param")
        self.timeStepsOperationDuration = self.getDataframe("timeStepsOperationDuration", indexNames=["technology", "time"], type="param")
        self.timeStepsInvestDuration    = self.getDataframe("timeStepsInvestDuration", indexNames=["technology", "year"], type="param")

    def loadResults(self, name, nameDir = None):
        """ load results from results folder"""
        if not nameDir:
            nameDir = self.nameDir
        with open(f"{nameDir}/{name}.pickle", "rb") as file:
            output = pickle.load(file)
        return output

    def getDataframe(self, name, indexNames, type = "var", subset = None, dct = {}):
        """plot built capacity"""
        # check whether parameter or variable is extracted
        if dct != {}:
            dct = dct[name]
        elif type == "param":
            dct = self.paramDict[name]
        else:
            dct = self.varDict[name]
        # get dct and values
        keys, values   = zip(*dct.items())
        if  len(indexNames)>1: # check if index is mulitiindex
            idx            = pd.MultiIndex.from_tuples(keys)
        else:
            idx            = list(keys)
        # create series
        df             = pd.Series(values, index=idx)
        # set index names and column name
        df.index.names = indexNames
        df.name        = name
        # round values
        try:
            df = df.round(decimals=4)
        except:
            pass
        # select a subset
        if subset:
            df = df.loc[subset]
        return df

    def updateTimeIndex(self, df, index="time"):
        """update the time index from numeric values to actual timestamps:
        index: indicate name of time index that is updated"""
        baseYear  = 2020
        yearsDict = {}
        for year in df.index.unique(index):
            yearsDict[year] = baseYear+year
        df = df.rename(index=yearsDict)
        return df

    def getSets(self):
        """ get sets from system"""
        # carriers
        self.setCarriers = self.system["setCarriers"]
        # conditioning technologies
        self.setConditioningTechnologies = self.system["setConditioningTechnologies"]
        self.setConditioningTechnologies.extend(["carbon_liquefication"])
        # conversion technologies
        self.setConversionTechnologies   = self.system["setHydrogenConversionTechnologies"]
        self.setConversionTechnologies   = list(set(self.setConversionTechnologies) - set(self.setConditioningTechnologies))
        # electricity generation Technologies
        self.setElectricityGenerationTechnologies = self.system["setElectricityGenerationTechnologies"]
        # transport technologies
        self.setTransportTechnologies             = self.system["setTransportTechnologies"]
        self.setTransportTechnologies.remove("electricity_transmission")
        # storage technologies
        self.setStorageTechnologies               = self.system["setStorageTechnologies"]
        self.setStorageTechnologies.append("carbon_storage")

    def barplot(self, title, df, stacked = False, ylabel=None, xlabel=None):
        """ stacked barplot"""
        if df.empty:
            print(f"{title} is empty.")
        elif df[df>0].isna().all().all():
            print(f"{title} all values are 0")
        else:
            fig, axs = plt.subplots()
            df.plot.bar(ax=axs, stacked=stacked)
            axs.set_title(title)
            axs.set_xlabel(xlabel)
            axs.set_ylabel(ylabel)
            fig.savefig(f"{self.nameDir}/plots/{title}.png")
            if self.pltShow:
                fig.show()
            plt.close(fig)

    def areaplot(self, name, df, ylabel=None, xlabel=None):
        """ area plot of dataframe"""
        if df.empty:
            print(f"{name} is empty.")
        elif df[df>0].isna().all().all():
            print(f"{name} all values are 0")
        else:
            fig, axs = plt.subplots()
            axs.set_title(name)
            df.plot.area(ax=axs)
            axs.set_xlabel(xlabel)
            axs.set_ylabel(ylabel)
            fig.savefig(f"{self.nameDir}/plots/{name}.png")
            if self.pltShow:
                fig.show()
            plt.close(fig)

    ## plot results
    def evaluateHydrogenDemand(self):
        """plot hydrogen demand"""
        demand = self.getDataframe("demandCarrier",["carrier", "node", "time"], type="param")
        demand = demand.unstack("node")
        demand = demand.apply(lambda row: row*self.timeStepsCarrierDuration)
        demand = demand.apply(lambda row: row*1e-6) #conversion from GWh in TWh
        # total hydrogen demand per country
        demandNodes = demand.loc["hydrogen"].sum()
        demandNodes = demandNodes[demandNodes >= 0.1 * demandNodes.max()]
        self.barplot("totalHydrogenDemandPerCountry", demandNodes, ylabel = "Hydrogen Demand in TWh" , xlabel= "years")
        # total hydrogen demand per country
        demandEvolution = demand.loc["hydrogen"].stack().groupby("time").sum()
        demandEvolution = demandEvolution[demandEvolution >= 0.1 * demandEvolution.max()]
        demandEvolution = self.updateTimeIndex(demandEvolution, index="time")
        self.barplot("totalHydrogenDemandTime", demandEvolution, ylabel = "Hydrogen Demand in TWh", xlabel = "years")

    def evaluateBuiltCapacity(self):
        """plot built capacity"""
        builtCapacity = self.getDataframe("builtCapacity",["technology", "location", "time"])

        # conversion technologies
        totalBuiltCapacity = builtCapacity[self.setConversionTechnologies].groupby(level=["technology","time"]).sum()
        self.barplot("totalBuiltCapacityConversion", totalBuiltCapacity.unstack("technology"), stacked=True)
        # electricity generation technologies
        totalBuiltCapacity = builtCapacity.loc[self.setElectricityGenerationTechnologies].groupby(level=["technology", "time"]).sum()
        self.barplot("totalBuiltCapacityElectricity", totalBuiltCapacity.unstack("technology"), stacked=True)

        # conditioning technologies
        totalBuiltCapacity = builtCapacity.loc[self.setConditioningTechnologies].groupby(level=["technology","time"]).sum()
        self.barplot("totalBuiltCapacityConditioning", totalBuiltCapacity.unstack("technology"), stacked=True)

        # transport technologies
        totalBuiltCapacity = builtCapacity.loc[self.setTransportTechnologies].groupby(level=["technology","time"]).sum()
        self.barplot("totalBuiltCapacityTransport", totalBuiltCapacity.unstack("technology"), stacked=True)

        # storage technologies
        totalBuiltCapacity = builtCapacity.loc[self.setStorageTechnologies].groupby(level=["technology","time"]).sum()
        self.barplot("totalBuiltCapacityStorage", totalBuiltCapacity.unstack("technology"), stacked=True)

    def evaluateCapacity(self):
        """plot installed capacity"""
        capacity = self.getDataframe("capacity", ["technology", "location", "time"]).round(decimals=4)

        # conversion technologies
        totalCapacity = capacity.loc[self.setConversionTechnologies].groupby(level=["technology","time"]).sum()
        self.barplot("totalCapacityConversion", totalCapacity.unstack("technology"), stacked=True)
        totalCapacity = capacity.loc[self.setConversionTechnologies].groupby(level=["technology", "location"]).sum()
        totalCapacity = totalCapacity[totalCapacity >= 0.1 * totalCapacity.max()]
        self.barplot("totalCapacityConversionNodes", totalCapacity.unstack("technology"), stacked=True)
        # electricity generation technologies
        totalCapacity = capacity.loc[self.setElectricityGenerationTechnologies].groupby(level=["technology", "time"]).sum()
        self.barplot("totalCapacityElectricity", totalCapacity.unstack("technology"), stacked=True)
        totalCapacity = capacity.loc[self.setElectricityGenerationTechnologies].groupby(level=["technology", "location"]).sum()
        totalCapacity = totalCapacity[totalCapacity>= 0.1*totalCapacity.max()]
        self.barplot("totalCapacityElectricityNodes", totalCapacity.unstack("technology"), stacked=True)

        # transport technologies
        totalCapacity = capacity.loc[self.setTransportTechnologies].groupby(level=["technology","time"]).sum()
        self.barplot("totalCapacityTransport", totalCapacity.unstack("technology"), stacked=True)

        # storage technologies
        totalCapacity = capacity.loc[self.setStorageTechnologies].groupby(level=["technology","time"]).sum()
        self.barplot("totalCapacityStorage", totalCapacity.unstack("technology"), stacked=True)

    def evaluateCarrierImports(self):
        """plot carrier imports"""
        carrierImports = self.getDataframe("importCarrierFlow", ["carrier", "location", "time"])
        carrierImports = carrierImports.reorder_levels(["carrier", "time", "location"]).unstack()
        carrierImports = carrierImports.apply(lambda row: row*self.timeStepsCarrierDuration)
        self.areaplot(f"carrierImports", carrierImports.stack().groupby(["carrier","time"]).sum().unstack("carrier"))
        # electricity and natural gas imports
        for carrier in carrierImports.index.unique("carrier"):
            imports = carrierImports.loc[carrier].sum()
            imports = imports[imports >= 0.1 * imports.max()]
            self.barplot(f"{carrier}Imports", imports, stacked=False)

    def checkCarrierExports(self):
        """check if carrier exports are zero"""
        carrierExports = self.getDataframe("exportCarrierFlow", ["carrier", "location", "time"])
        carrierExports = carrierExports.reorder_levels(["carrier", "time", "location"]).unstack()
        carrierExports = carrierExports.apply(lambda row: row * self.timeStepsCarrierDuration)
        if carrierExports.sum(axis=0).sum().round(2) != 0:
            print("Carrier exports are not 0.")

    def evaluateInputFlow(self):
        """plot carrier flow"""
        inputFlow  = self.getDataframe("inputFlow", ["technology","carrier", "location", "time"])
        inputFlow  = inputFlow * self.timeStepsCarrierDuration
        inputFlow  = inputFlow.reorder_levels(["technology", "carrier", "location", "time"]) * 1e-3

        # inputFlows Conversion
        inputFlowConversion = inputFlow.loc[self.setConversionTechnologies]
        inputFlowConversion = inputFlowConversion.unstack("carrier")
        inputFlowConversion = self.updateTimeIndex(inputFlowConversion, index="time")
        self.areaplot(f"inputFlowsConversion", inputFlowConversion.groupby("time").sum(), ylabel = "Input flow in TWh")
        for carrier in inputFlowConversion.columns:
            tmp = inputFlowConversion[carrier].groupby(["technology","time"]).sum()
            self.barplot(f"{carrier}InputFlowsConversion", tmp.unstack("technology"), stacked=True)

        # inputFlows Conditioning
        inputFlowCodnitioning = inputFlow.loc[self.setConditioningTechnologies].unstack("carrier")
        for carrier in inputFlowCodnitioning.columns:
            tmp = inputFlowCodnitioning[carrier].groupby(["technology", "time"]).sum()
            self.barplot(f"{carrier}InputFlowsConditioning", tmp.unstack("technology"), stacked=True)

        # inputFlows Storage
        inputFlowStorage = inputFlow.loc[self.setStorageTechnologies].unstack("carrier")
        for carrier in inputFlowStorage.columns:
            tmp = inputFlowStorage[carrier].groupby(["technology", "time"]).sum()
            self.barplot(f"{carrier}InputFlowsStorage", tmp.unstack("technology"), stacked=True)

    def evaluateOutputFlow(self):
        """evaluation output flows"""
        outputFlow = self.getDataframe("outputFlow", ["technology", "carrier", "location", "time"])
        tsDuration = self.timeStepsOperationDuration.loc[self.setConversionTechnologies]
        # inputFlows Conversion
        outputFlow = outputFlow.loc[self.setConversionTechnologies]
        outputFlow = outputFlow.reorder_levels(["carrier", "technology", "location", "time"])
        for carrier in outputFlow.index.unique("carrier"):
            output = outputFlow.loc[carrier].groupby(["technology", "time"]).sum() * tsDuration
            self.barplot(f"{carrier}OutputFlowsConversion", output.unstack("technology"), stacked=True)

    def evaluateCarrierFlowsTransport(self):
        """evaluate carrier flows transport techs"""
        carrierFlow = self.getDataframe("carrierFlow", ["technology", "location", "time"])
        for carrier in ["hydrogen", "carbon", "electricity"]:
            tmp = [tech for tech in carrierFlow.index.unique("technology") if carrier in tech]
            tmp = carrierFlow.loc[tmp].groupby(["technology", "time"]).sum()
            tmp = tmp * self.timeStepsOperationDuration.loc[tmp.index.unique("technology")]
            self.barplot(f"{carrier}FlowsTransport", tmp.unstack("technology"), stacked=True)

    def evaluateCarbonEmissions(self, decarbScen):
        """plot carbon emissions
        :param decarbScen: dictionary with information on decarbonization scenarios"""

        carbonEmissions = self.getDataframe("carbonEmissionsTotal", ["year"])
        self.barplot("carbonEmissionsYearly", carbonEmissions)
        carbonEmissions = carbonEmissions *1e3 # kilotons to tons

        if self.scenario == "default" and decarbScen != {}:
            # load min emissions results
            varDictMinEm       = self.loadResults("varDict", nameDir=self.nameDir + "_min_em", )
            minCarbonEmissions = self.getDataframe("carbonEmissionsTotal", ["year"], dct = varDictMinEm) *1e3 # kilotons to tons
            for scen, range in decarbScen.items():
                if scen == "linear":
                    self.computeCarbonEmissionsLimits(minCarbonEmissions, carbonEmissions, range)
                if scen == "carbonBudget":
                    self.computeCarbonBudget(minCarbonEmissions, carbonEmissions, range)

    def computeCarbonEmissionsLimits(self, minCarbonEmissions, carbonEmissions, range):
        """generate input files linear decarbonization pathway scenarios
        :param minCarbonEmissions: minimum carbon emissions
        :param carbonEmissions:    cost minimal carbon emissions"""
        # years and min and max values
        years     = carbonEmissions.index.unique("year")
        carbonMax = min(carbonEmissions)
        carbonMin = minCarbonEmissions.loc[years[-1]].round(2)
        # dataframe for results
        carbonLimits = pd.Series(np.nan, index=minCarbonEmissions.index, name="carbonEmissionsLimit")
        carbonLimits.index.name = "time"
        carbonLimits.loc[years[0]] = np.Inf
        for factor in range:
            name = str(factor).replace(".","-")
            if carbonMin != 0:
                print("Carbon emissions of 0 are not reached. The minimal carbon emissions are", carbonMin)
                # reduction    = (carbonMax - carbonMin) / max(years)
                # carbonLimits.loc[years[1:]] = [carbonMax - reduction * y for y in years[1:]]
                # carbonLimits.to_csv(f"{self.nameDir}/files/carbonEmissionsLimit_linear_min.csv")
                # carbonMin = 0
            reduction = (carbonMax - carbonMin) * factor / max(years)
            carbonLimits.loc[years[1:]] = [carbonMax - reduction * y for y in years[1:]]
            carbonLimits.to_csv(f"data/{self.dataset}/setScenarios/carbonEmissionsLimit_linear_{name}.csv")
        return carbonMin, carbonMax

    def computeCarbonBudget(self, minCarbonEmissions, carbonEmissions, range):
        """generate input files for carbonBudget scenarios"""
        # initial carbon budget
        initialCarbonBudget = carbonEmissions.sum()
        carbonBudget = {"index": "carbonEmissionsBudget", "value": initialCarbonBudget, "unit": "tons"}
        carbonBudget = pd.DataFrame(carbonBudget, index=[0])
        # carbon emissions target
        carbonLimits            = minCarbonEmissions.tail(1)
        carbonLimits.name       = "carbonEmissionsLimit"
        carbonLimits.index.name = "time"
        # carbon budget scenarios
        for factor in range:
            name = str(factor).replace(".", "-")
            carbonBudget["value"] = initialCarbonBudget * factor
            carbonBudget.to_csv(f"data/{self.dataset}/setScenarios/carbonEmissionsBudget_carbonBudget_{name}.csv", index=False)
            carbonLimits.to_csv(f"data/{self.dataset}/setScenarios/carbonEmissionsLimit_carbonBudget_{name}.csv", index=True)

    def computeLevelizedCost(self, carrier):
        """compute marginal cost"""
        tsDuration = self.timeStepsOperationDuration
        years      = self.timeStepsInvestDuration.index.unique("year")
        # carrier specific data
        if carrier == "electricity":
            subset = self.setElectricityGenerationTechnologies
            input  = None
            cost   = pd.Series(0, index=years)
        elif carrier == "hydrogen":
            subset = self.setConversionTechnologies
            input = self.getDataframe("inputFlow", ["technology", "carrier", "location", "time"], subset=subset).unstack("technology")
            price = self.getDataframe("importPriceCarrier", ["carrier", "location", "time"], type="param")
            costs = input.apply(lambda row: row * price)
        # get capital expenditures, operational expenditures, and output flows
        capex      = self.getDataframe("capexYearly", ["technology", "location", "year"],subset=subset)
        opex       = self.getDataframe("opex", ["technology", "location", "time"],subset=subset)
        output     = self.getDataframe("outputFlow", ["technology", "carrier", "location", "time"], subset=subset)
        # create empty dataframe for levelized cost of energy and determine LCOE for each technology
        LCOE = pd.DataFrame(np.nan,columns=years, index=subset)
        for tech in subset:
            capx = capex.loc[tech].groupby("year").sum()
            opx  = opex.loc[tech].groupby("time").sum() * tsDuration.loc[tech]
            out  = output.loc[tech,carrier,:].groupby("time").sum() * tsDuration.loc[tech]
            if isinstance(input, pd.DataFrame):
                cost = costs[tech].groupby(["time"]).sum() * tsDuration.loc[tech]
            for year in years:
                if not out.loc[year] == 0:
                    lc = (capx + opx + cost) / out
                    LCOE.loc[tech] = lc
        # plot levelized cost of energy
        if not LCOE.dropna().empty:
            LCOE.to_csv(f"{self.nameDir}/files/levelizedCost_{carrier}.csv")
            self.barplot(f"levelizedCost_{carrier}", LCOE.stack().unstack(0), stacked=False)

## method to run visualization
def run(dataset, scenario, pltShow=False, decarbScen = {}):
    """visualize and evaluate results"""
    visResults = VisualizeResults(dataset, scenario, pltShow=pltShow)
    ## params
    visResults.evaluateHydrogenDemand()
    ## vars
    # installed capacities
    visResults.evaluateBuiltCapacity()
    visResults.evaluateCapacity()
    visResults.evaluateCarbonEmissions(decarbScen)
    # carrier flows
    visResults.evaluateCarrierImports()
    visResults.checkCarrierExports()
    visResults.evaluateInputFlow()
    visResults.evaluateOutputFlow()
    visResults.evaluateCarrierFlowsTransport()
    ## compute marginal cost of electricity
    visResults.computeLevelizedCost("electricity")
    visResults.computeLevelizedCost("hydrogen")


if __name__ == "__main__":
    os.chdir("..")
    dataset = "HSC_NUTS0"
    pltShow = False  # True or False
    scenarios  = ["default"] #,"default_min_em" #"default_no_REN"
    #scenarios = ["linear_1.0", "linear_0.95", "carbonBudget_0.1", "carbonBudget_0.5"]
    # scenarios = ["carbonBudget"]
    decarbScen = {"linear": np.arange(0.1, 1, 0.1).round(2), "carbonBudget": np.arange(0.4, 0, -0.1).round(2)}
    #today      = datetime.now()
    #modelName  = "model_" + today.strftime("%Y-%m-%d") + "_perfectForesight_" + dataset
    #modelName = "model_" + today.strftime("%Y-%m-%d") + "_" + spatialRes
    for scenario in scenarios:
        run(dataset, scenario, pltShow=pltShow, decarbScen = decarbScen)
    a=1