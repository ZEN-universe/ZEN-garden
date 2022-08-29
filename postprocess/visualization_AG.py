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
import shutil
import copy
import numpy as np
import pandas as pd

import matplotlib.pyplot          as plt
from   postprocess.eth_colorsAG   import ETHColors
from   datetime             import datetime


class VisualizeResults:

    def __init__(self, dataset, scenario = "", pltShow = False, outputDir = "outputs"):
        """postprocessing of the results of the optimization
        :param model:     optimization model
        :param pyoDict:   input data dictionary
        :param modelName: model name used for the directory to save the results in"""
        # set modelName
        self.dataset  = dataset
        self.scenario = scenario
        self.setModelName(scenario)
        self.setOutputDir(outputDir)
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
        # plot settings
        self.colormap = ETHColors()

    def setOutputDir(self, outputDir):
        """set name of output directory"""
        self.outputDir = outputDir
        self.nameDir   = f"{self.outputDir}/{self.name}"


    ## general methods
    def initDirectory(self,folder):
        """clear directories to store plots and files"""
        if os.path.exists(f"{self.nameDir}/{folder}"):
            shutil.rmtree(f"{self.nameDir}/{folder}")
        os.makedirs(f"{self.nameDir}/{folder}")

    def setModelName(self, name):
        """set model name"""
        if name == str():
            self.name = self.dataset
        else:
            self.name = self.dataset + "_" + name

    def setTimeStepsDuration(self):
        """ set timesteps duration"""
        self.timeStepsCarrierDuration   = self.getDataframe("timeStepsOperationDuration", indexNames=["carrier", "time"], type="param", subset=self.setCarriers)
        self.timeStepsOperationDuration = self.getDataframe("timeStepsOperationDuration", indexNames=["technology", "time"], type="param", subset=self.system["setTechnologies"])
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
        self.setCarriers = copy.deepcopy(self.system["setCarriers"])
        # conditioning technologies
        self.setConditioningTechnologies = copy.deepcopy(self.system["setConditioningTechnologies"])
        # conversion technologies
        self.setConversionTechnologies   = copy.deepcopy(self.system["setHydrogenConversionTechnologies"])
        self.setConversionTechnologies   = list(set(self.setConversionTechnologies) - set(self.setConditioningTechnologies))
        if "carbon_storage" in self.setConversionTechnologies:
            self.setConversionTechnologies.remove("carbon_storage")
        # electricity generation Technologies
        self.setElectricityGenerationTechnologies = copy.deepcopy(self.system["setElectricityGenerationTechnologies"])
        # transport technologies
        self.setTransportTechnologies = copy.deepcopy(self.system["setTransportTechnologies"])
        # storage technologies
        self.setStorageTechnologies = copy.deepcopy(self.system["setStorageTechnologies"])
        self.setStorageTechnologies.append("carbon_storage")

    def barplot(self, title, df, stacked = False, ylabel=None, xlabel=None):
        """ stacked barplot"""
        if df.empty:
            print(f"{title} is empty.")
        elif df[df>0].isna().all().all():
            print(f"{title} all values are 0")
        else:
            fig, axs = plt.subplots()
            if isinstance(df, pd.Series) or len(df.columns)>1:
                df.plot.bar(ax=axs, stacked=stacked)
            else:
                df.plot.bar(ax=axs, stacked=stacked, color= self.colormap.retrieveColorsDict(df.columns))
            axs.set_title(title)
            axs.set_xlabel(xlabel)
            axs.set_ylabel(ylabel)
            fig.savefig(f"{self.nameDir}/plots/{title}.png")
            if self.pltShow:
                fig.show()
            plt.close(fig)

    def areaplot(self, name, df, ylabel=None, xlabel=None):
        """ area plot of dataframe"""
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if df.empty:
            print(f"{name} is empty.")
        elif df[df.round(2)>0].isna().all().all():
            print(f"{name} all values are 0")
        else:
            df = df.round(2)
            fig, axs = plt.subplots()
            axs.set_title(name)
            if isinstance(df, pd.Series) or len(df.columns) == 1:
                df.plot.area(ax=axs, linewidth=0)
            else:
                df.plot.area(ax=axs, color=self.colormap.retrieveColorsDict(df.columns), linewidth=0)
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
        builtCapacity = self.getDataframe("builtCapacity",["technology", "capacityType", "location", "time"])

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
        capacity = self.getDataframe("capacity", ["technology", "capacityType", "location", "time"]).round(decimals=4)

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

    def evaluateCarrierImportsExports(self):
        """plot carrier imports"""
        carrierImports = self.getDataframe("importCarrierFlow", ["carrier", "location", "time"])
        carrierImports = carrierImports.groupby(["carrier", "time"]).sum() * self.timeStepsCarrierDuration
        self.areaplot(f"carrierImports", carrierImports.unstack("carrier"))
        # electricity and natural gas imports
        # for carrier in carrierImports.index.unique("carrier"):
        #     imports = carrierImports.loc[carrier]
        #     imports = imports[imports >= 0.1 * imports.max()]
        #     self.barplot(f"{carrier}Imports", imports, stacked=False)

        #check if carrier exports are zero
        carrierExports = self.getDataframe("exportCarrierFlow", ["carrier", "location", "time"])
        carrierExports = carrierExports.reorder_levels(["carrier", "time", "location"]).unstack()
        carrierExports = carrierExports.apply(lambda row: row * self.timeStepsCarrierDuration)
        if carrierExports.sum(axis=0).sum().round(2) != 0:
            print("Carrier exports are not 0.")

    def evaluateCarrierFlow(self):
        """plot carrier flow"""
        unit = {"tons": ["wet_biomass","carbon","carbon_liquid"],
                "TWh": ["natural_gas", "biomethane", "dry_biomass", "hydrogen", "hydrogen_liquid", "hydrogen_high", "electricity"]}
        unitDict = {carrier: unit for unit, carriers in unit.items() for carrier in carriers}
        # outputFlows
        outputFlow = self.getDataframe("outputFlow", ["technology", "carrier", "location", "time"])
        outputFlow = outputFlow * self.timeStepsOperationDuration.loc[outputFlow.index.unique("technology")]
        # anaerobic digestion
        outputFlow = outputFlow.unstack(level="carrier")
        outputFlow.loc["anaerobic_digestion", "biomethane"]  = outputFlow.loc["anaerobic_digestion", "natural_gas"].values
        outputFlow.loc["anaerobic_digestion", "natural_gas"] = np.nan
        outputFlow = outputFlow.stack().reorder_levels(["carrier", "technology", "location", "time"])
        # unit conversion
        carriers = [carrier for carrier in unit["TWh"] if carrier in outputFlow.index.unique("carrier")]
        outputFlow.loc[carriers] = outputFlow.loc[carriers] * 1e-6  # MWh to TWh
        for carrier in outputFlow.index.unique("carrier"):
            output = outputFlow.loc[carrier].groupby(["technology", "time"]).sum()
            self.areaplot(f"{carrier}OutputFlows", output.unstack("technology"), ylabel=f"Output flow {carrier} in {unitDict[carrier]}")

        inputFlow  = self.getDataframe("inputFlow", ["technology","carrier","location","time"])
        inputFlow  = inputFlow * self.timeStepsOperationDuration.loc[inputFlow.index.unique("technology")]
        inputFlow = inputFlow.unstack(level="carrier")
        # unit conversion
        carriers = [carrier for carrier in unit["TWh"] if carrier in inputFlow.columns]
        inputFlow[carriers] = inputFlow[carriers] * 1e-6  # MWh to TWh
        # anaerobic digestion
        if outputFlow.loc["biomethane","anaerobic_digestion",:,:].sum() > 0:
            inputFlow["biomethane"] = np.nan
            # biomethane used in SMR and SMR90
            if "SMR" in inputFlow.index.unique("technology"):
                deltaSMR = inputFlow["natural_gas"].loc["SMR",:,:].values - outputFlow.loc["biomethane","anaerobic_digestion",:,:].values
                if (deltaSMR >= 0).all():
                    inputFlow["natural_gas"].loc["SMR", :, :] = deltaSMR
                    inputFlow["biomethane"].loc["SMR", :, :]  = outputFlow.loc["biomethane", "anaerobic_digestion", :,:].values
            # biomethane used in SMR
            if "SMR90" in inputFlow.index.unique("technology") and inputFlow["biomethane"].isna().all():
                deltaSMR90 = inputFlow["natural_gas"].loc["SMR90", :, :].values - outputFlow.loc["biomethane","anaerobic_digestion",:,:].values
                if (deltaSMR90 >= 0).all():
                    inputFlow["natural_gas"].loc["SMR90", :, :] = deltaSMR90
                    inputFlow["biomethane"].loc["SMR90", :, :]  = outputFlow.loc["biomethane", "anaerobic_digestion", :, :].values
            # biomethane used in SMR and SMR90
            if inputFlow["biomethane"].isna().all():
                totalGas = inputFlow["natural_gas"].loc["SMR", :, :].values + inputFlow["natural_gas"].loc["SMR90", :, :].values
                # biomethane SMR
                biomethaneSMR   = inputFlow["natural_gas"].loc["SMR", :, :].values
                biomethaneTotal = outputFlow.loc["biomethane", "anaerobic_digestion", :,:].values
                naturalGasSMR   = deltaSMR
                # only biomethane
                biomethaneSMR[deltaSMR>0] = biomethaneSMR[deltaSMR>0] - deltaSMR[deltaSMR>0]
                naturalGasSMR[deltaSMR<0] = 0
                # update input flows
                inputFlow["biomethane"].loc["SMR", :, :] = biomethaneSMR
                inputFlow["natural_gas"].loc["SMR", :, :] = naturalGasSMR
                # determine remaining biomethane
                biomethaneSMR90 = (biomethaneTotal - biomethaneSMR).round(5)
                assert (biomethaneSMR90>=0).all(), "Error in biomethane calculations."
                deltaSMR90      = inputFlow["natural_gas"].loc["SMR90", :, :].values - biomethaneSMR90
                # remaining NG
                naturalGasSMR[deltaSMR<0]     = 0
                # now determine how much of both is used
                inputFlow["biomethane"].loc["SMR90", :, :]  = biomethaneSMR90
                inputFlow["natural_gas"].loc["SMR90", :, :] = deltaSMR90
                newTotalGas = inputFlow["biomethane"].loc["SMR", :, :].values+inputFlow["biomethane"].loc["SMR90", :, :].values\
                              +inputFlow["natural_gas"].loc["SMR", :, :].values+inputFlow["natural_gas"].loc["SMR90", :, :].values
                assert (totalGas-newTotalGas).sum()==0, "Error in biomethane calculations"
        inputFlow  = inputFlow.stack() #.reorder_levels(["technology","carrier", "location", "time"])
        inputFlow  = self.updateTimeIndex(inputFlow, index="time")
        input      = inputFlow.loc[self.setConversionTechnologies].groupby(["technology", "time"]).sum()
        self.areaplot(f"CarrierInputFlowsConversion", input.unstack("technology"),ylabel=f"Inputs conversion in TWh")
        input = inputFlow.loc[self.setConversionTechnologies].groupby(["carrier", "time"]).sum().round(4)
        self.areaplot(f"CarrierInputFlowsConversion", input.unstack("carrier"), ylabel=f"Carrier inputs conversion in TWh")
        # hydrogen production technologies
        inputFlow  = inputFlow.reorder_levels(["carrier", "technology", "location", "time"])
        for carrier in inputFlow.index.unique("carrier"):
            input = inputFlow.loc[carrier].groupby(["technology", "time"]).sum()
            self.areaplot(f"{carrier}InputFlows", input.unstack("technology"), ylabel=f"Input flow {carrier} in {unitDict[carrier]}")

        # outputFlows Transport
        carrierFlow = self.getDataframe("carrierFlow", ["technology", "location", "time"])
        carrierFlow = carrierFlow * self.timeStepsOperationDuration.loc[carrierFlow.index.unique("technology")]
        for carrier in ["hydrogen", "carbon", "electricity"]:
            tmp = [tech for tech in carrierFlow.index.unique("technology") if carrier in tech]
            flow = carrierFlow.loc[tmp].groupby(["technology", "time"]).sum() * 1e-6
            self.areaplot(f"{carrier}CarrierFlowsTransport", flow.unstack("technology"), ylabel=f"Carrier flow {carrier} in {unitDict[carrier]}")
        tmp = [tech for tech in carrierFlow.index.unique("technology") if "dry_biomass" in tech]
        flow = carrierFlow.loc[tmp].groupby(["technology", "time"]).sum()
        self.areaplot(f"dry_biomassCarrierFlowsTransport", flow.unstack("technology"),ylabel=f"Carrier flow dry_biomass in tons")

    def evaluateCarbonEmissions(self, decarbScen):
        """plot carbon emissions
        :param decarbScen: dictionary with information on decarbonization scenarios"""

        carbonEmissions = self.getDataframe("carbonEmissionsTotal", ["year"])
        self.barplot("carbonEmissionsYearly", carbonEmissions)
        carbonEmissions = carbonEmissions #*1e3 # kilotons to tons

        if self.scenario in ["base","default",""] and decarbScen:
            # load min emissions results
            varDictMinEm       = self.loadResults("varDict", nameDir=self.nameDir + "_min_em", )
            minCarbonEmissions = self.getDataframe("carbonEmissionsTotal", ["year"], dct = varDictMinEm) *  1.005 # kilotons to tons
            for scen, range in decarbScen.items():
                if scen == "linear":
                    self.computeCarbonEmissionsLimits(minCarbonEmissions, carbonEmissions, range)
                if scen == "carbonBudget":
                    self.computeCarbonBudget(minCarbonEmissions, carbonEmissions, range)
        if "min_em" in self.scenario:
            carbonEmissions.name = "carbonEmissionsLimit"
            carbonEmissions.index.name = "time"
            carbonEmissions[carbonEmissions>0] = carbonEmissions[carbonEmissions>0] * 1.001
            carbonEmissions[carbonEmissions<0] = carbonEmissions[carbonEmissions<0] * 0.991
            carbonEmissions.to_csv(f"data/{self.dataset}/systemSpecification/carbonEmissionsLimit_{self.scenario}.csv")

    def computeCarbonEmissionsLimits(self, minCarbonEmissions, carbonEmissions, range):
        """generate input files linear decarbonization pathway scenarios
        :param minCarbonEmissions: minimum carbon emissions
        :param carbonEmissions:    cost minimal carbon emissions"""
        # years and min and max values
        years     = carbonEmissions.index.unique("year")
        carbonMax = min(carbonEmissions)
        carbonMin = minCarbonEmissions.loc[years[-1]].round(2)
        if carbonMin != 0:
            print("The minimal carbon emissions are", carbonMin)
            if carbonMin < 0:
                carbonMin = 0
        # dataframe for results
        carbonLimits = pd.Series(np.nan, index=minCarbonEmissions.index, name="carbonEmissionsLimit")
        carbonLimits.index.name = "time"
        carbonLimits.loc[years[0]] = np.Inf
        for factor in range:
            if factor == 0:
                carbonLimits.loc[years] = np.Inf
            else:
                reduction = (carbonMax - carbonMin) * factor / max(years)
                carbonLimits.loc[years[1:]] = [carbonMax - reduction * y for y in years[1:]]
            name = str(factor).replace(".", "-")
            carbonLimits.to_csv(f"data/{self.dataset}/systemSpecification/carbonEmissionsLimit_linear_{name}.csv")
        return carbonMin, carbonMax

    def computeCarbonBudget(self, minCarbonEmissions, carbonEmissions, range):
        """generate input files for carbonBudget scenarios"""
        # initial carbon budget
        initialCarbonBudget = carbonEmissions.sum()
        carbonBudget = {"index": "carbonEmissionsBudgetDefault", "value": initialCarbonBudget, "unit": "tons"}
        carbonBudget = pd.DataFrame(carbonBudget, index=[0])
        # carbon emissions target
        carbonLimits            = minCarbonEmissions.tail(1)
        if carbonLimits.iloc[-1] < 0:
            carbonLimits.iloc[-1] = 0
        carbonLimits.name       = "carbonEmissionsLimit"
        carbonLimits.index.name = "time"
        # carbon budget scenarios
        for factor in range:
            name = str(factor).replace(".", "-")
            carbonBudget["value"] = initialCarbonBudget * factor
            carbonBudget.to_csv(f"data/{self.dataset}/systemSpecification/attributes_carbonBudget_{name}.csv", index=False)
            carbonLimits.to_csv(f"data/{self.dataset}/systemSpecification/carbonEmissionsLimit_carbonBudget_{name}.csv", index=True)

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
        capex      = self.getDataframe("capexYearly", ["technology", "capacityType", "location", "year"],subset=subset)
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

    def determineExistingCapacities(self):
        """determine exisiting capacities to investigate how LCOE changes over time"""
        if "linear" not in self.scenario or "carbonBudget" not in self.scenario:
            referenceYear = self.system["referenceYear"]
            builtCapacity = self.getDataframe("builtCapacity", ["technology", "capacityType", "location", "year"]).round(decimals=4)
            builtCapacity = builtCapacity * self.timeStepsInvestDuration
            for set in self.analysis["subsets"]["setTechnologies"]:
                for tech in self.system[set]:
                    # fix design decisions
                    existingCapacity = builtCapacity.loc[tech, :, "power", :]
                    existingCapacity = existingCapacity[existingCapacity>0]
                    existingCapacity = existingCapacity.reorder_levels(["location","year"])
                    # update name and index name
                    existingCapacity.name        = "existingCapacity"
                    if "Transport" in set:
                        existingCapacity.index.names = ["edge", "yearConstruction"]
                    else:
                        existingCapacity.index.names = ["node", "yearConstruction"]
                    # adjust yearly time index
                    existingCapacity                      = existingCapacity.reset_index()
                    existingCapacity["yearConstruction"] +=  referenceYear
                    # save results
                    if not os.path.exists(f"data/{self.dataset}/{set}/{tech}/existingCapacity"):
                        os.mkdir(f"data/{self.dataset}/{set}/{tech}/existingCapacity")
                    existingCapacity.to_csv(f"data/{self.dataset}/{set}/{tech}/existingCapacity/existingCapacity_{self.scenario}.csv", index=False)

    ## method to run visualization
    def run(self, decarbScen = {}):
        """visualize and evaluate results"""

        ## params
        self.evaluateHydrogenDemand()

        ## vars
        # carbon emissions
        self.evaluateCarbonEmissions(decarbScen)

        # installed capacities
        self.evaluateBuiltCapacity()
        # self.evaluateCapacity()

        # carrier flows
        self.evaluateCarrierImportsExports()
        self.evaluateCarrierFlow()

        ## compute levelized cost
        self.computeLevelizedCost("electricity")
        self.computeLevelizedCost("hydrogen")

        ## compute invested capacities
        self.determineExistingCapacities()


if __name__ == "__main__":
    os.chdir("..")
    pltShow = False  # True or False
    outputFolder = "outputs" #
    #outputFolder = "outputs"

    datasets = [0]
    scenarios  = ["", "min_em"] #
    #scenarios  = scenarios + [f"linear_" + str(path).replace(".","-") for path in np.arange(0, 1.2, 0.2).round(2)]


    decarbScen = {"linear":  np.arange(0, 1.1, 0.1).round(2), # np.arange(0, 1.1, 0.1).round(2),
                  #"carbonBudget": np.arange(0.4, 0, -0.1).round(2)
                 }
    for dataset in datasets:
        dataset = f"HSC_NUTS{dataset}"
        for scenario in scenarios:
            visResults = VisualizeResults(dataset, scenario, pltShow=pltShow, outputDir=outputFolder)
            visResults.run(decarbScen = decarbScen)
