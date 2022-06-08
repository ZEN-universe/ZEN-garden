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
import pandas as pd
import matplotlib.pyplot    as plt
from datetime               import datetime


class VisualizeResults:

    def __init__(self, modelName = None):
        """postprocessing of the results of the optimization
        :param model:     optimization model
        :param pyoDict:   input data dictionary
        :param modelName: model name used for the directory to save the results in"""
        self.nameDir = f"../outputs/results{modelName}"
        if not os.path.exists(f"{self.nameDir}/Figures"):
            os.makedirs(f"{self.nameDir}/Figures")
        self.loadResults()
        self.getSets()


    def loadResults(self):
        """ load results from results folder"""
        with open(f"{self.nameDir}/params/paramDict.pickle", "rb") as file:
            self.paramDict = pickle.load(file)
        with open(f"{self.nameDir}/vars/varDict.pickle", "rb") as file:
            self.varDict = pickle.load(file)
        with open(f"{self.nameDir}/System.pickle", "rb") as file:
            self.system = pickle.load(file)
        with open(f"{self.nameDir}/Analysis.pickle", "rb") as file:
            self.analysis = pickle.load(file)

    def getDataframe(self, name, indexNames, type = "var"):
        """plot built capacity"""
        if type == "param":
            df = self.paramDict[name]
        else:
            df = self.varDict[name]

        keys, values   = zip(*df.items())
        if  len(indexNames)>1:
            idx            = pd.MultiIndex.from_tuples(keys)
        else:
            idx            = list(keys)

        df             = pd.Series(values, index=idx)
        df.index.names = indexNames
        df.name        = name

        return df

    def getSets(self):
        """ get sets from system"""
        # carriers
        self.setCarriers = self.system["setCarriers"]
        # conditioning technologies
        self.setConditioningTechnologies = self.system["setConditioningTechnologies"]
        # conversion technologies
        self.setConversionTechnologies   = self.system["setHydrogenConversionTechnologies"]
        self.setConversionTechnologies   = list(set(self.setConversionTechnologies) - set(self.setConditioningTechnologies))
        self.setConversionTechnologies.remove("hydrogen_expansion_high")
        self.setConversionTechnologies.remove("carbon_liquefication")
        self.setConversionTechnologies.remove("carbon_storage")
        # electricity generation Technologies
        self.setElectricityGenerationTechnologies = self.system["setElectricityGenerationTechnologies"]
        # transport technologies
        self.setTransportTechnologies             = self.system["setTransportTechnologies"]
        # storage technologies
        self.setStorageTechnologies               = self.system["setStorageTechnologies"]
        self.setStorageTechnologies.append("carbon_storage")

    def evaluateHydrogenDemand(self):
        """plot hydrogen demand"""
        demand = self.getDataframe("demandCarrier",["carrier", "node", "time"], type="param")
        # total hydrogen demand per country
        demand = demand.loc["hydrogen"].groupby("node").sum()
        demand.plot.bar()
        plt.savefig(f"{self.nameDir}/Figures/totalHydrogenDemand.png")

    def evaluateBuiltCapacity(self):
        """plot built capacity"""
        builtCapacity = self.getDataframe("builtCapacity",["technology", "location", "time"])

        # conversion technologies
        totalBuiltCapacity = builtCapacity.loc[self.setConversionTechnologies].groupby(level=["technology","time"]).sum()
        totalBuiltCapacity.unstack("technology").plot.bar(stacked=True)
        plt.savefig(f"{self.nameDir}/Figures/totalBuiltCapacityConversion.png")

        # electricity generation technologies
        totalBuiltCapacity = builtCapacity.loc[self.setElectricityGenerationTechnologies].groupby(level=["technology", "time"]).sum()
        totalBuiltCapacity.unstack("technology").plot.bar(stacked=True)
        plt.savefig(f"{self.nameDir}/Figures/totalBuiltCapacityElectricity.png")

        # conditioning technologies
        totalBuiltCapacity = builtCapacity.loc[self.setConditioningTechnologies].groupby(level=["technology","time"]).sum()
        totalBuiltCapacity.unstack("technology").plot.bar(stacked=True)
        plt.savefig(f"{self.nameDir}/Figures/totalBuiltCapacityConditioning.png")

        # transport technologies
        totalBuiltCapacity = builtCapacity.loc[self.setTransportTechnologies].groupby(level=["technology","time"]).sum()
        totalBuiltCapacity.unstack("technology").plot.bar(stacked=True)
        plt.savefig(f"{self.nameDir}/Figures/totalBuiltCapacityTransport.png")

        # transport technologies
        totalBuiltCapacity = builtCapacity.loc[self.setStorageTechnologies].groupby(level=["technology","time"]).sum()
        totalBuiltCapacity.plot.bar(stacked=True)
        plt.savefig(f"{self.nameDir}/Figures/totalBuiltCapacityStorage.png")

    def evaluateCapacity(self):
        """plot installed capacity"""
        capacity = self.getDataframe("capacity", ["technology", "location", "time"])

        # conversion technologies
        totalCapacity = capacity.loc[self.setConversionTechnologies].groupby(level=["technology","time"]).sum()
        totalCapacity.unstack("technology").plot.area(stacked=True)
        plt.savefig(f"{self.nameDir}/Figures/totalCapacityConversion.png")

        # electricity generation technologies
        totalCapacity = capacity.loc[self.setElectricityGenerationTechnologies].groupby(level=["technology", "time"]).sum()
        totalCapacity.unstack("technology").plot.area(stacked=True)
        plt.savefig(f"{self.nameDir}/Figures/totalCapacityElectricity.png")

        # transport technologies
        totalCapacity = capacity.loc[self.setTransportTechnologies].groupby(level=["technology","time"]).sum()
        totalCapacity.unstack("technology").plot.area(stacked=True)
        plt.savefig(f"{self.nameDir}/Figures/totalCapacityTransport.png")

        # storage technologies
        totalCapacity = capacity.loc[self.setStorageTechnologies].groupby(level=["technology","time"]).sum()
        totalCapacity.unstack("technology").plot.area(stacked=True)
        plt.savefig(f"{self.nameDir}/Figures/totalCapacityStorage.png")

    def evaluateCarbonEmissions(self):
        """plot carbon emissions"""
        carbonEmissions = self.getDataframe("carbonEmissionsTotal", ["year"])

        carbonEmissions.plot.area(stacked=False)
        plt.savefig(f"{self.nameDir}/Figures/carbonEmissionsYearly.png")

if __name__ == "__main__":
    today      = datetime.now()
    modelName  = "model_" + today.strftime("%Y-%m-%d") + "_perfectForesight"
    visResults = VisualizeResults(modelName)
    ## params
    visResults.evaluateHydrogenDemand()
    ## vars
    visResults.evaluateBuiltCapacity()
    visResults.evaluateCapacity()
    visResults.evaluateCarbonEmissions()
    plt.show()
    a=1