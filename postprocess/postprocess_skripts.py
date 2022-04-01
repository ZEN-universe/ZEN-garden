import pandas as pd
from model.optimization_setup import OptimizationSetup
from data.NUTS0_CCTS import config

# Ignore this line:
if False:
    optimizationSetup = OptimizationSetup(config.analysis, prepare)

installTechnology = pd.Series(optimizationSetup.model.installTechnology.extract_values())
demandCarrier = pd.Series(optimizationSetup.model.demandCarrier.extract_values())
varSeries = dict()
for key in evaluation.varDict.keys():
    varSeries.update({key: pd.Series(optimizationSetup.model.component(key).extract_values())})
