import pandas as pd
import geopandas as gpd
from model.optimization_setup import OptimizationSetup
from data.NUTS0_CCTS import config

# Ignore this line:
if False:
    optimizationSetup = OptimizationSetup(config.analysis, prepare)


varSeries = dict()
for key in evaluation.varDict.keys():
    varSeries.update({key: pd.Series(optimizationSetup.model.component(key).extract_values())})

paramSeries = dict()
for key in evaluation.paramDict.keys():
    paramSeries.update({key: pd.Series(optimizationSetup.model.component(key).extract_values())})


