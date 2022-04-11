"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        April-2022
Authors:        Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class containing the unit handling procedure.
==========================================================================================================================================================================="""
import logging
import numpy         as np
import pandas        as pd
from pint                                    import UnitRegistry
from pint.util                               import column_echelon_form

class UnitHandling:
    # unit registry
    ureg = UnitRegistry()

    def __init__(self,paths,roundDecimalPoints):
        """ initialization of the unitHandling instance"""
        self.paths                  = paths
        self.roundingDecimalPoints  = roundDecimalPoints
        self.getBaseUnits()

    def getBaseUnits(self):
        """ gets base units of energy system """
        _listBaseUnits                  = self.extractBaseUnits(self.paths["setScenarios"]["folder"])
        ureg                            = UnitHandling.ureg

        # load additional units
        ureg.load_definitions(self.paths["setScenarios"]["folder"]+"/unitDefinitions.txt")

        # empty base units and dimensionality matrix
        self.baseUnits                  = {}
        self.dimMatrix                  = pd.DataFrame(index=_listBaseUnits).astype(int)
        for _baseUnit in _listBaseUnits:
            dimUnit                     = ureg.get_dimensionality(ureg(_baseUnit))
            self.baseUnits[_baseUnit]   = ureg(_baseUnit).dimensionality
            self.dimMatrix.loc[_baseUnit, list(dimUnit.keys())] = list(dimUnit.values())
        self.dimMatrix                  = self.dimMatrix.fillna(0).astype(int).T

        # check if unit defined twice or more
        _duplicateUnits                 = self.dimMatrix.T.duplicated()
        if _duplicateUnits.any():
            _dimMatrixDuplicate         = self.dimMatrix.loc[:,_duplicateUnits]
            for _duplicate in _dimMatrixDuplicate:
                # if same unit twice (same order of magnitude and same dimensionality)
                if len(self.dimMatrix[_duplicate].shape) > 1:
                    logging.warning(f"The base unit <{_duplicate}> was defined more than once. Duplicates are dropped.")
                    _duplicateDim               = self.dimMatrix[_duplicate].T.drop_duplicates().T
                    self.dimMatrix              = self.dimMatrix.drop(_duplicate,axis=1)
                    self.dimMatrix[_duplicate]  = _duplicateDim
                else:
                    raise KeyError(f"More than one base unit defined for dimensionality {self.baseUnits[_duplicate]} (e.g., {_duplicate})")
        # get linearly dependent units
        M, I, pivot                     = column_echelon_form(np.array(self.dimMatrix), ntype=float)
        M                               = np.array(M).squeeze()
        I                               = np.array(I).squeeze()
        pivot                           = np.array(pivot).squeeze()
        # index of linearly dependent units in M and I
        idxLinDep                       = np.squeeze(np.argwhere(np.all(M==0,axis=1)))
        # index of linearly dependent units in dimensionality matrix
        _idxPivot                           = range(len(self.baseUnits))
        idxLinDepDimMatrix                  = list(set(_idxPivot).difference(pivot))
        self.dimAnalysis                    = {}
        self.dimAnalysis["dependentUnits"]  = self.dimMatrix.columns[idxLinDepDimMatrix]
        dependentDims                       = I[idxLinDep,:]
        # if only one dependent unit
        if len(self.dimAnalysis["dependentUnits"]) == 1:
            dependentDims                   = dependentDims.reshape(1,dependentDims.size)
        # reorder dependent dims to match dependent units
        DimOfDependentUnits                 = dependentDims[:,idxLinDepDimMatrix]
        # if not already in correct order (ones on the diagonal of dependentDims)
        if not np.all(np.diag(DimOfDependentUnits)==1):
            # get position of ones in DimOfDependentUnits
            posOnes         = np.argwhere(DimOfDependentUnits==1)
            assert np.size(posOnes,axis=0) == len(self.dimAnalysis["dependentUnits"]), \
                f"Cannot determine order of dependent base units {self.dimAnalysis['dependentUnits']}, " \
                f"because diagonal of dimensions of the dependent units cannot be determined."
            # pivot dependent dims
            dependentDims   = dependentDims[posOnes[:,1],:]
        self.dimAnalysis["dependentDims"]   = dependentDims
        # check that no base unit can be directly constructed from the others (e.g., GJ from GW and hour)
        assert ~UnitHandling.checkIfPosNegBoolean(dependentDims,axis=1), f"At least one of the base units {list(self.baseUnits.keys())} can be directly constructed from the others"

    def extractBaseUnits(self,folderPath):
        """ extracts base units of energy system
        :param folderPath: path to input files
        :return listBaseUnits: list of base units """
        listBaseUnits = pd.read_csv(folderPath +"/baseUnits.csv").squeeze().values.tolist()
        return listBaseUnits

    def getUnitMultiplier(self,inputUnit):
        """ calculates the multiplier for converting an inputUnit to the base units
        :param inputUnit: string of input unit
        :return multiplier: multiplication factor """
        ureg        = self.ureg
        baseUnits   = self.baseUnits
        dimMatrix   = self.dimMatrix
        # if input unit is already in base units --> the input unit is base unit, multiplier = 1
        if inputUnit in baseUnits:
            return 1
        # if input unit is nan --> dimensionless
        elif type(inputUnit) != str and np.isnan(inputUnit):
            return 1
        else:
            # create dimensionality vector for inputUnit
            dimInput    = ureg.get_dimensionality(ureg(inputUnit))
            dimVector   = pd.Series(index=dimMatrix.index, data=0)
            _missingDim = set(dimInput.keys()).difference(dimVector.keys())
            assert len(_missingDim) == 0, f"No base unit defined for dimensionalities <{_missingDim}>"
            dimVector[list(dimInput.keys())] = list(dimInput.values())
            # calculate dimensionless combined unit (e.g., tons and kilotons)
            combinedUnit = ureg(inputUnit).units
            # if unit (with a different multiplier) is already in base units
            if dimMatrix.isin(dimVector).all(axis=0).any():
                _baseUnit       = ureg(dimMatrix.columns[dimMatrix.isin(dimVector).all(axis=0)][0])
                combinedUnit    *= _baseUnit**(-1)
            # if inverse of unit (with a different multiplier) is already in base units (e.g. 1/km and km)
            elif (dimMatrix*-1).isin(dimVector).all(axis=0).any():
                _baseUnit       = ureg(dimMatrix.columns[(dimMatrix*-1).isin(dimVector).all(axis=0)][0])
                combinedUnit    *= _baseUnit
            else:
                dimAnalysis         = self.dimAnalysis
                # drop dependent units
                dimMatrixReduced    = dimMatrix.drop(dimAnalysis["dependentUnits"],axis=1)
                # solve system of linear equations
                combinationSolution = np.linalg.solve(dimMatrixReduced,dimVector)
                # check if only -1, 0, 1
                if UnitHandling.checkIfPosNegBoolean(combinationSolution):
                    # compose relevant units to dimensionless combined unit
                    for unit,power in zip(dimMatrixReduced.columns,combinationSolution):
                        combinedUnit *= ureg(unit)**(-1*power)
                else:
                    calculatedMultiplier = False
                    for unit, power in zip(dimMatrixReduced.columns, combinationSolution):
                        # try to substitute unit with power > 1 by a dependent unit
                        if np.abs(power) > 1:
                            # iterate through dependent units
                            for dependentUnit,dependentDim in zip(dimAnalysis["dependentUnits"],dimAnalysis["dependentDims"]):
                                idxUnitInMatrixReduced  = list(dimMatrixReduced.columns).index(unit)
                                # if the power of the unit is the same as of the dimensionality in the dependent unit
                                if np.abs(dependentDim[idxUnitInMatrixReduced]) == np.abs(power):
                                    dimMatrixReducedTemp                    = dimMatrixReduced.drop(unit,axis=1)
                                    dimMatrixReducedTemp[dependentUnit]     = dimMatrix[dependentUnit]
                                    combinationSolutionTemp                 = np.linalg.solve(dimMatrixReducedTemp, dimVector)
                                    if UnitHandling.checkIfPosNegBoolean(combinationSolutionTemp):
                                        # compose relevant units to dimensionless combined unit
                                        for unit, power in zip(dimMatrixReducedTemp.columns, combinationSolutionTemp):
                                            combinedUnit        *= ureg(unit) ** (-1 * power)
                                        calculatedMultiplier    = True
                                        break
                    assert calculatedMultiplier, f"Cannot establish base unit conversion for {inputUnit} from base units {baseUnits.keys()}"
            # magnitude of combined unit is multiplier
            multiplier = combinedUnit.to_base_units().magnitude
            # round to decimal points
            return round(multiplier,self.roundingDecimalPoints)

    @staticmethod
    def checkIfPosNegBoolean(array,axis=None):
        """ checks if the array has only positive or negative booleans (-1,0,1)
        :param array: numeric numpy array
        :param axis:
        :return isPosNegBoolean """
        if axis:
            isPosNegBoolean = np.apply_along_axis(lambda row: np.array_equal(np.abs(row), np.abs(row).astype(bool)),1,array).any()
        else:
            isPosNegBoolean = np.array_equal(np.abs(array), np.abs(array).astype(bool))
        return isPosNegBoolean
