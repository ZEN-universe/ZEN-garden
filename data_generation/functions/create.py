"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      January-2022
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Collection of methods to generate dataset respecting the platform's input standard.
==========================================================================================================================================================================="""

import pandas as pd
import numpy as np
import os

def createFolder(path, folderName):
    """
    Method to create new folders if not already existing 
    """
    try:
        os.makedirs(path+folderName)
    except OSError:
        pass

class Create:
    
    def __init__(self, dictionary, analysis):
        
        self.pathMainFolder = '..//..//data//'
        self.dictionary = dictionary
        self.analysis = analysis

        # create main folder
        self.mainFolder = '..//data//'+self.dictionary['mainFolder']+'//'
        self.newFolder(self.mainFolder)

    def newFolder(self, folder):

        try:
            os.mkdir(folder)
        except OSError:
            pass

    def independentData(self, name, headerInSource):
        """
        Method to create dataframe composed by independent columns. E.g. dataframe containing time-steps array
        """

        setName = 'set'+name
        folder = self.mainFolder+setName+'//'
        self.newFolder(folder)

        # list of file names also appearing in the column of the file
        headerNames = self.analysis['headerDataInputs'][setName]

        data = pd.DataFrame()
        for header in headerNames:
            # check if a value has been manually entered
            if header in self.dictionary:
                setattr(self, header, self.dictionary[header])
            # check if a dictionary has been entered to map the header to the entry of an external file
            elif header in headerInSource:
                setattr(self, header, self.dictionary['sourceData'].loc[:, headerInSource[header]].values)
            # assign a default value
            else:
                print(f'No input {setName} - 0 assigned')
                setattr(self, header, [0])
            data[header] = getattr(self, header)

        data.to_csv(folder+setName+'.csv', header=True, index=None)

    def indexedData(self, name, headerInSource):
        """
        Method to create dataframe composed by a set of columns which determine the index of the last column.
        """
        setName = 'set'+name
        folder = self.mainFolder+setName+'//'
        self.newFolder(folder)

        scenarioHeader = self.analysis['headerDataInputs']['setScenarios'][0]
        nodeHeader = self.analysis['headerDataInputs']['setNodes'][0]
        timeHeader = self.analysis['headerDataInputs']['setTimeSteps'][0]

        # list of file names also appearing in the column of the file - exclude parameters related to a distance
        headerNames = [name for name in self.analysis['headerDataInputs'][setName] if 'distance' not in name.lower()]
        for header in headerNames:
            for element in headerInSource:
                # create subfolder
                folder = self.mainFolder + setName + '//' + element + '//'
                self.newFolder(folder)
                # name in fixed input dictionary
                fixedInputKey = header+'_'+element
                if fixedInputKey in self.dictionary:
                    values = self.dictionary[fixedInputKey]
                elif header in headerInSource[element]:
                    values = self.dictionary['sourceData'].loc[:, headerInSource[element][header]]
                else:
                    print(f'No input {fixedInputKey} - 0 assigned')
                    values = [0]

                data = pd.DataFrame(columns=[scenarioHeader, timeHeader, nodeHeader, header])
                for scenario in getattr(self, self.analysis['headerDataInputs']['setScenarios'][0]):
                    for timeStep in getattr(self, self.analysis['headerDataInputs']['setTimeSteps'][0]):
                        for node in getattr(self, self.analysis['headerDataInputs']['setNodes'][0]):
                            if len(values) == 1:
                                valuesToAdd = {scenarioHeader: scenario, timeHeader: timeStep, nodeHeader:node,
                                               header:values[0]}
                            else:
                                valuesToAdd = {scenarioHeader: scenario, timeHeader: timeStep, nodeHeader:node,
                                               header:values[list(getattr(self, nodeHeader)).index(node)]}
                            data = data.append(valuesToAdd, ignore_index=True)

                data.to_csv(folder+header+'.csv', header=True, index=None)

    def distanceData(self, name, headerInSource):
        """
        Method to fill the datasets of the parameters related to the distance between two nodes.
        """

        setName = 'set'+name
        folder = self.mainFolder+setName+'//'
        self.newFolder(folder)

        nodes = getattr(self, self.analysis['headerDataInputs']['setNodes'][0])
        for element in headerInSource:
            folder = self.mainFolder + setName + '//' + element + '//'
            data = pd.DataFrame(columns=nodes, index=nodes)

            # list of file names also appearing in the column of the file - consider only parameters related to a distance
            headerNames = [name for name in self.analysis['headerDataInputs'][setName] if 'distance' in name.lower()]
            for header in headerNames:
                for element in headerInSource:
                    # create subfolder
                    folder = self.mainFolder + setName + '//' + element + '//'
                    self.newFolder(folder)
                    # name in fixed input dictionary
                    fixedInputKey = header + '_' + element
                    if fixedInputKey in self.dictionary:
                        values = self.dictionary[fixedInputKey]
                    elif header in headerInSource[element]:
                        values = self.dictionary['sourceData'].loc[:, headerInSource[element][header]]
                    else:
                        print(f'No input {fixedInputKey} - 0 assigned')
                        values = [0]

                    if len(values) == 1:
                        data.loc[:, :] = values[0]
                    else:
                        data.loc[:,:] = values

                    data.to_csv(folder+header+'.csv')

    def newFiles(self, name, headerInSource, newFiles):
        """
        Method to generate new empty files with indexqes and columns specified in the input dictionary.
        """

        setName = 'set'+name
        folder = self.mainFolder+setName+'//'
        self.newFolder(folder)

        for element in headerInSource:
            folder = self.mainFolder + setName + '//' + element + '//'
            for fileName in newFiles:
                data = pd.DataFrame(columns = newFiles[fileName]['columns'],
                                    index = newFiles[fileName]['index'])
                data.to_csv(folder + fileName + '.csv')

    def distanceMatrix(self, distanceType='eucledian'):
        """
        Compute a matrix containing the distance between any two points in the domain based on the Euclidean distance
        """

        def f_eucl_dist(P0, P1):
            """
                Compute the Eucledian distance of two points in 2D
            """
            return ((P0[0] - P1[0]) ** 2 + (P0[1] - P1[1]) ** 2) ** 0.5

        N = getattr(self, self.analysis['headerDataInputs']['setNodes'][0]).size
        xArr = getattr(self, self.analysis['headerDataInputs']['setNodes'][1])
        yArr = getattr(self, self.analysis['headerDataInputs']['setNodes'][2])
        dist = np.zeros([N, N])

        for idx0 in np.arange(N):
            for idx1 in np.arange(N):
                P0 = (xArr[idx0], yArr[idx0])
                P1 = (xArr[idx1], yArr[idx1])

                if distanceType == 'eucledian':
                    dist[idx0, idx1] = f_eucl_dist(P0, P1)
                else:
                    raise "Distance type not implemented"

        self.eucledian_distance = dist