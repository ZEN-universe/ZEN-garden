#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:34:37 2021

@author: Davide Tonelli, PhD candidate UCLouvain
         davide.tonelli@uclouvain.be
         davidetonelli@outlook.com
"""

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

    def columnIndepentData(self, name, headerInSource):

        setName = 'set'+name
        folder = self.mainFolder+setName+'//'
        self.newFolder(folder)

        headerNames = self.analysis['headerDataInputs'][setName]

        data = pd.DataFrame()
        for header in headerNames:
            if header in self.dictionary:
                setattr(self, setName, self.dictionary[header])
            elif header in headerInSource:
                setattr(self, setName, self.dictionary['sourceData'].loc[:, headerInSource[header]])
            else:
                print(f'No input {setName} - none assigned')
                setattr(self, setName, None)
            data[header] = getattr(self, setName)

        data.to_csv(folder+setName+'.csv', header=True, index=None)