"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class used to create the results dictionary to visualize the results on the dashboard.
==========================================================================================================================================================================="""
from datetime import datetime

class DashboardDictionary:

    def __init__(self, pyoDict, modelName = None, author = None, description = None):
        """create results dictionary for dashboard to visualize results
        :param modelName: name of the model
        :param pyoDict:   dictionary containing input data and results"""

        self.dashboardDict = dict()
        # recommended
        self.dashboardDict['title'] = modelName
        self.dasboardDict['uid']    = ''
        self.dasboardDict['geonet'] = 'alpha0.1'
        # optional
        self.dashboardDict['author']     = author
        self.dasboardDict['description'] = description
        self.dashboardDict['created']    = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

        self.inputData = pyoDict[None]
        self.results   = pyoDict['varDict']

        self.addNodes()
        self.addEdges()
        self.addSettings()


    def editDictionary(self):
        """ """
        self.dashboardDict['edited'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    def addNodes(self):
        """ """

        node = dict()
        keys = [
            # required
            "label",       # string providing a human readable label for the node
            "coordinates", # coordinates MUST be provided for each node in the WGS 1984 coordinate system
            # recommended
            "id",          # string providing a unique identifier for a node
            # optional
            "class",       # string assigning one or more class names to a node
            "from",        # datetime used to indicate the beginning of the period of activity of a node
            "to",          # datetime used to indicate the end of the period of activity of a node
            "properties",  # object containing additional static information about the node
            "states"       # array of state objects containing time dependent information about the state of node
            ]

        self.dashboardDict['nodes'] = list()
        for node in self.inputData['setNodes'][None]:
            # required
            node['label']       = node
            node['coordinates'] = list(self.inputData['coordinates'][node])
            # recommended
            node['id'] = node
            # optional
            states = list()
            for var in ['varlist']:
                states.append(self.addStates(var))
            node['state'] = states


            # use states to describe what is going on, and when
            State: [{"from": "1994-11-05T08:15:30-05:00",
                        "to": "1994-11-05T09:15:30-05:00",
                        "production": 3e5, }]

            self.dashboardDict['nodes'].append(node)

    def addStates(self, var):

        state = dict()
        for t in self.inputData['setTimeSteps'][None]:
            state['from'] = 1
            state['to'] = t+1
            state[var] = self.results

    def addEdges(self):
        Edges = [
            {"source": 1, "target": "rotterdam", "from": "1994-11-05T08:15:30-05:00", "to": "", "id": "",
             "properties": {"type": "truck"},
             "path": {"type": "custom", "points": [[1.5, 4.5]]},
             "states": [{"from": "", "to": "", "flow": 555}]
             }

    def addSettings(self):
            Settings = {
                "cursor": "1994-11-05T08:15:30-05:00",
                "viewbox": [3.0, 5.1, 4.2, 6.3],
                "styles": [
                    {"match": "edge[type='truck'][]", "fill": "blue", "stroke": "black"}
                ],
                "sizes": {
                    "node_prop": "emissions",
                    "node_range": [1, 5],
                    "node_scale": "linear",
                    "node_defaut": 3,
                    "edge_default": 2
                }
            }
            }




