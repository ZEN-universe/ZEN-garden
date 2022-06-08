"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      May-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  ETH colors for plots
==========================================================================================================================================================================="""
class ETHColors:

    def __init__(self):
        # load ETH colors
        self.loadColors()
        # set colors
        self.setColors()

    def retrieveColors(self,inputComponents,category):
        assert category in self.colors.keys(),f"category {category} not known. Currently categories {list(self.colors.keys())} available."
        _listColors = []
        if type(inputComponents) == "str":
            _listColors.append(self.retrieveSpecificColor(inputComponents,category))
        else:
            for component in inputComponents:
                _listColors.append(self.retrieveSpecificColor(component,category))
        return _listColors

    def retrieveSpecificColor(self,component,category):
        if component in self.colors[category]:
            _color = self.colors[category][component]
        elif component in self.manualColors:
            _color = self.manualColors[component]
        else:
            print(f"component {component} is neither in colors of category {category} nor in manual colors. Set to blue")
            _color = self.getColor("blue")
        return _color

    def setColors(self):
        self.colors = {}
        self.setColorsCosts()
        self.setColorsTechs()
        self.setColorsScenarios()
        self.setManualColors()

    def setColorsCosts(self):
        self.colors["costs"] = {
            "costTotal": self.getColor("blue"),
            "capexTotal": self.getColor("petrol"),
            "opexTotal": self.getColor("bronze"),
            "costCarrierTotal": self.getColor("red"),
            "costCarbonEmissionsTotal": self.getColor("purple"),
        }

    def setColorsTechs(self):
        self.colors["techs"] = {
            "natural_gas_turbine": self.getColor("petrol"),
            "hard_coal_plant": self.getColor("grey","dark"),
            "natural_gas_turbine_CCS": self.getColor("petrol",60),
            "hard_coal_plant_CCS": self.getColor("grey"),
            "coal": self.getColor("bronze","dark"),
            "nuclear": self.getColor("red",60),
            "lignite_coal_plant": self.getColor("bronze","dark"),
            "oil_plant": self.getColor("grey"),
            "lng_terminal": self.getColor("grey",40),
            "wind_onshore": self.getColor("blue"),
            "wind": self.getColor("blue",60),
            "photovoltaics": self.getColor("bronze",60),
            "wind_offshore": self.getColor("blue","dark"),
            "biomass_plant": self.getColor("green"),
            "biomass_plant_CCS": self.getColor("green",60),
            "CCS": self.getColor("red"),
            "waste_plant": self.getColor("grey",60),
            "run-of-river_hydro": self.getColor("blue",80),
            "reservoir_hydro": self.getColor("blue",60),
            "hydro": self.getColor("blue"),
            "heat_pump": self.getColor("blue"),
            "natural_gas_boiler": self.getColor("petrol",60),
            "oil_boiler": self.getColor("bronze",60),
            "electrode_boiler": self.getColor("blue",60),
            "others": self.getColor("grey",60),
            "battery": self.getColor("bronze",60),
            "hydrogen_storage": self.getColor("purple",60),
            "natural_gas_storage": self.getColor("petrol",60),
            "pumped_hydro": self.getColor("blue",60)
        }

    def setColorsScenarios(self):
        self.colors["scenarios"] = {
            "TIER1a": self.getColor("purple",60),
            "TIER0": self.getColor("petrol",60),
            "TIER1": self.getColor("blue",60),
            "TIER2": self.getColor("green",60),
            "TIER3": self.getColor("bronze",60),
            "TIER4": self.getColor("red",60),
        }

    def setManualColors(self):
        self.manualColors = {}
        self.manualColors["Carbon Emission Budget"] = self.getColor("grey","dark")
        self.manualColors["Carbon Intensity"] = self.getColor("grey", "dark")

    def getColor(self,color,shade = 100):
        assert color in self.baseColors, f"color {color} not in base colors. Select from {list(self.baseColors.keys())}."
        assert shade in self.baseColors[color], f"shade {shade} not in shades of color {color}. Select from {list(self.baseColors[color].keys())}."
        return self.hex2rgb(self.baseColors[color][shade])

    def loadColors(self):
        self.baseColors = {}
        self.baseColors["blue"] = {
            100:    "#215CAF",
            80:     "#4D7DBF",
            60:     "#7A9DCF",
            40:     "#A6BEDF",
            20:     "#D3DEEF",
            10:     "#E9EFF7",
            "dark": "#08407E",
        }
        self.baseColors["petrol"] = {
            100: "#007894",
            80: "#3395AB",
            60: "#66AFC0",
            40: "#99CAD5",
            20: "#CCE4EA",
            10: "#E7F4F7",
            "dark": "#00596D",
        }
        self.baseColors["green"] = {
            100: "#627313",
            80: "#818F42",
            60: "#A1AB71",
            40: "#C0C7A1",
            20: "#E0E3D0",
            10: "#EFF1E7",
            "dark": "#365213",
        }
        self.baseColors["bronze"] = {
            100: "#8E6713",
            80: "#A58542",
            60: "#BBA471",
            40: "#D2C2A1",
            20: "#E8E1D0",
            10: "#F4F0E7",
            "dark": "#956013",
        }
        self.baseColors["red"] = {
            100: "#B7352D",
            80: "#C55D57",
            60: "#D48681",
            40: "#E2AEAB",
            20: "#F1D7D5",
            10: "#F8EBEA",
            "dark": "#96272D",
        }
        self.baseColors["purple"] = {
            100: "#A30774",
            80: "#B73B92",
            60: "#CA6CAE",
            40: "#DC9EC9",
            20: "#EFD0E3",
            10: "#F8E8F3",
            "dark": "#8C0A59",
        }
        self.baseColors["grey"] = {
            100: "#6F6F6F",
            80: "#8C8C8C",
            60: "#A9A9A9",
            40: "#C5C5C5",
            20: "#E2E2E2",
            10: "#F1F1F1",
            "dark": "#575757",
        }

    @staticmethod
    def hex2rgb(hexString, normalized=True):
        if normalized:
            _fac = 255
        else:
            _fac = 1
        hexString = hexString.lstrip('#')
        rgb = tuple(int(hexString[i:i + 2], 16)/_fac for i in (0, 2, 4))
        return rgb
