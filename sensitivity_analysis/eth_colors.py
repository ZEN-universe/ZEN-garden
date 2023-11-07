"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      May-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  ETH colors for plots
==========================================================================================================================================================================="""
import matplotlib.colors as mcolors


class ETHColors:

    def __init__(self):
        # load ETH colors
        self.load_colors()
        # set colors
        self.setColors()

    def retrieve_colors(self, input_components, category):
        assert category in self.colors.keys(), f"category {category} not known. Currently categories {list(self.colors.keys())} available."
        _listColors = []
        if type(input_components) == "str":
            _listColors.append(self.retrieve_specific_color(input_components, category))
        else:
            for component in input_components:
                _listColors.append(self.retrieve_specific_color(component, category))
        return _listColors

    def retrieve_colors_dict(self, input_components, category):
        assert category in self.colors.keys(), f"category {category} not known. Currently categories {list(self.colors.keys())} available."
        _dictColors = dict()
        if type(input_components) == "str":
            _dictColors[input_components] = self.retrieve_specific_color(input_components, category)
        else:
            for component in input_components:
                _dictColors[component] = self.retrieve_specific_color(component, category)
        return _dictColors

    def retrieve_specific_color(self, component, category):
        if component in self.colors[category]:
            _color = self.colors[category][component]
        elif component.replace("_", " ") in self.colors[category]:
            _color = self.colors[category][component.replace("_", " ")]
        elif component.replace(" ", "_") in self.colors[category]:
            _color = self.colors[category][component.replace(" ", "_")]
        elif component in self.manualColors:
            _color = self.manualColors[component]
        else:
            print(
                f"component {component} is neither in colors of category {category} nor in manual colors. Set to blue")
            _color = self.get_color("blue")
        return _color

    def setColors(self):
        self.colors = {}
        self.set_colors_costs()
        self.set_colors_carriers()
        self.set_colors_techs()
        self.set_colors_scenarios()
        self.set_manual_colors()

    def set_colors_costs(self):
        self.colors["carriers"] = {
            "natural gas": self.get_color("blue", 60),
            "dry biomass": self.get_color("green", 40),
            "wet biomass": self.get_color("green", 80),
            "electricity": self.get_color("blue", "dark"),
            "biomethane": self.get_color("green", 80),
        }

    def set_colors_carriers(self):
        self.colors["costs"] = {
            "cost_total": self.get_color("blue"),
            "capexTotal": self.get_color("green"),
            "opexTotal": self.get_color("bronze"),
            "cost_carrierTotal": self.get_color("red"),
            "costCarbonEmissionsTotal": self.get_color("purple"),
        }

    def set_colors_techs(self):
        self.colors["techs"] = {
            # Hydrogen categories
            "H$_\mathrm{2}$ production": self.get_color("blue", 60),
            "H$_\mathrm{2}$ from electricity": self.get_color("bronze", 80),
            "H$_\mathrm{2}$ from natural gas": self.get_color("blue", 40),
            "H$_\mathrm{2}$ from biomass": self.get_color("green", 40),

            "H$_\mathrm{2}$ conditioning": self.get_color("petrol", 40),
            "H$_\mathrm{2}$ transport": self.get_color("petrol", 40),  # self.get_color("petrol", 40),
            "Biomass transport": self.get_color("green", 80),  #
            "Carbon supply chain": self.get_color("grey", 60),

            # hydrogen production technologies
            "electrolysis": self.get_color("bronze", 80),
            "SMR": self.get_color("blue", 40),
            "SMR_CCS": self.get_color("blue", 80),
            "biomethane SMR": self.get_color("petrol", 40),
            "biomethane SMR_CCS": self.get_color("petrol", 80),
            "gasification": self.get_color("green", 40),
            "gasification_CCS": self.get_color("green", 80),
            "anaerobic_digestion": self.get_color("green", 80),
            "carbon_storage": self.get_color("grey", 60),
            "carbon_removal": self.get_color("grey", "dark"),

            # conditioning technologies
            "carbon_liquefication": self.get_color("grey", 60),
            "hydrogen_compressor_high": self.get_color("green", 60),
            "hydrogen_liquefication": self.get_color("green", 20),

            # electricity generation
            "wind_onshore": self.get_color("blue"),
            "wind_offshore": self.get_color("blue", "dark"),
            "pv_rooftop": self.get_color("red", 60),
            "pv_ground": self.get_color("red", 40),

            # transport technologies
            # hydrogen transport
            "hydrogen_truck_gas": self.get_color("blue", 80),
            "hydrogen_train": self.get_color("blue", 60),
            # carbon transport
            "CCS": self.get_color("grey", 60),
            "carbon_truck": self.get_color("grey", 60),
            "carbon_train": self.get_color("grey", 40),
            # electricity transmission
            "electricity_transmission": self.get_color("bronze", 60),
            # other
            "Other": self.get_color("grey", 80)
        }

    def set_colors_scenarios(self):
        self.colors["scenarios"] = {
            # biomass scenarios
            "referenceBM": self.get_color("green", "dark"),
            "reducedBM": self.get_color("green", 80),
            "noBM": self.get_color("green", 40),
            # hydrogen scenarios
            "min": self.get_color("petrol", 40),
            "low": self.get_color("petrol", 100),
            "med": self.get_color("red", 40),
            "high": self.get_color("green", 40),
            "max": self.get_color("green", 100),
        }

    def set_manual_colors(self):
        self.manualColors = {}
        self.manualColors["Carbon Emission Budget"] = self.get_color("grey", "dark")

    def get_color(self, color, shade=100):
        assert color in self.base_colors, f"color {color} not in base colors. Select from {list(self.base_colors.keys())}."
        assert shade in self.base_colors[
            color], f"shade {shade} not in shades of color {color}. Select from {list(self.base_colors[color].keys())}."
        return self.hex_2_rgb(self.base_colors[color][shade])

    def get_custom_colormaps(self, color, diverging=False, reverse=False, skip_white=False):
        """Returns a LinearSegmentedColormap
        :param color_seq: a sequence of floats and RGB-tuples. The floats should be increasing
        and in the interval (0,1).
        :return cmap: returns eth_colors"""
        if diverging:
            colorN = (1, 1, 1)
            color1 = [self.get_color(color[1], shade) for shade in list(self.base_colors[color[1]].keys())]
            color2 = [self.get_color(color[0], shade) for shade in list(self.base_colors[color[0]].keys())]
            color2.reverse()
            color_seq = color1 + [colorN, 0.5, colorN] + color2
        else:
            color_seq = [self.get_color(color, shade) for shade in list(self.base_colors[color].keys())] + [(1, 1, 1)]
        if not reverse:
            color_seq.reverse()
        if skip_white:
            color_seq = [(None,) * 3, 0.0] + list(color_seq) + [1.0, (None,) * 3]
        cdict = {'red': [], 'green': [], 'blue': []}
        for i, item in enumerate(color_seq):
            if isinstance(item, float):
                r1, g1, b1 = color_seq[i - 1]
                r2, g2, b2 = color_seq[i + 1]
                cdict['red'].append([item, r1, r2])
                cdict['green'].append([item, g1, g2])
                cdict['blue'].append([item, b1, b2])
        # LinearSegmentedColormap.from_list("", [(0, "red"), (.1, "violet"), (.5, "blue"), (1.0, "green")])
        return mcolors.LinearSegmentedColormap('CustomMap', cdict)

    def load_colors(self):
        self.base_colors = {}
        self.base_colors["blue"] = {
            "dark": "#08407E",
            100: "#215CAF",
            80: "#4D7DBF",
            60: "#7A9DCF",
            40: "#A6BEDF",
            20: "#D3DEEF",
            10: "#E9EFF7",
        }
        self.base_colors["petrol"] = {
            "dark": "#00596D",
            100: "#007894",
            80: "#3395AB",
            60: "#66AFC0",
            40: "#99CAD5",
            20: "#CCE4EA",
            10: "#E7F4F7",
        }
        self.base_colors["green"] = {
            "dark": "#365213",
            100: "#627313",
            80: "#818F42",
            60: "#A1AB71",
            40: "#C0C7A1",
            20: "#E0E3D0",
            10: "#EFF1E7",
        }
        self.base_colors["bronze"] = {
            "vdark": "#70480E",
            "dark": "#956013",
            100: "#8E6713",
            80: "#A58542",
            60: "#BBA471",
            40: "#D2C2A1",
            20: "#E8E1D0",
            10: "#F4F0E7",
        }
        self.base_colors["red"] = {
            "dark": "#96272D",
            100: "#B7352D",
            80: "#C55D57",
            60: "#D48681",
            40: "#E2AEAB",
            20: "#F1D7D5",
            10: "#F8EBEA",
        }
        self.base_colors["purple"] = {
            "dark": "#8C0A59",
            100: "#A30774",
            80: "#B73B92",
            60: "#CA6CAE",
            40: "#DC9EC9",
            20: "#EFD0E3",
            10: "#F8E8F3",
        }
        self.base_colors["grey"] = {
            "dark": "#575757",
            100: "#6F6F6F",
            80: "#8C8C8C",
            60: "#A9A9A9",
            40: "#C5C5C5",
            20: "#E2E2E2",
            10: "#F1F1F1",
        }
        self.base_colors["PFred"] = {
            "dark": "#58585",
            "light": "#EDB2B2",
        }
        self.base_colors["PFyellow"] = {
            "dark": "#EFDE7B",
            "light": "#F6EEAF",
        }
        self.base_colors["PFgreen"] = {
            "dark": "#80D680",
            "light": "#BBDFBC",
        }
        self.base_colors["PFblue"] = {
            "dark": "#70C6E0",
            "light": "#B4D9E8",
        }
        self.base_colors["PFgrey"] = {
            "1": "#569941",
            # "dark": "#B5B5B5",
            # "light": "#E2E2E2",
            # "-1": "#E1F0DC",
            "-2": "#EDF6EA",
        }

    @staticmethod
    def hex_2_rgb(hex_string, normalized=True):
        if normalized:
            _fac = 255
        else:
            _fac = 1
        hex_string = hex_string.lstrip('#')
        rgb = tuple(int(hex_string[i:i + 2], 16) / _fac for i in (0, 2, 4))
        return rgb
