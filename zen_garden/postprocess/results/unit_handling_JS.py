"""
:Title:        ZEN-GARDEN unit_handling
:Created:      May-2024
:Authors:      Jara Spate (jspaete@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

This module defines a class for managing unit definitions in an energy system.
The class reads unit definitions from a CSV file and provides methods to create
dictionaries for technology capacities and carriers.
"""
import os
import pandas as pd
import pint


class EnergySystemUnits:
    """
    This class reads and manages the units for an energy system.
    """

    def __init__(self, path_energy_system):
        """
        Initializes the EnergySystemUnits class with a given path.

        :param path_energy_system: Path to the energy system directory.
        """
        self.path_energy_system = path_energy_system
        self.unit_power, self.unit_energy, self.unit_water, self.unit_water_power = self._define_units()


    @classmethod
    def convert_to_cubic_meters(self, unit_str):
        ureg = pint.UnitRegistry()
        # Define conversion factors for different volume units
        conversion_factors = {
            'kilowatervolumen': 1000,  # Assume 1 kilowatervolumen = 1000 cubic meters
            'gigawatervolumen': 1e9,    # Assume 1 gigawatervolumen = 1 billion cubic meters
            'megawatervolumen': 1e6,    # Assume 1 megawatervolumen = 1 million cubic meters
            'watervolumen': 1           # Assume 1 watervolumen = 1 cubic meter
        }
        # Split the unit string to extract the numerical value and the unit type
        parts = unit_str.split('*')
        if len(parts) == 2:
            value_str, unit_type = parts
            value = float(value_str)
        else:
            unit_type = parts[0]
            value = 1  # Default value if not specified in the string

        # Get the conversion factor for the unit type
        conversion_factor = conversion_factors.get(unit_type.strip(), 1)  # Default to 1 if unit type not found

        # Convert the value to cubic meters
        value_cubic_meters = value * conversion_factor
        output_unit = str(value_cubic_meters) + '*meter**3'

        return output_unit

    def _define_units(self):
        """
        Load the units from the base_units.csv file and define the units for power, energy, and water.

        :return: The units for power, energy, and water.
        """
        current_path = os.path.dirname(os.path.abspath(__file__))
        print(current_path)

        base_units = pd.read_csv("base_units.csv")
        unit_power = str(base_units['unit'].iloc[1])
        unit_energy = unit_power + 'h'

        # unit water: replace watervolumen with m3
        unit_water = str(base_units['unit'].iloc[5])
        unit_water = self.convert_to_cubic_meters(unit_water)
        unit_water_power = unit_water + '/h'

        return unit_power, unit_energy, unit_water, unit_water_power

    def create_unit_dictionary_capacity(self):
        """
        Create a dictionary containing the units for the capacity of each technology type.

        :return: The dictionary containing the units for the capacity of each technology type.
        """
        tech_capacity_unit_mapping = {
            ('PV', 'power'): self.unit_power,
            ('power_line', 'power'): self.unit_power,
            ('battery', 'energy'): self.unit_energy,
            ('battery', 'power'): self.unit_power,
            ('diesel_WP', 'power'): self.unit_energy,
            ('el_WP', 'power'): self.unit_power,
            ('water_storage', 'energy'): self.unit_water,
            ('water_storage', 'power'): self.unit_water_power,
            ('irrigation_sys', 'power'): self.unit_water_power,
            ('irrigation_sys', 'energy'): self.unit_water
        }

        return tech_capacity_unit_mapping

    def create_unit_dictionary_carrier(self):
        """
        Create a dictionary containing the units for each carrier.

        :return: The dictionary containing the units for each carrier.
        """
        unit_mapping_carrier = {
            'water': self.unit_water,
            'electricity': self.unit_energy,
            'diesel': self.unit_energy,
            'irrigation_water': self.unit_water,
            'blue_water': self.unit_water
        }

        return unit_mapping_carrier
