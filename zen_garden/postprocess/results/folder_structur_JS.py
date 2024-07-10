"""
:Title:        ZEN-GARDEN folder_structure
:Created:      March-2024
:Authors:      Jara Spate (jspaete@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Used for plotting the results of the optimization model.
Define the folder structure for the figures and get the folder path based on the data in the DataFrame and the dictionary with the title of the plot.
"""
import os

def create_folder(directory):
    '''Create the folder structure for the figures
    :param directory: str - directory to save the figures
    :return: None
    '''
    # Define the folders to create
    #folders = ['Figures', 'Empty', 'Water', 'Electricity', 'Diesel', 'Cost', 'Capacity']
    folders = ['Figures', 'Water', 'Electricity', 'Diesel', 'Capacity']

    # Create the main Figures folder
    path_folder_fig = os.path.join(directory, 'Figures')
    if not os.path.exists(path_folder_fig):
        os.makedirs(path_folder_fig)

    # Create subfolders inside the Figures folder
    for folder_name in folders[1:]:
        path_subfolder = os.path.join(path_folder_fig, folder_name)
        if not os.path.exists(path_subfolder):
            os.makedirs(path_subfolder)




def get_folder_path(df, column_name, dictionary, directory):
    '''
    This function returns the path to the folder where the figure should be saved based on the data in the DataFrame and the dictionary with the title of the plot.
    :param df: DataFrame
    :param column_name: str - name of the column with the data to plot
    :param dictionary: dict - dictionary with the title of the plot
    :param directory: str - directory to save the figures
    :return: str - path to the folder where the figure should be saved'''
    folder_names = {
        # 'empty': 'Empty',
        'water': 'Water',
        'blue_water': 'Water',
        'electricity': 'Electricity',
        'diesel': 'Diesel',
        # 'cost': 'Cost',
        'capacity': 'Capacity'
    }
    path_folder_fig = os.path.join(directory, 'Figures')
    # if df[column_name].empty or all(df[column_name] == 0):
    #     return os.path.join(path_folder_fig, folder_names['empty'])

    title = dictionary.get('title', '').lower()

    # Check for 'capacity' first
    if 'capacity' in title:
        return os.path.join(path_folder_fig, folder_names['capacity'])

    # Check other conditions if 'capacity' is not found
    for key, value in folder_names.items():
        if key in title:
            return os.path.join(path_folder_fig, value)

    # If no specific condition matches, return the default folder path
    return path_folder_fig
