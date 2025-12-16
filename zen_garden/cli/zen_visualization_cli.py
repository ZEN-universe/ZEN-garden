from zen_temple.main import parse_arguments_and_run

def create_zen_visualization_cli():
    """
    Entry point for the `zen-visualization` command-line interface.

    This function initializes and runs the command-line interface (CLI)
    for the ZEN-garden visualization platform. It delegates argument parsing
    and command execution to the `parse_arguments_and_run` command of 
    the ``ZEN-temple`` package.

    The ``[project.scripts]`` section of the pyproject.toml declares that 
    this function will be called whenever a user enters ``zen-visualization`` 
    into the command prompt. 

    Returns:
        None

    Examples:
        Basic usage in a command line prompt:

        >>> zen-visualization
    """
    parse_arguments_and_run()
