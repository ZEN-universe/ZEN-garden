"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      September-2022
Authors:      Janis Fluri (janis.fluri@id.ethz.ch)
              Alissa Ganter (aganter@ethz.ch)
              Davide Tonelli (davidetonelli@outlook.com)
              Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Compilation  of the optimization problem.
==========================================================================================================================================================================="""
from ._internal import compile
import importlib.util
import argparse
import sys

def main(args=None):
    """
    main function of ZEN-Garden
    :param args: Arguments to parse
    """
    if args is None:
        args = sys.argv[1:]

    # parse the args
    description = "Run ZEN-Garden with a given config file. Per default, the config file will be read out from the " \
                  "current working directory. You can specify a config file with the --config argument. However, " \
                  "note that the output directory will always be the current working directory, independent of the " \
                  "dataset specified in the config file."
    parser = argparse.ArgumentParser(description=description, add_help=True,
                                     usage="usage: python -m zen_garden [-h] [--config CONFIG] [--dataset DATASET]")

    parser.add_argument("--config", required=False, type=str, default="config.py",
                        help="The config file used to run the pipeline, "
                             "defaults to config.py in the current directory.")
    parser.add_argument("--dataset", required=False, type=str, default=None,
                        help="Path to the dataset used for the run. IMPORTANT: This will overwrite the "
                             "config.analysis['dataset'] attribute of the config file!")
    args = parser.parse_args(args)

    ### import the config
    spec = importlib.util.spec_from_file_location("module", args.config)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.config

    ### run
    compile(config=config, dataset_path=args.dataset)

if __name__ == "__main__":
    main()
