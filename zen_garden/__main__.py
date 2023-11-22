"""
:Title:        ZEN-GARDEN
:Created:      September-2022
:Authors:      Janis Fluri (janis.fluri@id.ethz.ch),
              Alissa Ganter (aganter@ethz.ch),
              Davide Tonelli (davidetonelli@outlook.com),
              Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Compilation  of the optimization problem.
"""
from ._internal import main
import importlib.util
import argparse
import sys
import os


def run_module(args=None):
    """
    Runs the main function of ZEN-Garden

    :param args: Arguments to parse
    """
    if args is None:
        args = sys.argv[1:]

    # parse the args
    description = "Run ZEN-Garden with a given config file. Per default, the config file will be read out from the " \
                  "current working directory. You can specify a config file with the --config argument. However, " \
                  "note that the output directory will always be the current working directory, independent of the " \
                  "dataset specified in the config file."
    parser = argparse.ArgumentParser(description=description, add_help=True, usage="usage: python -m zen_garden [-h] [--config CONFIG] [--dataset DATASET]")

    parser.add_argument("--config", required=False, type=str, default="./config.py", help="The config file used to run the pipeline, "
                                                                                        "defaults to config.py in the current directory.")
    parser.add_argument("--dataset", required=False, type=str, default=None, help="Path to the dataset used for the run. IMPORTANT: This will overwrite the "
                                                                                  "config.analysis['dataset'] attribute of the config file!")
    parser.add_argument("--job_index", required=False, type=str, default=None, help="A comma separated list (no spaces) of indices of the scenarios to run, if None, all scenarios are run in sequence")
    parser.add_argument("--job_index_var", required=False, type=str, default="SLURM_ARRAY_TASK_ID", help="Try to read out the job index from the environment variable specified here. "
                                                                                                         "If both --job_index and --job_index_var are specified, --job_index will be used.")
    args = parser.parse_args(args)

    # change working directory to the directory of the config file
    config_path, config_file = os.path.split(args.config)
    os.chdir(config_path)

    ### import the config
    spec = importlib.util.spec_from_file_location("module", config_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.config

    ### get the job index
    job_index = args.job_index
    if job_index is None:
        if (job_index := os.environ.get(args.job_index_var)) is not None:
            job_index = int(job_index)
    else:
        job_index = [int(i) for i in job_index.split(",")]

    ### run
    main(config=config, dataset_path=args.dataset, job_index=job_index)


if __name__ == "__main__":
    run_module()
