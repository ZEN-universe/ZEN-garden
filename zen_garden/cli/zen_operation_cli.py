from zen_garden.cli.zen_garden_cli import build_parser, resolve_job_index
from zen_garden.wrapper.operation_scenarios import operation_scenarios
import argparse

def build_parser_op() -> argparse.ArgumentParser:

    # load parser from zen-garden
    parser = build_parser()
    
    parser.add_argument(
        "--config_op",
        required=False,
        type=str,
        default=None,
        help=
        "The config file used to run the operation-only model, defaults to " \
        " --config."
    )
    parser.add_argument(
        "--dataset_op",
        required=False,
        type=str,
        default=None,
        help=
        "Name of the dataset used for the operation-only runs. The outputs " \
        "will be saved under this dataset name"
    )
    parser.add_argument(
        "--scenarios_op",
        required=False,
        type=str,
        default=None,
        help=
        "Path to the scenarios.json file used in the operation-only runs. " \
        "Defaults to the scenarios.json file from --dataset"
    )
    parser.add_argument(
        "--delete_data",
        action="store_true",
        help=
        "Deletes the created operation-only models upon termination to avoid " \
        "cluttering the data directory"
    )

    # add parser description
    parser.description = "Run ZEN garden with a given config file. Per default, the" \
                  "config file will be read out from the current working " \
                  "directory. You can specify a config file with the --config "\
                  "argument. However, note that the output directory will " \
                  "always be the current working directory, independent of " \
                  "the dataset specified in the config file."
    
    parser.usage = "usage: python -m zen_garden.wrapper.operational_scenarios [-h] " \
        "[--config CONFIG] [--dataset DATASET] [--folder_output FOLDER_OUTPUT] "\
        "[--job_index JOB_INDEX] [--job_index_var JOB_INDEX_VAR] "\
        "[--scenarios_op SCENARIOS_OP] " \
        "[--delete_data] [--use_existing]"
    
    return parser


def create_zen_operation_cli() -> None:

    # create parser and parse command line argument
    parser = build_parser_op()
    args = parser.parse_args()

    # Make dataset a required argument
    if args.dataset is None:
        raise argparse.ArgumentError(
            "Missing required argument --dataset."
        ) 
      
    # Resolve job index
    job_index = resolve_job_index(args.job_index, args.job_index_var)

    # run operation scenarios
    operation_scenarios(
        config=args.config,
        dataset = args.dataset,
        folder_output=args.folder_output,
        job_index=job_index,
        scenarios_op=args.scenarios_op,
        delete_data=args.delete_data,
    )

if __name__ == "__main__":
    
    create_zen_operation_cli()