import os
import argparse
from log import logger
from beyondfid.feature_computation import compute_features 
from beyondfid.metrics.fid import compute_fid
from beyondfid.config import config
from beyondfid.metrics.prdc import compute_prdc


class UpdateConfigAction(argparse.Action):
    """Helper function to overwrite config values"""
    def __call__(self, parser, namespace, values, option_string=None):
        # Iterate over each update string (split by comma)
        if len(values) == 1: 
            updates = values[0].split(",")
        else: 
            updates = values

        for update in updates:
            # Split the argument into the config path and the value
            logger.info(f"Overwriting config: config.{update}")
            config_path, value = update.split("=")
            # Split the config path into keys
            keys = config_path.split(".")
            
            # Access the config object and set the value
            cfg = config
            for key in keys[:-1]:
                cfg = getattr(cfg, key)
            # Cast the value to the same type as the existing value in config
            existing_value = getattr(cfg, keys[-1])
            setattr(cfg, keys[-1], type(existing_value)(value))


def main(args):
    # precompute features for all models and all data. They will be saved as tensors in output_path with the name being a hash
    hashreal, hashsnth = compute_features(config, args.pathreal, args.pathsynth, args.output_path)

    # compute metrics, 
    if "fid" in args.metrics:
        compute_fid(config, args.output_path, args.results_filename, hashreal, hashsnth)

    if "prdc" in args.metrics:
        compute_prdc(config, args.output_path, args.results_filename, hashreal, hashsnth)

    #, args.results_filename


def get_args():
    parser = argparse.ArgumentParser(description="BeyondFID CLI")
    parser.add_argument("pathreal", type=str, help="data dir or csv with paths to data. Recursively looks through data dir")
    parser.add_argument("pathsynth", type=str, help="data dir or csv with paths to synthetic data. Recursively looks through data dir")
    parser.add_argument("--metrics", type=str, default="fid,prdc")
    parser.add_argument("--config", type=str, default="config.py", help="Configuration file. Defaults to config.py")
    parser.add_argument("--output_path", type=str, default="generative_metrics", help="Output path.")
    parser.add_argument("--results_filename", type=str, default="results.json", help="Name of file with results. Defaults to output_path/results.json")

    # Add an argument for dynamic config updates
    parser.add_argument('--config-update', action=UpdateConfigAction, nargs='+', help="Update config parameters, e.g., --config-update=config.feature_extractors.byol.batch_size=16")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.metrics = list(args.metrics.split(","))
    main(args)