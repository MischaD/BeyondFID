import os
import argparse
import ml_collections
import importlib.util
from beyondfid.log import logger
from beyondfid.feature_computation import compute_features 
from beyondfid.metrics import log_paths, load_metric
from beyondfid.default_config import config
from beyondfid.utils import json_to_dict


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


def update_config(cfg, new_config):
    """
    Update the original configuration (cfg) with values from a new configuration file.

    Args:
        cfg (ml_collections.ConfigDict): The original configuration object.
        config_path (str or ml_collections.ConfigDict): The path to the new configuration file or new config file.

    Returns:
        ml_collections.ConfigDict: The updated configuration object.
    """
    if isinstance(new_config, str):
        # Load the new configuration from the provided path
        spec = importlib.util.spec_from_file_location("new_config", new_config)
        new_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(new_config_module)
        
        # Get the new configuration dictionary, excluding special attributes
        new_config = {k: v for k, v in new_config_module.__dict__.items() if not k.startswith('__')}

    def recursive_update(d, u):
        """Recursively update a ConfigDict or dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) or isinstance(v, ml_collections.ConfigDict):
                d[k] = recursive_update(d.get(k, ml_collections.ConfigDict()), v)
            else:
                d[k] = v
        return d
    
    # Update the original config with the new config
    updated_config = recursive_update(cfg, new_config)

    return updated_config


def run(pathtrain, pathtest, pathsynth, metrics, output_path, results_filename, config):
    # precompute features for all models and all data. They will be saved as tensors in output_path with the name being a hash
    if isinstance(metrics, str):
        metrics = list(metrics.split(","))

    hashtrain, hashtest, hashsnth = compute_features(config, pathtrain, pathtest, pathsynth, output_path)
    res_path = os.path.join(output_path, results_filename)

    logger.info(f"Computing metrics. Saving results to {res_path}")
    # log hash paths to results file
    log_paths(output_path, results_filename, hashtrain, hashtest, hashsnth)

    for metric_name in metrics: 
        logger.info(f"Computing {metric_name}")
        config_metric = config.metrics.get(metric_name)
        metric = load_metric(metric_name, config_metric) 
        metric.compute_from_path(output_path, hashtrain, hashtest, hashsnth, results_path=results_filename)

    return json_to_dict(res_path)


def get_args():
    parser = argparse.ArgumentParser(description="BeyondFID CLI")
    parser.add_argument("pathtrain", type=str, help="Train data dir or csv with paths to train data. Recursively looks through data dir")
    parser.add_argument("pathtest", type=str, help="Test data dir or csv with paths to test data. Recursively looks through data dir")
    parser.add_argument("pathsynth", type=str, help="Synth data dir or csv with paths to synthetic data. Recursively looks through data dir")
    parser.add_argument("--metrics", type=str, default="prdc,fid,is_score,cttest,authpct,fld,kid")
    parser.add_argument("--config", type=str, default="", help="Configuration file. Defaults all values to config.py. All values set here will be overwritten")
    parser.add_argument("--output_path", type=str, default="generative_metrics", help="Output path.")
    parser.add_argument("--results_filename", type=str, default="results.json", help="Name of file with results. Defaults to output_path/results.json")

    # Add an argument for dynamic config updates
    parser.add_argument('--config-update', action=UpdateConfigAction, nargs='+', help="Update config parameters, e.g., --config-update=config.feature_extractors.byol.batch_size=16")
    return parser.parse_args()


def main():
    args = get_args()
    global config
    if args.config != "": 
        config = update_config(config, args.config)
    run(args.pathtrain, args.pathtest, args.pathsynth, args.metrics, args.output_path, args.results_filename, config)


if __name__ == "__main__":
    main()
