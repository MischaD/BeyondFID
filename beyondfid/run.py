import os
import socket
import argparse
import ml_collections
import importlib.util
from beyondfid.log import logger
from beyondfid.feature_computation import compute_features 
from beyondfid.feature_extractor_models import _FEATURE_MODELS
from beyondfid.metrics import log_paths, load_metric, _METRICS
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


def run_generic(pathtrain, pathtest, pathsynth, forward_function, config, output_path, batch_size=256, name="generic"):
    # helper function to run generic models only defined by a forward function. 
    config.feature_extractors.names = name
    config.feature_extractors.generic = ml_collections.ConfigDict()

    # load_feature_model(fe_config) will be called in feature_computation.process - fe_config == config.feature_extractors.generic.config 
    config.feature_extractors.generic.batch_size = batch_size # necessary for all feature extractors
    config.feature_extractors.generic.name = name # necessary for all feature extractors
    config.feature_extractors.generic.config = ml_collections.ConfigDict()
    config.feature_extractors.generic.config.forward = forward_function

    metrics = config.metric_list
    if isinstance(metrics, str):
        metrics = list(metrics.split(","))
    if metrics is None: 
        metrics = []

    hashtrain, hashtest, hashsnth = compute_features(config, pathtrain, pathtest, pathsynth, output_path)

    results = {}
    for metric_name in metrics: 
        logger.info(f"Computing {metric_name}")
        config_metric = config.metrics.get(metric_name)
        config_metric.models = name # overwrite to 
        metric = load_metric(metric_name, config_metric) 
        results[metric_name] = metric.compute_from_path(output_path, hashtrain, hashtest, hashsnth, results_path=None)
    results["paths"] = {"train": hashtrain, "test": hashtest, "snth": hashsnth}
    return results 


def hash_only(hash): 
    return hash.split("_")[-1]  


def run(pathtrain, pathtest, pathsynth, output_path, results_filename, config):
    # precompute features for all models and all data. They will be saved as tensors in output_path with the name being a hash
    metrics = config.metric_list
    if isinstance(metrics, str):
        metrics = list(metrics.split(","))

    hashtrain, hashtest, hashsnth = compute_features(config, pathtrain, pathtest, pathsynth, output_path)
    hashtrain = hash_only(hashtrain) 
    hashtest = hash_only(hashtest) 
    hashsnth = hash_only(hashsnth)

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


def run_model(model, name, config, output_path, pathtrain, pathtest=None, pathsynth=None, batch_size=128, results_filename="results.json"):
    """model with forward function, name for the model and its feature tensors, path to safe tensors to.
    """

    from beyondfid.feature_extractor_models import BaseFeatureModel, register_feature_model
    import torch.nn as nn

    @register_feature_model(name=name)
    class GenericPickleableFeatureModel(BaseFeatureModel, nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def compute_latent(self, x):
            return model(x)

    config.feature_extractors.names =",".join([name,] + config.feature_extractors.names.split(","))
    if pathtest is None: 
        pathtest = pathtrain # will not be computed twic
    if pathsynth is None: 
        pathsynth = pathtrain # will not be computed twic

    setattr(config.feature_extractors, name, ml_collections.ConfigDict())
    
    fe_config = getattr(config.feature_extractors, name)

    # load_feature_model(fe_config) will be called in feature_computation.process - fe_config == config.feature_extractors.name.config 
    fe_config.batch_size = batch_size # necessary for all feature extractors
    fe_config.name = name # necessary for all feature extractors
    fe_config.config = ml_collections.ConfigDict()

    run(pathtrain, pathtest, pathsynth, output_path, results_filename, config)



def get_args():
    parser = argparse.ArgumentParser(description="BeyondFID CLI")
    parser.add_argument("pathtrain", type=str, help="Train data dir or csv with paths to train data. Recursively looks through data dir")
    parser.add_argument("pathtest", type=str, help="Test data dir or csv with paths to test data. Recursively looks through data dir")
    parser.add_argument("pathsynth", type=str, help="Synth data dir or csv with paths to synthetic data. Recursively looks through data dir")

    parser.add_argument("--feature_extractors", type=str, nargs="+", default=[], help="What feature extractors to use. Leave empty to compute all available features.", choices=_FEATURE_MODELS.keys())
    parser.add_argument("--metrics", type=str, nargs="+", default=[], help="What metrics to use. Leave empty to compute all available metrics.", choices=_METRICS.keys())

    parser.add_argument("--config", type=str, default="", help="Configuration file. Defaults all values to config.py. All values set here will be overwritten")
    parser.add_argument("--output_path", type=str, default="resultsbeyondfid", help="Output path to save feature tensors and results.")
    parser.add_argument("--results_filename", type=str, default="results.json", help="Name of file with results. Defaults to <output_path>/results.json")

    # Add an argument for dynamic config updates
    parser.add_argument('--config-update', action=UpdateConfigAction, nargs='+', help="Update config parameters, e.g., --config-update=config.feature_extractors.byol.batch_size=16")
    parser.add_argument('--master_port', type=int, default=12344)
    return parser.parse_args()


def find_free_port(start_port):
    """Finds a free port starting from `start_port`, falling back to any available port."""
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                port += 1  # Try the next port

    # If all ports are occupied, get a random free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def main():
    args = get_args()
    global config
    if args.config != "": 
        config = update_config(config, args.config)
    if args.metrics != []: 
        config.metric_list = ",".join(args.metrics)
    if args.feature_extractors != []: 
        config.feature_extractors.names = ",".join(args.feature_extractors)
        logger.info(f"Overwriting models setting for all metrics and setting it to {config.feature_extractors.names}")
        for metric in _METRICS.keys():
            getattr(config.metrics, metric).models =  config.feature_extractors.names
    config.master_port = find_free_port(args.master_port)
    run(args.pathtrain, args.pathtest, args.pathsynth, args.output_path, args.results_filename, config)


if __name__ == "__main__":
    main()
