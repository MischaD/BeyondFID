import os
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
    if metrics is None : 
        return {"train": hashtrain, "test": hashtest, "snth": hashsnth}

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



_FEATURE_MODEL_DESCRIPTIONS = {
    "inception":     "InceptionV3 — standard backbone for FID/IS (ImageNet)",
    "swav":          "ResNet-50 trained with SwAV self-supervised contrastive learning (recommended for IRS)",
    "dinov2":        "DINOv2 ViT-B/14 self-supervised transformer (Facebook)",
    "convnext":      "ConvNeXt-XL trained on ImageNet-21k (timm)",
    "mae":           "Masked Autoencoder ViT-L/16 (Facebook)",
    "data2vec":      "data2vec ViT-L vision transformer (HuggingFace checkpoint)",
    "byol":          "ResNet-50 trained with BYOL (local checkpoint required)",
    "sdvae":         "Stable Diffusion VAE latent encoder (requires diffusers)",
    "clip":          "OpenAI CLIP ViT-B/32 (optional — install separately)",
    "cxr":           "CheXNet ResNet-50 for chest X-ray images (domain-specific)",
    "flatten":       "Raw pixel values flattened to a 1-D vector (baseline)",
    "flatten_resize":"Raw pixel values after resize to 224×224, then flattened (baseline)",
    "generic":       "Custom feature extractor defined by a user-supplied forward function",
}

_METRIC_DESCRIPTIONS = {
    "irs":       "IRS — Image Retrieval Score: measures diversity / memorisation (CVPR 2025)",
    "fid":       "FID — Fréchet Inception Distance: fidelity + diversity (lower is better)",
    "kid":       "KID — Kernel Inception Distance: unbiased FID alternative",
    "fld":       "FLD — Fréchet LPIPS Distance (requires optional fld package)",
    "prdc":      "Precision, Recall, Density, Coverage: separate fidelity / diversity scores",
    "is_score":  "Inception Score: quality and variety via InceptionV3 class predictions",
    "authpct":   "AuthPct: fraction of synthetic samples not memorised from training data",
    "cttest":    "C2ST: two-sample classifier test for distribution matching",
    "diversity": "Diversity: intra-set feature distance (Rényi entropy of α=2,4)",
    "vendi":     "Vendi Score: diversity via matrix-rank of the kernel similarity matrix",
}


def get_args():
    feature_extractor_help = "\n".join(
        f"  {name:<16} {desc}" for name, desc in _FEATURE_MODEL_DESCRIPTIONS.items()
        if name in _FEATURE_MODELS
    )
    metric_help = "\n".join(
        f"  {name:<12} {desc}" for name, desc in _METRIC_DESCRIPTIONS.items()
        if name in _METRICS
    )

    description = (
        "BeyondFID — evaluation toolkit for unconditional image generation.\n\n"
        "Computes feature-based generative metrics (FID, KID, IRS, Precision/Recall, …)\n"
        "from three image datasets: a training set, a held-out test set, and a synthetic set.\n"
        "Features are cached on disk keyed by a hash of the file list, so re-running with\n"
        "different metrics on the same data skips the expensive feature extraction step.\n\n"
        "Paper: https://openaccess.thecvf.com/content/CVPR2025/html/"
        "Dombrowski_Image_Generation_Diversity_Issues_and_How_to_Tame_Them_CVPR_2025_paper.html"
    )

    epilog = (
        "Available feature extractors:\n"
        f"{feature_extractor_help}\n\n"
        "Available metrics:\n"
        f"{metric_help}\n\n"
        "Examples:\n"
        "  # IRS with the recommended SwAV backbone (fastest, most reliable)\n"
        "  beyondfid data/train data/test data/synth --feature_extractors swav --metrics irs\n\n"
        "  # FID + KID with Inception features (classic setup)\n"
        "  beyondfid data/train data/test data/synth --feature_extractors inception --metrics fid kid\n\n"
        "  # Multiple extractors and metrics in one pass (features computed once, reused per metric)\n"
        "  beyondfid data/train data/test data/synth \\\n"
        "      --feature_extractors swav inception dinov2 \\\n"
        "      --metrics irs fid prdc\n\n"
        "  # Override a single config value on the command line\n"
        "  beyondfid data/train data/test data/synth \\\n"
        "      --config-update metrics.prdc.nearest_k=3\n\n"
        "  # Use a custom config file (see beyondfid/default_config.py for the format)\n"
        "  beyondfid data/train data/test data/synth --config my_config.py\n\n"
        "Dataset formats accepted for each path argument:\n"
        "  folder   Plain directory — all images found recursively\n"
        "  .csv     CSV with columns 'FileName' and 'Split' (TRAIN / VAL / TEST)\n"
        "  .pt      Pre-saved torch tensor (3-D = image, 4-D = video)\n"
        "  .h5      HDF5 file — all images are always used (no split filtering)\n\n"
        "HDF5 usage:\n"
        "  beyondfid train.h5 test.h5 synth.h5 --feature_extractors swav --metrics irs\n\n"
        "  # Override the HDF5 dataset key (default: 'images')\n"
        "  beyondfid train.h5 test.h5 synth.h5 --config-update h5_dataset_key=data\n"
    )

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "pathtrain",
        type=str,
        help="Real training data: folder, .csv file, or .pt tensor.",
    )
    parser.add_argument(
        "pathtest",
        type=str,
        help=(
            "Real held-out test data: folder, .csv file, or .pt tensor. "
            "Used as the reference distribution for IRS and as a second anchor for FLD. "
            "Can be the same path as pathtrain when a separate test split is unavailable."
        ),
    )
    parser.add_argument(
        "pathsynth",
        type=str,
        help="Synthetic data to evaluate: folder, .csv file, or .pt tensor.",
    )
    parser.add_argument(
        "--feature_extractors",
        type=str,
        nargs="+",
        default=[],
        choices=_FEATURE_MODELS.keys(),
        metavar="EXTRACTOR",
        help=(
            "One or more feature extractors to use. Features are computed once and reused "
            "across all requested metrics. Defaults to the extractors listed in default_config.py "
            "(inception, byol). Run with -h to see all available extractors and their descriptions."
        ),
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=[],
        choices=_METRICS.keys(),
        metavar="METRIC",
        help=(
            "One or more metrics to compute. Defaults to the metric list in default_config.py. "
            "Run with -h to see all available metrics and their descriptions."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help=(
            "Path to a Python config file that overrides values in default_config.py. "
            "See beyondfid/default_config.py for the expected format and all available keys."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="resultsbeyondfid",
        help=(
            "Directory where cached feature tensors and the results file are saved. "
            "Features are keyed by a hash of the file list and reused on subsequent runs. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--results_filename",
        type=str,
        default="results.json",
        help="Name of the JSON file written inside --output_path. Default: %(default)s",
    )
    parser.add_argument(
        "--config-update",
        action=UpdateConfigAction,
        nargs="+",
        metavar="KEY=VALUE",
        help=(
            "Override individual config values without a full config file. "
            "Use dot notation matching the structure of default_config.py. "
            "Example: --config-update metrics.prdc.nearest_k=3 feature_extractors.inception.batch_size=32"
        ),
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=None,
        help=(
            "Port used by torch.distributed for multi-GPU communication. "
            "A free port is chosen automatically if this is omitted or if the given port is occupied."
        ),
    )
    return parser.parse_args()


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
    if args.master_port is not None:
        from beyondfid.utils import find_free_port
        config.master_port = find_free_port(args.master_port)
    run(args.pathtrain, args.pathtest, args.pathsynth, args.output_path, args.results_filename, config)


if __name__ == "__main__":
    main()
