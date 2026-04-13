from importlib.machinery import SourceFileLoader
from beyondfid.data import ALLOWED_EXTENSIONS
import os
import json
import socket
from beyondfid.log import logger
import torch
import torch.nn.functional as F


def find_free_port(port=None):
    """Return a free port. If `port` is given and available, use it; otherwise let the OS pick one."""
    if port is not None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                logger.warning(f"Port {port} is already in use, picking a free port automatically.")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def make_exp_config(exp_file):
    if exp_file.endswith(".json"): 
        from utils import make_exp_config, json_to_dict 
        import ml_collections
        config = ml_collections.ConfigDict(json_to_dict(exp_file))
        exp_name = exp_file.split('/')[-1].rstrip('.json')
        config.name = exp_name
        return ml_collections.ConfigDict({"config": config})
        
    # get path to experiment
    exp_name = exp_file.split('/')[-1].rstrip('.py')

    # import experiment configuration
    exp_config = SourceFileLoader(exp_name, exp_file).load_module()
    exp_config.name = exp_name
    return exp_config

def dict_to_json(dct, json_path): 
    with open(json_path, 'w') as json_file:
        json.dump(dct, json_file, indent=4)
    return

def json_to_dict(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def update_config(config, feature_extractors:str, metrics:str): 
    config.metric_list = metrics
    config.feature_extractors.names = feature_extractors
    for metric in config.metrics.keys(): 
        getattr(config.metrics, metric).models = feature_extractors
    return config

class ToTensorIfNotTensor:
    def __call__(self, input):
        if isinstance(input, torch.Tensor):
            return input
        return F.to_tensor(input)
