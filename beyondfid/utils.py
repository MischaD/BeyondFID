from importlib.machinery import SourceFileLoader
from beyondfid.data import ALLOWED_EXTENSIONS
import os
import json
from beyondfid.log import logger

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
