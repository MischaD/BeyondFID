from importlib.machinery import SourceFileLoader
from beyondfid.data import ALLOWED_EXTENSIONS
import os

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

