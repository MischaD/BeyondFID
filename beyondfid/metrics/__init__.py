from abc import ABC, abstractmethod
from beyondfid.utils import json_to_dict, dict_to_json
from beyondfid.log import logger
import torch
import math
import os


_METRICS = {}

class BaseMetric(ABC):
    def __init__(self, config):
        self.config = config

    def path_to_tensor(self, output_path, model, hashtrain, hashtest, hashsnth):
        path_train = os.path.join(output_path, model, f"hashdata_{model}_{hashtrain}.pt")
        path_test = os.path.join(output_path, model, f"hashdata_{model}_{hashtest}.pt")
        path_snth = os.path.join(output_path, model, f"hashdata_{model}_{hashsnth}.pt")
        train = torch.load(path_train)
        test = torch.load(path_test)
        snth = torch.load(path_snth)
        return train, test, snth

    def set_models(self, new_models: list): 
        if isinstance(new_models, list): 
            new_models = ",".join(new_models)
        self.config.models = new_models

    @property
    def models(self): 
        return self.config.models.split(",")

    @abstractmethod
    def compute(self, train, test, synth):
        # tensor of features --> dict with metric
        pass

    @abstractmethod
    def compute_from_path(self, output_path, results_path, hashtrain, hashtest, hashsnth):
        # 
        pass


def register_metric(cls=None, *, name=None):
    def wrapper(cls):
        metric_name = name if name else cls.__name__
        _METRICS[metric_name] = cls
        return cls
    if cls is None:
        return wrapper
    return wrapper(cls)


def load_metric(metric_name, config):
    if metric_name not in _METRICS:
        raise ValueError(f"No metric found with name {metric_name}")
    
    metric_class = _METRICS[metric_name]
    return metric_class(config)


def save_metric(results_path, model, key, value): 
    """Save metric but check first if key already exists"""
    if os.path.exists(results_path):
        results = json_to_dict(results_path)
    else: 
        results = {}

    if results.get(model) is None: 
        results[model] = {}

    results[model][key] = value
    dict_to_json(results, results_path)


def log_paths(output_path, results_filename, hashtrain, hashtest, hashsnth):
    res_path = os.path.join(output_path, results_filename)
    if os.path.exists(res_path):
        results = json_to_dict(res_path)
    else: 
        results = {}

    results["_train"] = hashtrain
    results["_test"] = hashtest 
    results["_synth"] = hashsnth 
    dict_to_json(results, res_path)



from beyondfid.metrics.inception_score import ISScore
from beyondfid.metrics.fid import FID
from beyondfid.metrics.prdc import PRDCMetric
from beyondfid.metrics.authpct import AuthPctMetric
from beyondfid.metrics.cttest import CTTestMetric
from beyondfid.metrics.fld import FLDMetric
from beyondfid.metrics.kid import KIDMetric
from beyondfid.metrics.vendi import VendiMetric 