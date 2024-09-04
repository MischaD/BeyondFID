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



@register_metric(name="authpct")
class AuthPctMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
    
    def compute(self, train, test, snth):
        from fld.metrics.AuthPct import AuthPct
        authpct = AuthPct().compute_metric(train, test, snth)
        return {"authpct": authpct}

    def compute_from_path(self, output_path, hashtrain, hashtest, hashsnth, results_path=None):
        for model in self.models:
            train, test, snth = self.path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)
            metrics = self.compute(train, test, snth)
            if results_path is not None: 
                for key, value in metrics.items():
                    save_metric(os.path.join(output_path, results_path), model=model, key=key, value=value)


@register_metric(name="cttest")
class CTTestMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
    
    def compute(self, train, test, snth):
        from fld.metrics.CTTest import CTTest
        try:
            cttest = CTTest().compute_metric(train, test, snth)
        except Exception as e:
            logger.warning(f"cttest failed due to {e}\nSetting cttest to -1")
            cttest = -1
        return {"cttest": cttest}

    def compute_from_path(self, output_path, hashtrain, hashtest, hashsnth, results_path=None):
        for model in self.models:
            train, test, snth = self.path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)
            metrics = self.compute(train, test, snth)
            if results_path is not None: 
                for key, value in metrics.items():
                    save_metric(os.path.join(output_path, results_path), model=model, key=key, value=value)


@register_metric(name="fld")
class FLDMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
    
    def compute(self, train, test, snth):
        from fld.metrics.FLD import FLD
        test_fld = FLD(eval_feat="test").compute_metric(train, test, snth)
        train_fld = FLD(eval_feat="train").compute_metric(train, test, snth)

        if math.isnan(train_fld):
            logger.warning(f"Train FLD is NaN - setting it to -1")
            train_fld = -1
        if math.isnan(test_fld):
            logger.warning(f"Test FLD is NaN - setting it to -1")
            test_fld = -1

        return {"fld_train": train_fld, "fld_test": test_fld}

    def compute_from_path(self, output_path, hashtrain, hashtest, hashsnth, results_path=None):
        for model in self.models:
            train, test, snth = self.path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)
            metrics = self.compute(train, test, snth)
            if results_path is not None: 
                for key, value in metrics.items():
                    save_metric(os.path.join(output_path, results_path), model=model, key=key, value=value)


@register_metric(name="kid")
class KIDMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
    
    def compute(self, train, test, snth):
        from fld.metrics.KID import KID
        test_kid = KID(ref_feat="test").compute_metric(None, test, snth)
        train_kid = KID(ref_feat="train").compute_metric(train, None, snth)

        return {"kid_train": train_kid, "kid_test": test_kid}

    def compute_from_path(self, output_path, hashtrain, hashtest, hashsnth, results_path=None):
        for model in self.models:
            train, test, snth = self.path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)
            metrics = self.compute(train, test, snth)
            if results_path is not None: 
                for key, value in metrics.items():
                    save_metric(os.path.join(output_path, results_path), model=model, key=key, value=value)



from beyondfid.metrics.inception_score import ISScore
from beyondfid.metrics.fid import FID
from beyondfid.metrics.prdc import PRDCMetric