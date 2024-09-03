from utils import json_to_dict, dict_to_json
from beyondfid.log import logger
import torch
import math
import os


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


def path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth):
    path_train = os.path.join(output_path, model, f"hashdata_{model}_{hashtrain}.pt")
    path_test = os.path.join(output_path, model, f"hashdata_{model}_{hashtest}.pt")
    path_snth = os.path.join(output_path, model, f"hashdata_{model}_{hashsnth}.pt")
    train = torch.load(path_train)
    test = torch.load(path_test)
    snth = torch.load(path_snth)
    return train, test, snth


def compute_authpct(config, output_path, results_filename, hashtrain, hashtest, hashsnth):
    from fld.metrics.AuthPct import AuthPct

    for model in config.metrics.fid.model.split(","):
        train, test, snth = path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)
        authpct = AuthPct().compute_metric(train, test, snth)

        save_metric(os.path.join(output_path, results_filename), model=model, key=f"authpct", value=authpct)
        

def compute_cttest(config, output_path, results_filename, hashtrain, hashtest, hashsnth):
    from fld.metrics.CTTest import CTTest
    for model in config.metrics.fid.model.split(","):
        train, test, snth = path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)
        try: 
            cttest = CTTest().compute_metric(train, test, snth)
        except Exception as e: 
            logger.warning(f"cttest for model {model} failed due to {e}\nSetting cttest for model {model} to -1")
            #e.g. Cell 0 lacks test samples and/or training samples. Consider reducing the number of cells in partition.
            cttest = -1 

        save_metric(os.path.join(output_path, results_filename), model=model, key=f"cttest", value=cttest)


def compute_fld(config, output_path, results_filename, hashtrain, hashtest, hashsnth):
    from fld.metrics.FLD import FLD 
    for model in config.metrics.fid.model.split(","):
        train, test, snth = path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)
        test_fld = FLD(eval_feat="test").compute_metric(train, test, snth)
        train_fld = FLD(eval_feat="train").compute_metric(train, test, snth)

        if math.isnan(train_fld): 
            logger.warning(f"Train FLD is NaN for model {model} - setting it to -1")
            train_fld = -1

        if math.isnan(test_fld): 
            logger.warning(f"Test FLD is NaN for model {model} - setting it to -1")
            test_fld = -1

        save_metric(os.path.join(output_path, results_filename), model=model, key=f"fld_train", value=train_fld)
        save_metric(os.path.join(output_path, results_filename), model=model, key=f"fld_test", value=test_fld)


def compute_kid(config, output_path, results_filename, hashtrain, hashtest, hashsnth):
    from fld.metrics.KID import KID 
    for model in config.metrics.fid.model.split(","):
        train, test, snth = path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)
        test_kid = KID(ref_feat="test").compute_metric(None, test, snth)
        train_kid = KID(ref_feat="train").compute_metric(train, None, snth)

        save_metric(os.path.join(output_path, results_filename), model=model, key=f"kid_train", value=train_kid)
        save_metric(os.path.join(output_path, results_filename), model=model, key=f"kid_test", value=test_kid)
