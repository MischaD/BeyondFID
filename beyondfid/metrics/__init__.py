from utils import json_to_dict, dict_to_json
import os

def save_metric(results_path, key, value): 
    """Save metric but check first if key already exists"""
    if os.path.exists(results_path):
        results = json_to_dict(results_path)
    else: 
        results = {}

    results[key] = value 
    dict_to_json(results, results_path)
