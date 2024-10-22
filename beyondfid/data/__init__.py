import hashlib
import os
import pandas as pd
from beyondfid.log import logger

ALLOWED_EXTENSIONS = ('.png', '.jpg', '.pt', '.mp4', '.avi')


def hash_dataset_path(dataset_root_dir, img_list, descriptor=""):
    """Takes a list of paths and joins it to a large string - then uses it as hash input stringuses it filename for the entire datsets for quicker loading"""
    name = "".join([x for x in img_list])
    name = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return os.path.join(dataset_root_dir, "hashdata_" + descriptor + "_" + name)


def get_data_csv(path, fe_name, config=None, split=None):
    data_csv = pd.read_csv(path)
    if split is not None: 
        data_csv = data_csv[data_csv["Split"]==split]
    file_list = list(data_csv["FileName"])
    output_filename = hash_dataset_path(os.path.dirname(path),img_list=file_list, descriptor=fe_name)
    return file_list, os.path.basename(output_filename)


def get_data_from_folder(path, fe_name, config=None):
    """
    Recursively searches the specified directory for files with specific extensions
    and returns a list of their relative paths.

    This function traverses through the directory tree starting from the given `path`,
    identifying files that match any of the allowed extensions: `.png`, `.jpg`, `.pt`,
    `.mp4`, and `.avi`. The returned paths are relative to the provided `path` parameter.

    Parameters:
    -----------
    config : dict
        A configuration dictionary (currently unused in this function but reserved for future use).
    
    path : str
        The base directory path from which the search should begin. The function will
        look for files in this directory and all its subdirectories.
    
    fe_name : str
        A feature name string (currently unused in this function but reserved for future use).

    Returns:
    --------
    list of str
        A list of relative paths to files that have one of the specified extensions.
        Each path is relative to the `path` parameter provided.
    """
    
    # Walk through the directory
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ALLOWED_EXTENSIONS):
                # Get the relative file path
                relative_path = os.path.relpath(os.path.join(root, file), path)
                file_list.append(relative_path)
    
    file_list = sorted(file_list)
    output_filename = hash_dataset_path(os.path.dirname(path),img_list=file_list, descriptor=fe_name)
    logger.info(f"{len(file_list)} files found for path {path}")
    return file_list, os.path.basename(output_filename)


def get_data(config, path, fe_name, split):
    """Returns list of files in path. Path can be csv or folder that will be searched recursively. 
    Also computes the hash for the output tensor.
    """
    if isinstance(path, dict): 
        # path is dict containing a list of images already loaded. 
        file_key, img_list = next(iter(path.items()))
        return img_list, "hashdata_" + fe_name + "_" + f"{file_key}" 

    if path.endswith(".csv"):
        return get_data_csv(path, fe_name, config, split)
    else: 
        return get_data_from_folder(path, fe_name, config)
