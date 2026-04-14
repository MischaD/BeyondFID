import hashlib
import os
import pandas as pd
from beyondfid.log import logger

ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.JPEG', '.pt', '.mp4', '.avi', '.h5')


def hash_dataset_path(dataset_root_dir, img_list, descriptor=""):
    """Takes a list of paths and joins it to a large string - then uses it as hash input stringuses it filename for the entire datsets for quicker loading"""
    name = "".join([x for x in img_list])
    name = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return os.path.join(dataset_root_dir, "hashdata_" + descriptor + "_" + name)


def get_data_csv(path, fe_name, config=None, split=None):
    data_csv = pd.read_csv(path)
    if split is not None:
        data_csv = data_csv[data_csv["Split"]==split]
    filename_key = getattr(config, "filename_key", "FileName")
    file_list = list(data_csv[filename_key])
    output_filename = hash_dataset_path(os.path.dirname(path), img_list=file_list, descriptor=fe_name)
    return file_list, os.path.basename(output_filename)


def get_data_h5(path, fe_name, dataset_key="images"):
    """Return (None, hash_name) for a standalone .h5 file — all images are used."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for H5 support. Install with: pip install h5py")
    with h5py.File(path, "r") as f:
        if dataset_key not in f:
            available = list(f.keys())
            dataset_key = available[0]
            logger.warning(
                f"H5 key 'images' not found in {path}. "
                f"Using '{dataset_key}'. Available keys: {available}"
            )
        n = len(f[dataset_key])
    file_size = os.path.getsize(path)
    hash_input = f"{os.path.abspath(path)}:{file_size}:{dataset_key}:{n}"
    hash_name = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()
    logger.info(f"{n} images found in H5 file {path} (key='{dataset_key}')")
    return None, f"hashdata_{fe_name}_{hash_name}"


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

def get_data_from_list(out_path, file_list, fe_name): 
    file_list = sorted(file_list)
    logger.info(f"{len(file_list)} files found in list")
    output_filename = hash_dataset_path(out_path, img_list=file_list, descriptor=fe_name)
    return file_list, output_filename
 


def get_data(config, path, fe_name, split):
    """Returns (file_list, hash_name) for the given path.

    Supported path types:
      - dict  : in-memory image tensor dict
      - list  : in-memory file/tensor list
      - .csv  : CSV with 'FileName' and optionally 'Split' columns
      - .h5   : HDF5 file — all images are always used (split is ignored)
      - folder: directory searched recursively for images
    """
    if isinstance(path, dict):
        file_key, img_list = next(iter(path.items()))
        return img_list, "hashdata_" + fe_name + "_" + f"{file_key}"

    elif isinstance(path, list):
        out_path = config.get("generic_out_path", ".")
        return get_data_from_list(out_path, path, fe_name)

    if path.endswith(".csv"):
        return get_data_csv(path, fe_name, config, split)
    elif path.endswith(".h5"):
        dataset_key = getattr(config, "h5_dataset_key", "images")
        return get_data_h5(path, fe_name, dataset_key=dataset_key)
    else:
        return get_data_from_folder(path, fe_name, config)
