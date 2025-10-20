import os
import pandas as pd
from beyondfid.feature_extractor_models import load_feature_model
from beyondfid.data.dataloader import get_distributed_dataloader, get_dataloader
from beyondfid.data import get_data 
import torch.multiprocessing as mp
import torch
import socket
import torch.distributed as dist
from beyondfid.data import get_data_csv
from tqdm import tqdm
import numpy as np
from beyondfid.log import logger


def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)  # Assign the free port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def compute(dataloader, model, device):
    # progress on gpu 0 
    def no_progress_bar(x):
        return x
    # Check if distributed training has been initialized
    if dist.is_available() and dist.is_initialized():
        # Only rank 0 should display the progress bar
        progress_bar = no_progress_bar if dist.get_rank() != 0 else tqdm
    else:
        # If dist is not initialized, assume a single GPU and always show progress bar
        progress_bar = tqdm
        
    latents = []
    indices_list = []
    model.eval()
    with torch.no_grad():
        for images, indices, paths in progress_bar(dataloader):
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            latents.append(outputs)
            indices_list.append(indices)

    latents = np.concatenate(latents, axis=0)
    indices = torch.cat(indices_list).cpu().numpy()
    return (latents, indices)

def process(rank, world_size, config, basedir, file_list, model, fe_config, return_dict, master_port=12344):
    setup(rank, world_size, master_port=master_port)

    if config.basedir != "":
        basedir = config.basedir
    dataloader = get_distributed_dataloader(basedir, file_list, rank, world_size, batch_size=fe_config.batch_size, num_workers=config.num_workers)
       
    device = f"cuda:{rank}"
    model = load_feature_model(fe_config).to(device)

    return_dict[rank] = compute(dataloader, model, device=device)
    cleanup()


def run_compute_features(config, model, basedir, file_list, fe_config):
    world_size = torch.cuda.device_count()
    mp_manager = mp.Manager()
    return_dict = mp_manager.dict()
    master_port = config.master_port
    mp.spawn(process, args=(world_size, config, basedir, file_list, model, fe_config, return_dict, master_port), nprocs=world_size, join=True)

    # Combine the latents from all processes
    all_latents = []
    all_indices = []

    for rank in range(world_size):
        latents, indices = return_dict[rank]
        all_latents.append(latents)
        all_indices.append(indices)

    all_latents = np.concatenate(all_latents, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)

    # Sort latents according to indices to get the correct order
    sorted_indices = np.argsort(all_indices)
    all_latents = all_latents[sorted_indices]

    latents = torch.Tensor(all_latents)
    return latents


def run_compute_features_single_gpu(config, model, basedir, file_list, fe_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_feature_model(fe_config).to(device)

    if config.basedir != "":
        basedir = config.basedir
    dataloader = get_dataloader(basedir, file_list, batch_size=fe_config.batch_size)

    latents, indices = compute(dataloader, model, device=device)

    # Sort latents according to indices to get the correct order
    sorted_indices = np.argsort(indices)
    latents = latents[sorted_indices]

    # Convert latents to Tensor
    latents = torch.Tensor(latents)
    return latents

def precompute_features_from_tensor(config, fe_config, outdir, path, fe_name, split=None):
    # check if real features already computed
    images = torch.load(path)
    file_list_path = path.replace(".pt", ".csv")
    if not os.path.exists(file_list_path): 
        raise ValueError("To compute on a dataset consisting of one big tensor you need to prove the filelist which has the same name but .csv as ending")
    basedir = path.replace(".pt", "")

    file_list_csv = pd.read_csv(file_list_path)

    _, hash_name = get_data_csv(file_list_path, fe_name=fe_name, split=split)
    #file_list = imgdict
    file_list = images 
    hash_path = os.path.join(outdir, hash_name + ".pt")

    if not os.path.exists(hash_path):
        logger.info(f"Computing features for {fe_name} and saving to {hash_path}")
        if fe_name == "generic":
            # might not be pickleable --> Single-GPU computation
            real_latents = run_compute_features_single_gpu(config=config, model=fe_name, basedir=basedir, file_list=file_list, fe_config=fe_config)
        else:
            # Multi-GPU computation
            real_latents = run_compute_features(config=config, model=fe_name, basedir=basedir, file_list=file_list, fe_config=fe_config)
        
        # save tensor 
        torch.save(real_latents, hash_path)
        # save list as csv 
        file_list_csv.to_csv(hash_path.replace(".pt",".csv"))
    else: 
        logger.info(f"Precomputed feature tensor already found in: {hash_path}")
    return hash_name  + ".pt"


def precompute_features_from_path(config, fe_config, outdir, path, fe_name, split=None, force_recompute=False):
    """
    Precompute features from a given path and store them in the specified output directory.

    This function processes data from the provided path, which can either be a directory 
    containing files or a CSV file. The features are computed based on the configurations 
    provided (`config` and `fe_config`) and are saved in the `outdir` directory with the 
    specified feature extractor name (`fe_name`). The filename is a hash that depends on 
    the input filelist and will be retturned. 

    Parameters:
    -----------
    config : dict
        Configuration dictionary with general settings for feature computation.
    
    fe_config : dict
        Feature extraction configuration, containing parameters for the feature extraction process.
    
    outdir : str
        The path to the directory where the precomputed features will be saved.
    
    path : str
        The path to the data source, which can be either a directory containing files or a .csv file.
    
    fe_name : str
        Name of the feature extractor to be used, which determines the method for feature computation.
    
    split : str, optional
        Data split identifier (e.g., 'TRAIN', 'VAL', 'TEST'). Default is None.

    Returns:
    --------
    None
        hashname of the dataset. 
    """
    # path can be directory containing files or .csv, or large tenors 
    if isinstance(path, str):
        if path.endswith(".pt"): 
            return precompute_features_from_tensor(path, fe_config, outdir, path, fe_name, split=None)
        file_list_are_paths = True
        basedir = os.path.dirname(path) if not os.path.isdir(path) else path 
    else: 
        file_list_are_paths = False 
        basedir = None
        if hasattr(config, "overwrite_basedir"):
            basedir = config.overwrite_basedir
            
    #os.makedirs(basedir, exist_ok=True)

    # check if real features already computed
    file_list, hash_name = get_data(config, path, fe_name=fe_name, split=split)
    hash_path = os.path.join(outdir, hash_name + ".pt")
    if not os.path.exists(hash_path) or force_recompute:
        logger.info(f"Computing features for {fe_name} and saving to {hash_path}")

        if fe_name == "generic":
            # might not be pickleable --> Single-GPU computation
            real_latents = run_compute_features_single_gpu(config=config, model=fe_name, basedir=basedir, file_list=file_list, fe_config=fe_config)
        else:
            # Multi-GPU computation
            real_latents = run_compute_features(config=config, model=fe_name, basedir=basedir, file_list=file_list, fe_config=fe_config)
        
        # save tensor 
        torch.save(real_latents, hash_path)
        # save list as csv 
        if file_list_are_paths:
            pd.DataFrame({"FileName":file_list}).to_csv(hash_path.rstrip(".pt") + ".csv")
    else: 
        logger.info(f"Precomputed feature tensor already found in: {hash_path}")
    return hash_name  + ".pt"


def compute_features(config, pathtrain, pathtest, pathsynth, output_path, features_out_dir=None): 
    feature_extractor_names = config.feature_extractors.names.split(",")
    feature_extractor_names = [x for x in feature_extractor_names if x != ""]

    for feature_extractor_name in feature_extractor_names:
        logger.info(f"Computing features for model: {feature_extractor_name}")
        fe_config = config.feature_extractors.get(feature_extractor_name)

        # prepare out dir
        if features_out_dir is None: 
            features_out_dir_fe = os.path.join(output_path, feature_extractor_name)
        else: 
            features_out_dir_fe = features_out_dir
            
        os.makedirs(features_out_dir_fe, exist_ok=True)

        logger.info("Precomputing features of train data")
        train_hash = precompute_features_from_path(config, fe_config, features_out_dir_fe, pathtrain, feature_extractor_name, split="TRAIN")

        logger.info("Precomputing features of test data")
        test_hash = precompute_features_from_path(config, fe_config, features_out_dir_fe, pathtest, feature_extractor_name, split="TEST")

        if pathsynth != pathtrain: 
            logger.info("Precomputing features of synth data")
            snth_hash = precompute_features_from_path(config, fe_config, features_out_dir_fe, pathsynth, feature_extractor_name, force_recompute=config.feature_extractors.always_overwrite_snth)
        else: 
            logger.info("Skipping synth features as they have the same path as train")
            snth_hash = train_hash
    # returns only the hashpath of the saved tensor - fullpath is hashdata_{modelname}_<{real/snth}_hash_name>
    train_hash = "_".join(train_hash.split("_")[2:])[:-3]
    test_hash = "_".join(test_hash.split("_")[2:])[:-3]
    snth_hash = "_".join(snth_hash.split("_")[2:])[:-3]
    return train_hash, test_hash, snth_hash
