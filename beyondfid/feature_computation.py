import os
import pandas as pd
from beyondfid.feature_extractor_models import load_feature_model
from beyondfid.data.dataloader import get_distributed_dataloader, get_dataloader
from beyondfid.data import get_data 
import torch.multiprocessing as mp
import torch
import socket
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
from beyondfid.log import logger

def find_free_port():
    # Use socket to find a free port dynamically
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def setup(rank, world_size):
    port = find_free_port()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)  # default port
    # Try initializing the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# Example usage

def cleanup():
    dist.destroy_process_group()

def compute(dataloader, model, device):
    latents = []
    indices_list = []
    model.eval()
    with torch.no_grad():
        for images, indices, paths in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            latents.append(outputs)
            indices_list.append(indices)

    latents = np.concatenate(latents, axis=0)
    indices = torch.cat(indices_list).cpu().numpy()
    return (latents, indices)

def process(rank, world_size, basedir, file_list, model, fe_config, return_dict):
    setup(rank, world_size)

    dataloader = get_distributed_dataloader(basedir, file_list, rank, world_size, batch_size=fe_config.batch_size)
       
    device = f"cuda:{rank}"
    model = load_feature_model(fe_config).to(device)

    return_dict[rank] = compute(dataloader, model, device=device)
    cleanup()


def run_compute_features(model, basedir, file_list, fe_config):
    world_size = torch.cuda.device_count()
    mp_manager = mp.Manager()
    return_dict = mp_manager.dict()
    mp.spawn(process, args=(world_size, basedir, file_list, model, fe_config, return_dict), nprocs=world_size, join=True)

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


def run_compute_features_single_gpu(model, basedir, file_list, fe_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_feature_model(fe_config).to(device)

    dataloader = get_dataloader(basedir, file_list, batch_size=fe_config.batch_size)

    latents, indices = compute(dataloader, model, device=device)

    # Sort latents according to indices to get the correct order
    sorted_indices = np.argsort(indices)
    latents = latents[sorted_indices]

    # Convert latents to Tensor
    latents = torch.Tensor(latents)
    return latents



def precompute_features_from_path(config, fe_config, outdir, path, fe_name, split=None):
    # path can be directory containing files or .csv
    if isinstance(path, str):
        file_list_are_paths = True
        basedir = os.path.dirname(path) if not os.path.isdir(path) else path 
    else: 
        file_list_are_paths = False 
        basedir = None
    #os.makedirs(basedir, exist_ok=True)

    # check if real features already computed
    file_list, hash_name = get_data(config, path, fe_name=fe_name, split=split)
    hash_path = os.path.join(outdir, hash_name + ".pt")
    if not os.path.exists(hash_path):
        logger.info(f"Computing features for {fe_name} and saving to {hash_path}")

        if fe_name == "generic":
            # might not be pickleable --> Single-GPU computation
            real_latents = run_compute_features_single_gpu(model=fe_name, basedir=basedir, file_list=file_list, fe_config=fe_config)
        else:
            # Multi-GPU computation
            real_latents = run_compute_features(model=fe_name, basedir=basedir, file_list=file_list, fe_config=fe_config)
        
        # save tensor 
        torch.save(real_latents, hash_path)
        # save list as csv 
        if file_list_are_paths:
            pd.DataFrame({"FileName":file_list}).to_csv(hash_path.rstrip(".pt") + ".csv")
    else: 
        logger.info(f"Precomputed feature tensor already found in: {hash_path}")
    return hash_path


def compute_features(config, pathtrain, pathtest, pathsynth, output_path): 
    feature_extractor_names = config.feature_extractors.names.split(",")

    for feature_extractor_name in feature_extractor_names:
        logger.info(f"Computing features for model: {feature_extractor_name}")
        fe_config = config.feature_extractors.get(feature_extractor_name)
        # prepare out dir
        features_out_dir = os.path.join(output_path, feature_extractor_name)
        os.makedirs(features_out_dir, exist_ok=True)

        logger.info("Precomputing features of train data")
        train_hash = precompute_features_from_path(config, fe_config, features_out_dir, pathtrain, feature_extractor_name, split="TRAIN")

        logger.info("Precomputing features of test data")
        test_hash = precompute_features_from_path(config, fe_config, features_out_dir, pathtest, feature_extractor_name, split="TEST")

        logger.info("Precomputing features of synth data")
        snth_hash = precompute_features_from_path(config, fe_config, features_out_dir, pathsynth, feature_extractor_name)

    # returns only the hashpath of the saved tensor - fullpath is hashdata_{modelname}_<{real/snth}_hash_name>
    train_hash = "_".join(train_hash.split("_")[2:])[:-3]
    test_hash = "_".join(test_hash.split("_")[2:])[:-3]
    snth_hash = "_".join(snth_hash.split("_")[2:])[:-3]
    return train_hash, test_hash, snth_hash
