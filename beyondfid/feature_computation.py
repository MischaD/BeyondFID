import os
import pandas as pd
from beyondfid.feature_extractor_models import load_feature_model
from beyondfid.data.dataloader import get_distributed_dataloader 
from beyondfid.data import get_data 
from log import logger
import torch.multiprocessing as mp
import torch
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
from log import logger


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def process(rank, world_size, basedir, file_list, model, fe_config, return_dict):
    setup(rank, world_size)

    dataloader = get_distributed_dataloader(basedir, file_list, rank, world_size, batch_size=fe_config.batch_size, input_size=fe_config.input_size)
    model = model.to(f"cuda:{rank}")

    latents = []
    indices_list = []
    model.eval()
    with torch.no_grad():
        for images, indices, paths in tqdm(dataloader):
            images = images.to(rank)
            outputs = model(images).cpu().numpy()
            latents.append(outputs)
            indices_list.append(indices)

    latents = np.concatenate(latents, axis=0)
    indices = torch.cat(indices_list).cpu().numpy()

    return_dict[rank] = (latents, indices)
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


def compute_features(config, pathreal, pathsynth, output_path): 
    feature_extractor_names = config.feature_extractors.names.split(",")

    for feature_extractor_name in feature_extractor_names:
        logger.info(f"Computing features for model: {feature_extractor_name}")
        fe_config = config.feature_extractors.get(feature_extractor_name)
        # prepare out dir
        features_out_dir = os.path.join(output_path, feature_extractor_name)
        os.makedirs(features_out_dir, exist_ok=True)

        model = load_feature_model(fe_config)

        basedir_real = os.path.dirname(pathreal) if not os.path.isdir(pathreal) else pathreal
        basedir_snth = os.path.dirname(pathsynth) if not os.path.isdir(pathsynth) else pathsynth 
        os.makedirs(basedir_real, exist_ok=True)
        os.makedirs(basedir_snth, exist_ok=True)

        # check if real features already computed
        real_file_list, real_hash_name = get_data(config, pathreal, fe_name=feature_extractor_name)
        real_hash_path = os.path.join(features_out_dir, real_hash_name + ".pt")
        if not os.path.exists(real_hash_path):
            logger.info(f"Computing real features for {feature_extractor_name} and saving to {real_hash_path}")
            # compute
            real_latents = run_compute_features(model=model, basedir=basedir_real, file_list=real_file_list, fe_config=fe_config)
            # save tensor 
            torch.save(real_latents, real_hash_path)
            # save list as csv 
            pd.DataFrame({"FileName":real_file_list}).to_csv(real_hash_path.rstrip(".pt") + ".csv")
        else: 
            logger.info(f"Precomputed feature tensor already found in: {real_hash_path}")
            real_latents = torch.load(real_hash_path)

        # check if synthetic features already computed
        snth_file_list, snth_hash_name = get_data(config, pathsynth, fe_name=feature_extractor_name)
        snth_hash_path = os.path.join(features_out_dir, snth_hash_name + ".pt")
        if not os.path.exists(snth_hash_path):
            logger.info(f"Computing synthetic features for {feature_extractor_name} and saving to {snth_hash_path}")
            # compute
            snth_latents = run_compute_features(model=model, basedir=basedir_snth, file_list=snth_file_list, fe_config=fe_config)
            # save tensor 
            torch.save(snth_latents, snth_hash_path)
            # save list as csv 
            pd.DataFrame({"FileName":snth_file_list}).to_csv(snth_hash_path.rstrip(".pt") + ".csv")
        else: 
            logger.info(f"Precomputed feature tensor already found in: {snth_hash_path}")
            snth_latents = torch.load(snth_hash_path)


