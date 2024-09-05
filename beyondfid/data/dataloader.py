from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from beyondfid.data.datasets import GenericDataset, TensorDataset


def get_dataloader(basedir, filelist, batch_size):
    if not isinstance(filelist[0], str): 
        # filelist is list of tensor etc. 
        dataset = TensorDataset(filelist)
    else: 
        dataset = GenericDataset(filelist, basedir)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, prefetch_factor=2)
    return dataloader

def get_distributed_dataloader(basedir, filelist, rank, world_size, batch_size):
    if not isinstance(filelist[0], str): 
        # filelist is list of tensor etc. 
        dataset = TensorDataset(filelist)
    else: 
        dataset = GenericDataset(filelist, basedir)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8, prefetch_factor=2)
    return dataloader
