from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .datasets import VideoDataset

def get_distributed_dataloader(basedir, filelist, rank, world_size, batch_size, input_size):
    dataset = VideoDataset(filelist, basedir, imagesize=input_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8, prefetch_factor=2)
    return dataloader
