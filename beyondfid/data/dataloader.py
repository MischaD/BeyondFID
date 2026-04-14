from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from beyondfid.data.datasets import GenericDataset, H5Dataset, TensorDataset


def _make_dataset(basedir, filelist):
    """Select the right Dataset class based on basedir and filelist type."""
    if isinstance(basedir, str) and basedir.endswith(".h5"):
        return H5Dataset(basedir)
    if not isinstance(filelist[0], str):
        # in-memory tensors
        return TensorDataset(filelist)
    return GenericDataset(filelist, basedir)


def get_dataloader(basedir, filelist, batch_size, num_workers=8):
    dataset = _make_dataset(basedir, filelist)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def get_distributed_dataloader(basedir, filelist, rank, world_size, batch_size, num_workers=4):
    dataset = _make_dataset(basedir, filelist)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
