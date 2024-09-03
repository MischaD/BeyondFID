import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.io as io
import torch
from PIL import Image
from beyondfid.log import logger


def first_frame(x):
    """Take first frame of video as reference frame"""
    return x[0]


def load_video_as_tensor(video_path):
    video_tensor, _, _ = io.read_video(video_path, pts_unit='sec')
    video_tensor = video_tensor.float() / 255.0
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    return video_tensor


class GenericDataset(Dataset):
    def __init__(self, file_list, basedir):
        self.basedir = basedir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        file_ending = path.split(".")[-1]
        if file_ending in ["mp4", "avi"] : 
            video = load_video_as_tensor(os.path.join(self.basedir, path))
            frame = first_frame(video)
        elif file_ending == ".pt": 
            tensor = torch.load(os.path.join(self.basedir, path))
            if tensor.min() < 0 or tensor.max() > 1:
                logger.warning(f"Tensor {os.path.join(self.basedir, path)} not 0-1 normalized.") 
            if tensor.dim() == 4:
                frame = first_frame(tensor)
            else: 
                frame = tensor
        else:
            # Load and normalize an image
            image_path = os.path.join(self.basedir, path)
            frame = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
            frame = transforms.ToTensor()(frame)

        return frame, idx, path  # Return index to maintain order

