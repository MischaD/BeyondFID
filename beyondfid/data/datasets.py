import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.io as io

def first_frame(x):
    """Take first frame of video as reference frame"""
    return x[0]


def load_video_as_tensor(video_path):
    video_tensor, _, _ = io.read_video(video_path, pts_unit='sec')
    video_tensor = video_tensor.float() / 255.0
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    return video_tensor


class VideoDataset(Dataset):
    def __init__(self, file_list, basedir, imagesize):
        self.basedir = basedir
        self.file_list = file_list
        self.transform = transforms.Compose([
            transforms.Resize((imagesize, imagesize)),
            transforms.CenterCrop(imagesize),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        vid_path = self.file_list[idx]
        video = load_video_as_tensor(os.path.join(self.basedir, vid_path))
        frame = first_frame(video)
        frame = self.transform(frame)
        return frame, idx, vid_path  # Return index to maintain order

