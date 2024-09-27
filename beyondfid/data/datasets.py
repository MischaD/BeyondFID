import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.io as io
import torch
from PIL import Image
from beyondfid.log import logger
import cv2


def first_frame(x):
    """Take first frame of video as reference frame"""
    return x[0]


def load_video_as_tensor(video_path):
    video_tensor, _, _ = io.read_video(video_path, pts_unit='sec')
    video_tensor = video_tensor.float() / 255.0
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    video_tensor = transforms.Resize(512)(video_tensor)
    return video_tensor


# Function to load first n frames from the video
def load_first_n_frames(video_path, n_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(n_frames):
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Failed to load any frames from video: {video_path}")
    
    return frames

# Function to resize the frames to 512x512 and convert them to tensors
def process_frames(frames, size=(512, 512)):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts frame to tensor with shape [C, H, W]
        transforms.Resize(size),
    ])
    tensor_frames = [transform(frame) for frame in frames]
    return torch.stack(tensor_frames)  # Shape [n_frames, C, H, W]


class GenericDataset(Dataset):
    def __init__(self, file_list, basedir, n_frames=1):
        super().__init__()
        self.basedir = basedir
        self.file_list = file_list
        self.n_frames = n_frames

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        file_ending = path.split(".")[-1]
        if file_ending in ["mp4", "avi"] : 
            video_path = os.path.join(self.basedir, path)
            # Main function to process all videos in the directory
            frame = process_frames(load_first_n_frames(video_path, self.n_frames))
            if self.n_frames == 1: 
                frame = frame[0]
        elif file_ending == "pt": 
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


class TensorDataset(Dataset): 
    """Use a tensor instead of a filelist"""
    def __init__(self, img_list):
        super().__init__()
        self.images = img_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], idx, ""  # path is ""