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
    video_tensor = transforms.Resize(512)(video_tensor)
    return video_tensor


# Function to load first n frames from the video
def load_first_n_frames(video_path, n_frames):
    import cv2
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
        self.got_warned = False

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
                if not self.got_warned:
                    logger.warning(f"Tensor {os.path.join(self.basedir, path)} not 0-1 normalized.") 
                    self.got_warned = True
            if tensor.dim() == 4:
                frame = first_frame(tensor)
            else: 
                frame = tensor
        else:
            # Load and normalize an image
            image_path = os.path.join(self.basedir, path)
            frame = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
            trafos = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512), transforms.ToTensor()])
            frame = trafos(frame)

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


class H5Dataset(Dataset):
    """Load all images from an HDF5 file.

    The file is opened lazily (once per worker) so that the closed handle
    is safely pickled across DataLoader workers and multiprocessing.spawn.

    Args:
        h5_path:     Path to the .h5 file.
        dataset_key: HDF5 dataset name containing the images.
                     Defaults to 'images'; falls back to the first key found.
    """
    def __init__(self, h5_path, dataset_key="images"):
        super().__init__()
        self.h5_path = h5_path
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for H5 support. Install with: pip install h5py")
        with h5py.File(h5_path, "r") as f:
            if dataset_key not in f:
                available = list(f.keys())
                dataset_key = available[0]
                logger.warning(
                    f"H5 key 'images' not found in {h5_path}. "
                    f"Using '{dataset_key}'. Available keys: {available}"
                )
            self.length = len(f[dataset_key])
        self.dataset_key = dataset_key
        self._file = None  # opened lazily in __getitem__

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        import h5py
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        img = self._file[self.dataset_key][idx]
        img = torch.from_numpy(img).float()
        if img.ndim == 3 and img.shape[-1] in (1, 3, 4):  # H x W x C → C x H x W
            img = img.permute(2, 0, 1)
        if img.shape[0] == 4:  # RGBA → RGB
            img = img[:3]
        if img.max() > 1.0:
            img = img / 255.0
        # Resize to 512 to match GenericDataset default (ImageNet-512)
        if img.shape[-1] != 512 or img.shape[-2] != 512:
            img = transforms.Resize(512)(img)
            img = transforms.CenterCrop(512)(img)
        return img, idx, str(idx)