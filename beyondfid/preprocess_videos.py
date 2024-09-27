import argparse
import os
import cv2
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# Function to load first n frames from the video
def load_first_n_frames(video_path, n_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(n_frames):
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Failed to load any frames from video: {video_path}")
    
    return frames

# Function to resize the frames to 512x512 and convert them to tensors
def process_frames(frames, size=(512, 512)):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor()  # Converts frame to tensor with shape [C, H, W]
    ])
    tensor_frames = [transform(frame) for frame in frames]
    return torch.stack(tensor_frames)  # Shape [n_frames, C, H, W]

# Function to save tensor to specified directory with original video name
def save_tensor(tensor, output_dir, video_name):
    output_path = os.path.join(output_dir, video_name.replace(".mp4", ".pt"))
    torch.save(tensor, output_path)

# Main function to process all videos in the directory
def process_videos(input_dir, output_dir, n_frames):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_name in tqdm(os.listdir(input_dir), "Loading Videos"):
        if video_name.endswith(".mp4"):
            video_path = os.path.join(input_dir, video_name)
            try:
                frames = load_first_n_frames(video_path, n_frames)
                tensor = process_frames(frames)
                save_tensor(tensor, output_dir, video_name)
            except Exception as e:
                print(f"Error processing {video_name}: {e}")

# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess videos and save as tensors.")
    parser.add_argument("--input_dir", required=True, help="Directory containing mp4 videos")
    parser.add_argument("--output_dir", required=True, help="Directory to save output tensors")
    parser.add_argument("--n_frames", type=int, default=5, help="Number of frames to load from each video")

    args = parser.parse_args()
    
    process_videos(args.input_dir, args.output_dir, args.n_frames)