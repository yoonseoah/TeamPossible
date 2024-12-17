# 비디오에 대해 rgb images & Optical Flow images를 처리하는 과정
import os
import cv2
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor

def extract_frames(video_path, output_dir, target_fps=6):
    """Extract frames from a video and save as images at the specified FPS."""
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / target_fps)
    frame_count = 0
    saved_count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")

def calculate_optical_flow(frames_dir, output_dir):
    """Calculate optical flow from frames and save as images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if len(frame_files) < 2:
        print(f"Not enough frames in {frames_dir} for optical flow calculation.")
        return

    prev_frame = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)

    for i in range(1, len(frame_files)):
        curr_frame = cv2.imread(frame_files[i], cv2.IMREAD_GRAYSCALE)
        
        # Ensure frames are properly loaded
        if prev_frame is None or curr_frame is None:
            print(f"Error reading frames in {frames_dir}. Skipping frame {i}.")
            continue

        # Optical Flow calculation
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = np.uint8(angle * 180 / np.pi / 2)  # Hue: direction
        hsv[..., 1] = 255  # Saturation: constant
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude

        # Convert HSV to BGR for saving
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        output_path = os.path.join(output_dir, f"flow_{i:04d}.jpg")
        cv2.imwrite(output_path, flow_bgr)

        prev_frame = curr_frame

    print(f"Optical flow saved in {output_dir}")

def process_video(video_path, rgb_dir, flow_dir, class_name, target_fps=6):
    """Process a single video for frame extraction and optical flow calculation."""
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    class_rgb_dir = os.path.join(rgb_dir, class_name)
    class_flow_dir = os.path.join(flow_dir, class_name)
    frames_dir = os.path.join(class_rgb_dir, video_id)
    optical_flow_dir = os.path.join(class_flow_dir, video_id)

    extract_frames(video_path, frames_dir, target_fps=target_fps)
    calculate_optical_flow(frames_dir, optical_flow_dir)

def process_class_videos(class_path, class_name, rgb_dir, flow_dir, target_fps=6, max_videos=1000):
    video_files = [f for f in os.listdir(class_path) if f.endswith((".mp4", ".avi", ".mkv"))]
    if len(video_files) > max_videos:
        video_files = random.sample(video_files, max_videos)  # Randomly select 1000 videos

    for video_file in video_files:
        video_path = os.path.join(class_path, video_file)
        try:
            process_video(video_path, rgb_dir, flow_dir, class_name, target_fps=target_fps)
        except Exception as e:
            print(f"Error processing video {video_file}: {e}")

if __name__ == "__main__":
    raw_data_dir = "Audio-Feature-Generation/data/raw/training"
    rgb_dir = "Audio-Feature-Generation/data/processed/rgb"
    flow_dir = "Audio-Feature-Generation/data/processed/flow"

    max_videos_per_class = 1000
    target_fps = 6

    with ThreadPoolExecutor() as executor:
        futures = []
        for class_dir in os.listdir(raw_data_dir):
            class_path = os.path.join(raw_data_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            futures.append(executor.submit(process_class_videos, class_path, class_dir, rgb_dir, flow_dir, target_fps, max_videos_per_class))

        for future in futures:
            future.result()

    print("All videos processed.")