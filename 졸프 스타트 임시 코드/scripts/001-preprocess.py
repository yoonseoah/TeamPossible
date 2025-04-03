import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def extract_frames(video_path, output_dir, target_fps=6, target_frames=18):
    """Extract frames from a video, pad to 18 frames if necessary, and save as images."""
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / target_fps)
    saved_frames = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            saved_frames.append(frame)
        frame_count += 1

    cap.release()

    # Padding to ensure exactly `target_frames` frames
    while len(saved_frames) < target_frames:
        saved_frames.append(saved_frames[-1])  # Pad with last frame
    saved_frames = saved_frames[:target_frames]  # Trim excess frames

    # Save frames
    for i, frame in enumerate(saved_frames):
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)
    
    print(f"Extracted {len(saved_frames)} frames to {output_dir}")

def calculate_optical_flow(frames_dir, output_dir, target_frames=18):
    """Calculate optical flow and save as images, ensuring exactly 18 frames."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if len(frame_files) < 2:
        print(f"Not enough frames in {frames_dir} for optical flow calculation.")
        return

    # Load the first frame for optical flow padding
    prev_frame = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)

    saved_flows = []
    for i in range(1, len(frame_files)):
        curr_frame = cv2.imread(frame_files[i], cv2.IMREAD_GRAYSCALE)
        if prev_frame is None or curr_frame is None:
            print(f"Error reading frames in {frames_dir}. Skipping frame {i}.")
            continue

        # Optical Flow calculation
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = np.uint8(angle * 180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        saved_flows.append(flow_bgr)
        prev_frame = curr_frame

    # Padding Optical Flow to `target_frames`
    while len(saved_flows) < target_frames:
        saved_flows.insert(0, saved_flows[0])  # Pad with first optical flow frame
    saved_flows = saved_flows[:target_frames]  # Trim excess flows

    # Save optical flow
    for i, flow in enumerate(saved_flows):
        output_path = os.path.join(output_dir, f"flow_{i:04d}.jpg")
        cv2.imwrite(output_path, flow)

    print(f"Optical flow saved in {output_dir}")

def process_video(video_path, rgb_dir, flow_dir, class_name, target_fps=6, target_frames=18):
    """Process a single video for frame extraction and optical flow calculation."""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(rgb_dir, class_name, video_id)
    optical_flow_dir = os.path.join(flow_dir, class_name, video_id)

    extract_frames(video_path, frames_dir, target_fps, target_frames)
    calculate_optical_flow(frames_dir, optical_flow_dir, target_frames)

def process_class_videos(class_path, class_name, rgb_dir, flow_dir, target_fps=6, target_frames=18):
    """Process all videos in a class directory."""
    video_files = [f for f in os.listdir(class_path) if f.endswith((".mp4", ".avi", ".mkv"))]
    for video_file in video_files:
        video_path = os.path.join(class_path, video_file)
        try:
            process_video(video_path, rgb_dir, flow_dir, class_name, target_fps, target_frames)
        except Exception as e:
            print(f"Error processing video {video_file}: {e}")

if __name__ == "__main__":
    # 경로 설정
    raw_data_dir = "Audio-Feature-Generation/data/raw"
    rgb_dir = r"D:\vid-processed\rgb"
    flow_dir = r"D:\vid-processed\flow"


    target_fps = 6
    target_frames = 18

    # 클래스 단위로 모든 비디오 처리
    with ThreadPoolExecutor() as executor:
        futures = []
        for class_dir in os.listdir(raw_data_dir):
            class_path = os.path.join(raw_data_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            futures.append(executor.submit(process_class_videos, class_path, class_dir, rgb_dir, flow_dir, target_fps, target_frames))

        for future in futures:
            future.result()

    print("All videos processed.")
