import os
import cv2
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

# 경로 설정
rgb_base_dir = r"D:\RGB"
flow_base_dir = r"D:\OpticalFlow"

# Optical Flow 저장 (비동기 방식)
def save_flow_image(flow_bgr, output_path):
    cv2.imwrite(output_path, flow_bgr)

# Optical Flow 계산 (HSV → BGR 방식)
def calculate_optical_flow(frames_dir, output_dir):
    frame_files = sorted([
        os.path.join(frames_dir, f) 
        for f in os.listdir(frames_dir) 
        if f.endswith((".jpg", ".png"))
    ])

    if len(frame_files) < 2:
        print(f"⚠️ Not enough frames in {frames_dir} for optical flow calculation.")
        return

    frames = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in frame_files]

    with ThreadPoolExecutor() as save_executor:
        saved_flows = []

        for i in range(1, len(frames)):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]

            if prev_frame is None or curr_frame is None:
                continue

            # Optical Flow 계산
            flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None,
                                                pyr_scale=0.3,
                                                levels=2,
                                                winsize=7,
                                                iterations=2,
                                                poly_n=5,
                                                poly_sigma=1.1,
                                                flags=0)

            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            hsv = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = np.uint8(angle * 180 / np.pi / 2)
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            saved_flows.append(flow_bgr)

        # Optical Flow 프레임 수를 18개로 맞추기
        if len(saved_flows) >= 17:
            saved_flows.insert(0, saved_flows[0])  # 첫 프레임 복제
        else:
            print(f"⚠️ {frames_dir} has less than 17 optical flow frames.")

        # Optical Flow 저장
        for i, flow_img in enumerate(saved_flows[:18]):
            output_path = os.path.join(output_dir, f"flow_{i:04d}.png")
            save_executor.submit(save_flow_image, flow_img, output_path)

    print(f"✅ Optical Flow saved in {output_dir}")

# 비디오 단위 처리 함수
def process_video_for_optical_flow(rgb_dir, flow_dir, phase, class_dir, video_id):
    frames_dir = os.path.join(rgb_dir, phase, class_dir, "18frames", video_id)
    output_dir = os.path.join(flow_dir, phase, class_dir, "18frames", video_id)  # ✅ 중복 폴더 제거

    # ✅ 이미 처리된 경우 스킵
    if os.path.exists(output_dir) and len([f for f in os.listdir(output_dir) if f.endswith(".png")]) >= 18:
        print(f"⏭️ Skipping {video_id} (already processed)")
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        calculate_optical_flow(frames_dir, output_dir)
    except Exception as e:
        print(f"❌ Error processing optical flow for {video_id}: {e}")

# Optical Flow 추출 메인 함수
def run_optical_flow_extraction(classes_to_process):
    print(f"🎥 Optical Flow 추출 시작: {classes_to_process}")
    start_time = time.time()
    num_workers = multiprocessing.cpu_count()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for phase in ["training", "validation"]:
            for class_dir in classes_to_process:
                class_path = os.path.join(rgb_base_dir, phase, class_dir, "18frames")
                if os.path.isdir(class_path):
                    for video_id in os.listdir(class_path):
                        if os.path.isdir(os.path.join(class_path, video_id)):
                            futures.append(executor.submit(
                                process_video_for_optical_flow,
                                rgb_base_dir, flow_base_dir, phase, class_dir, video_id
                            ))

        for future in futures:
            future.result()  # 병렬 처리 완료 대기

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"✅ Optical Flow 추출 완료 (⏱️ {int(minutes)}분 {int(seconds)}초)")

# ✅ 외부 호출 가능
if __name__ == "__main__":
    run_optical_flow_extraction(["default_class"])
