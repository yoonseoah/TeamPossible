import os
import cv2
import csv
import time
import shutil
from concurrent.futures import ThreadPoolExecutor

# ✅ 경로 설정
raw_data_dir = r"C:\Users\swu\Desktop\원본비디오"
rgb_base_dir = r"D:\RGB"

# ✅ CSV 저장 경로
csv_paths = {
    "training": os.path.join(rgb_base_dir, "training", "18frames_videos.csv"),
    "validation": os.path.join(rgb_base_dir, "validation", "18frames_videos.csv")
}

# ✅ 파일명 중간을 생략하여 길이 제한 적용 (앞뒤 30자 유지)
def shorten_filename(filename, max_length=200, keep_length=30):
    if len(filename) > max_length:
        return filename[:keep_length] + "..." + filename[-keep_length:]
    return filename

# ✅ CSV 초기화 (파일이 없을 경우에만 초기화)
def initialize_csv():
    for phase in ["training", "validation"]:
        if not os.path.exists(csv_paths[phase]):
            with open(csv_paths[phase], mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["id", "actionclass"])

# ✅ 중복 기록 방지 함수
def is_already_logged(video_id, class_name, phase):
    if os.path.exists(csv_paths[phase]):
        with open(csv_paths[phase], 'r') as file:
            return any(f"{video_id},{class_name}" in line for line in file)
    return False

# ✅ RGB 프레임 추출 함수
def extract_frames(video_path, output_dir, phase, class_name, target_fps=6):
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # ✅ 파일명 길이 제한 적용 (200자 초과 시 앞뒤 50자 유지하고 중간 생략)
    short_video_id = shorten_filename(video_id, max_length=200, keep_length=50)

    target_dir = os.path.join(output_dir, phase, class_name, "18frames")
    final_dir = os.path.join(target_dir, short_video_id)  # ✅ video-id 폴더 한 번만 생성

    # ✅ 이미 처리된 경우 스킵
    if os.path.exists(final_dir):
        print(f"⏭️ Skipping {video_id} (already processed)")
        return

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / target_fps) if original_fps > 0 else 1
    saved_frames = []

    # ✅ temp 폴더 경로 단축 (파일명 200자 초과 시 자동 생략)
    temp_dir = os.path.join(output_dir, phase, class_name, "temp", short_video_id)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                saved_frames.append(frame)
                frame_path = os.path.join(temp_dir, f"frame_{len(saved_frames)-1:04d}.jpg")
                cv2.imwrite(frame_path, frame)
            frame_count += 1

        cap.release()

        # ✅ 18개 프레임 분류 및 저장
        if len(saved_frames) == 18 and not is_already_logged(video_id, class_name, phase):
            with open(csv_paths[phase], mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([video_id, class_name])
        else:
            target_dir = os.path.join(output_dir, phase, class_name, "others")
            final_dir = os.path.join(target_dir, short_video_id)

        os.makedirs(final_dir, exist_ok=True)

        # ✅ temp 폴더에서 최종 디렉토리로 이동
        for frame_file in os.listdir(temp_dir):
            shutil.move(os.path.join(temp_dir, frame_file), os.path.join(final_dir, frame_file))

        print(f"✅ Processed {video_id} ({phase}) - {len(saved_frames)} frames saved")

    except Exception as e:
        print(f"❌ Error processing {video_id}: {e}")

    finally:
        # ✅ temp 폴더 삭제
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"⚠️ Error deleting temp directory {temp_dir}: {e}")

# ✅ 클래스 단위로 모든 비디오 처리
def process_class_videos(class_path, class_name, phase):
    video_files = [f for f in os.listdir(class_path) if f.endswith((".mp4", ".avi", ".mkv"))]
    for video_file in video_files:
        video_path = os.path.join(class_path, video_file)
        extract_frames(video_path, rgb_base_dir, phase, class_name)

# ✅ 병렬 처리 실행
def run_rgb_extraction(classes_to_process):
    print(f"🎬 RGB 프레임 추출 시작: {classes_to_process}")
    start_time = time.time()
    initialize_csv()

    with ThreadPoolExecutor() as executor:
        futures = []
        for phase in ["training", "validation"]:
            for class_dir in classes_to_process:
                class_path = os.path.join(raw_data_dir, phase, class_dir)
                if os.path.isdir(class_path):
                    futures.append(executor.submit(process_class_videos, class_path, class_dir, phase))

        for future in futures:
            future.result()

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"✅ RGB 프레임 추출 완료 (⏱️ {int(minutes)}분 {int(seconds)}초)")

# ✅ 외부 호출 가능 (005-preprocess.py에서 호출 가능)
if __name__ == "__main__":
    run_rgb_extraction(["default_class"])
