import os
import wave
import numpy as np
import contextlib
import csv
import time
from concurrent.futures import ThreadPoolExecutor

# 경로 설정
audio_base_dir = r"D:/Audio"

# ✅ 기존 데이터 여부 확인
def is_already_logged(video_id, action_class, csv_path):
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as file:
            return any(f"{video_id},{action_class}" in line for line in file)
    return False

# ✅ 무음 감지 함수
def is_silent(audio_path, silence_threshold=-50.0):
    try:
        with contextlib.closing(wave.open(audio_path, 'r')) as wf:
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            if len(audio_data) == 0:
                return True
            rms = np.sqrt(np.mean(audio_data ** 2))
            db = 20 * np.log10(rms + 1e-6)  # 안정성을 위한 작은 값 추가
            return db < silence_threshold
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return False

# ✅ 무음 감지 및 삭제 + CSV 기록 함수
def process_silence_detection(phase, classes_to_process, silence_threshold):
    audio_phase_dir = os.path.join(audio_base_dir, phase)
    no_audio_csv_path = os.path.join(audio_phase_dir, "18frames_no_audios.csv")

    # ✅ CSV 초기화 (없을 경우만 생성)
    if not os.path.exists(no_audio_csv_path):
        with open(no_audio_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["id", "actionclass"])  # ✅ CSV 헤더 작성

    deleted_count = 0  # ✅ 삭제된 무음 파일 카운트

    # 클래스별 폴더 탐색
    for action_class in classes_to_process:
        class_audio_path = os.path.join(audio_phase_dir, action_class, "18frames")
        if not os.path.isdir(class_audio_path):
            continue

        for audio_file in os.listdir(class_audio_path):
            if audio_file.endswith(".wav"):
                audio_path = os.path.join(class_audio_path, audio_file)
                video_id = os.path.splitext(audio_file)[0]

                # ✅ 무음 감지
                if is_silent(audio_path, silence_threshold):
                    print(f"🔇 Detected silence in: {audio_path}")

                    # ✅ 이미 기록된 경우 스킵 (중복 방지)
                    if is_already_logged(video_id, action_class, no_audio_csv_path):
                        print(f"⏭️ Skipping {video_id} (already logged)")
                        continue

                    # ✅ 무음 파일 삭제
                    try:
                        os.remove(audio_path)
                        print(f"🗑️ Deleted silent audio: {audio_path}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"❌ Error deleting {audio_path}: {e}")
                        continue

                    # ✅ 무음 파일 기록 (추가 모드)
                    with open(no_audio_csv_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([video_id, action_class])

    return deleted_count  # ✅ 삭제된 파일 수 반환

# ✅ 병렬 처리로 성능 최적화
def run_silence_detection(classes_to_process, silence_threshold=-50.0):
    print(f"🔊 무음 감지 및 삭제 시작: {classes_to_process}")
    start_time = time.time()  # ✅ 실행 시작 시간

    total_deleted = 0

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_silence_detection, phase, classes_to_process, silence_threshold)
            for phase in ["training", "validation"]
        ]
        for future in futures:
            total_deleted += future.result()

    end_time = time.time()  # ✅ 실행 종료 시간
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print(f"✅ 무음 감지 및 삭제 완료 (총 삭제 파일: {total_deleted}개)")
    print(f"⏱️ 총 처리 시간: {int(minutes)}분 {int(seconds)}초")

# ✅ 외부 호출 가능
if __name__ == "__main__":
    run_silence_detection(["default_class"], silence_threshold=-50.0)
