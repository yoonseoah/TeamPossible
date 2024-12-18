import os
import subprocess
import glob
import pandas as pd

# 입력 비디오 경로 및 출력 오디오 경로 설정
video_dir = os.path.normpath("Audio-Feature-Generation/data/raw")
audio_output_dir = os.path.normpath(r"D:/aud-processed")
output_csv_path = os.path.normpath("Audio-Feature-Generation/results/no_audio_videos.csv")
ffmpeg_path = r"C:/Users/swu/Desktop/ffmpeg-2024-12-16-git-d2096679d5-essentials_build/ffmpeg-2024-12-16-git-d2096679d5-essentials_build/bin/ffmpeg.exe"

# 오디오 스트림 확인 함수
def has_audio_stream(video_path):
    try:
        result = subprocess.run(
            [ffmpeg_path, "-i", video_path],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        return "Audio:" in result.stderr
    except Exception as e:
        print(f"Error checking audio stream for {video_path}: {e}")
        return False

# 비디오에서 오디오 추출 함수
def extract_audio_from_videos(video_root, audio_root, output_csv):
    no_audio_videos = []  # 오디오 없는 비디오 저장 리스트

    for class_name in os.listdir(video_root):
        class_video_path = os.path.normpath(os.path.join(video_root, class_name))
        class_audio_path = os.path.normpath(os.path.join(audio_root, class_name))

        if not os.path.isdir(class_video_path):
            print(f"Skipping non-directory: {class_video_path}")
            continue

        os.makedirs(class_audio_path, exist_ok=True)
        video_files = glob.glob(os.path.join(class_video_path, "*.mp4"))

        for video_path in video_files:
            video_file = os.path.basename(video_path)
            audio_output_path = os.path.normpath(os.path.join(class_audio_path, video_file.replace('.mp4', '.wav')))

            print(f"Processing video file: {video_path}")

            if not os.path.exists(video_path):
                print(f"File not found: {video_path}")
                continue

            if not has_audio_stream(video_path):
                print(f"Skipping {video_path}: No audio stream found.")
                no_audio_videos.append({"class": class_name, "file": video_file})
                continue

            try:
                subprocess.run([
                    ffmpeg_path, "-i", video_path, "-ar", "16000", "-ac", "1",
                    "-acodec", "pcm_s16le", audio_output_path, "-y"
                ], check=True)
                print(f"Extracted audio: {audio_output_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error extracting audio from {video_path}: {e}")
                no_audio_videos.append({"class": class_name, "file": video_file})
            except Exception as e:
                print(f"Unexpected error with {video_path}: {e}")
                no_audio_videos.append({"class": class_name, "file": video_file})

    # CSV 파일 저장
    if no_audio_videos:
        df = pd.DataFrame(no_audio_videos)
        df.to_csv(output_csv, index=False)
        print(f"Saved no-audio video list to: {output_csv}")
    else:
        print("All videos have audio streams. No CSV file generated.")

# 실행
if __name__ == "__main__":
    print("Starting audio extraction...")
    extract_audio_from_videos(video_dir, audio_output_dir, output_csv_path)
    print("Audio extraction completed!")
