import os
import pandas as pd

def generate_label_csv(raw_data_dir, output_csv):
    """
    액션 클래스별로 구분된 비디오 파일을 탐색하여 
    비디오 파일명, 클래스명, 정수 클래스를 CSV로 저장합니다.

    Args:
        raw_data_dir (str): 액션 클래스별 비디오가 저장된 최상위 디렉토리 경로
        output_csv (str): 생성될 CSV 파일 경로
    """
    data = []
    class_names = sorted(os.listdir(raw_data_dir))  # 클래스 이름을 정렬 후 정수 라벨로 매핑

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(raw_data_dir, class_name)
        if not os.path.isdir(class_path):
            continue  # 디렉토리 아닌 경우 스킵

        for video_file in os.listdir(class_path):
            if video_file.endswith(('.mp4', '.avi', '.mkv')):  # 비디오 파일만 처리
                video_id = os.path.splitext(video_file)[0]  # 확장자 제거
                data.append([video_id, class_name, class_idx])

    # DataFrame 생성 및 CSV 저장
    df = pd.DataFrame(data, columns=['video_id', 'class_name', 'encoded_label'])
    df.to_csv(output_csv, index=False)
    print(f"CSV 파일이 성공적으로 저장되었습니다: {output_csv}")


if __name__ == "__main__":
    raw_data_dir = "Audio-Feature-Generation/data/raw"
    output_csv = "Audio-Feature-Generation/results/lstm_labels.csv"

    generate_label_csv(raw_data_dir, output_csv)
