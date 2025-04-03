import pandas as pd

# ✅ 원본 CSV 파일 경로
csv_path = r"D:\RGB\validation\18frames_videos.csv"

# ✅ 수정된 CSV 파일 저장 경로
modified_csv_path = r"D:\RGB\validation\modified_csv.csv"

# ✅ 삭제할 actionclass 목록
remove_classes = ["boiling", "bulldozing", "buying", "clawing"]

# ✅ CSV 파일 읽기
df = pd.read_csv(csv_path)

# ✅ 특정 actionclass가 포함된 행 삭제
df_filtered = df[~df['actionclass'].isin(remove_classes)]

# ✅ 수정된 CSV 파일 저장
df_filtered.to_csv(modified_csv_path, index=False)

print(f"✅ {modified_csv_path} 파일이 생성되었습니다. (제거된 클래스: {', '.join(remove_classes)})")
