# dataset_to_txt.py

# 각 데이터를 개별 텍스트 파일로 저장하는 모듈
# 저장된 텍스트 파일들은 후에 실행될 TAACO, TAASSC에서 사용될 예정

import os
import zipfile

# 데이터를 텍스트 파일로 저장하고, 바로 압축하는 함수
def save_and_compress_dataset_to_zip(dataset, name_suffix, output_dir="loaded_data_txt"):
    """
    Args:
        dataset: 데이터셋의 리스트 형태. 예: (references, hypotheses)
        name_suffix: 데이터셋의 이름 (예: 'CNN', 'WMT', 'SQuAD').
        output_dir: 파일이 저장될 디렉토리 경로 (기본값: "loaded_data_txt").
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    zip_filename = os.path.join(output_dir, f"{name_suffix}_data.zip")

    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for idx, data in enumerate(zip(*dataset)):  # 두 리스트를 병렬 처리
            entry_filename = f"{name_suffix}_Entry_{idx + 1}.txt"
            entry_path = os.path.join(output_dir, entry_filename)
            with open(entry_path, 'w') as f:
                for item in data:
                    f.write(str(item) + '\n')  # 리스트의 각 요소를 저장

            # 생성된 파일을 압축에 추가
            zipf.write(entry_path, arcname=entry_filename)

            # 개별 파일 삭제 (압축 후에는 필요 없음)
            os.remove(entry_path)

    print(f"Dataset saved and compressed to {zip_filename}")
    return zip_filename  # 압축 파일 경로 반환