# dataset_to_txt.py

# 각 데이터를 개별 텍스트 파일로 저장하는 모듈
# 저장된 텍스트 파일들은 후에 실행될 TAACO, TAASSC에서 사용될 예정

import os
import zipfile

# 데이터를 텍스트 파일로 저장하는 함수
def save_dataset_to_folder(dataset, name_suffix, output_dir="loaded_data_txt"):
    """
    Args:
        dataset: 데이터셋의 리스트 형태. 예: (references, hypotheses)
        name_suffix: 데이터셋의 이름 (예: 'CNN', 'WMT', 'SQuAD').
        output_dir: 파일이 저장될 디렉토리 경로 (기본값: "loaded_data_txt").
    """
    # 각 데이터셋별 폴더 생성
    dataset_dir = os.path.join(output_dir, name_suffix)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # 데이터셋의 각 항목을 텍스트 파일로 저장
    for idx, data in enumerate(zip(*dataset)):  # 두 리스트를 병렬 처리
        entry_filename = f"{name_suffix}_Entry_{idx + 1}.txt"
        entry_path = os.path.join(dataset_dir, entry_filename)
        with open(entry_path, 'w') as f:
            for item in data:
                f.write(str(item) + '\n')  # 리스트의 각 요소를 저장

    print(f"Dataset saved in {dataset_dir}")
    return dataset_dir  # 저장된 디렉토리 경로 반환
