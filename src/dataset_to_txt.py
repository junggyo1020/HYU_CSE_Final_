# dataset_to_txt.py

# 각 데이터를 개별 텍스트 파일로 저장하는 모듈
# 저장된 텍스트 파일들은 후에 실행될 TAACO, TAASSC에서 사용될 예정

import os


# 데이터를 텍스트 파일로 저장하는 함수
def save_dataset_to_folder(dataset, name_suffix, output_dir="sub_datasets"):
    """
    Args:
        dataset: 데이터셋의 리스트 형태. 예: (references, hypotheses)
        name_suffix: 데이터셋의 이름 (예: 'CNN', 'WMT', 'SQuAD').
        output_dir: 파일이 저장될 디렉토리 경로 (기본값: "sub_datasets").
    """
    # 각 데이터셋별 폴더 생성
    dataset_dir = os.path.join(output_dir, name_suffix)
    os.makedirs(dataset_dir, exist_ok=True)

    # 데이터를 10개씩 나눔
    chunk_size = 10
    sub_datasets = list(zip(*dataset))
    sub_datasets = [sub_datasets[i:i + chunk_size] for i in range(0, len(sub_datasets), chunk_size)]

    # 각 서브 데이터셋을 폴더와 파일로 저장
    for idx, sub_dataset in enumerate(sub_datasets):
        sub_folder = os.path.join(dataset_dir, f"{name_suffix}_sub_{idx + 1}")
        os.makedirs(sub_folder, exist_ok=True)

        # 서브 데이터셋의 각 항목을 텍스트 파일로 저장
        for entry_idx, (reference, hypothesis) in enumerate(sub_dataset, start=1):  # references와 hypotheses를 병렬 처리
            entry_filename = f"{name_suffix}_Entry_{entry_idx}.txt"
            entry_path = os.path.join(sub_folder, entry_filename)
            with open(entry_path, "w") as f:
                f.write(f"{reference}\n")  # 첫 번째 줄에 reference 저장
                f.write(f"{hypothesis}\n")  # 두 번째 줄에 hypothesis 저장

        print(f"Sub-dataset {idx + 1} saved in {sub_folder}")

    print(f"{name_suffix} 데이터셋이 {len(sub_datasets)}개의 서브 데이터셋으로 저장되었습니다.")
    return dataset_dir


def save_highlights_to_folder(highlights, name_suffix, output_dir="sub_datasets"):
    """
    CNN_Highlights 데이터를 개별 텍스트 파일로 저장하는 함수.
    Args:
        highlights: 요약문 텍스트 리스트.
        name_suffix: 데이터셋 이름 (예: 'CNN_Highlights').
        output_dir: 파일 저장 디렉토리 (기본값: "sub_datasets").
    """
    dataset_dir = os.path.join(output_dir, name_suffix)
    os.makedirs(dataset_dir, exist_ok=True)

    # 데이터를 10개씩 나누기
    chunk_size = 10
    sub_datasets = [highlights[i:i + chunk_size] for i in range(0, len(highlights), chunk_size)]

    # 서브 데이터셋 저장
    for idx, sub_dataset in enumerate(sub_datasets):
        sub_folder = os.path.join(dataset_dir, f"{name_suffix}_sub_{idx + 1}")
        os.makedirs(sub_folder, exist_ok=True)

        for entry_idx, highlight in enumerate(sub_dataset, start=1):
            entry_filename = f"{name_suffix}_Entry_{entry_idx}.txt"
            entry_path = os.path.join(sub_folder, entry_filename)
            with open(entry_path, "w") as f:
                f.write(f"{highlight}\n")

        print(f"Sub-dataset {idx + 1} saved in {sub_folder}")

    print(f"{name_suffix} 데이터셋이 {len(sub_datasets)}개의 서브 데이터셋으로 저장되었습니다.")
    return dataset_dir


def save_full_squads_to_folder(dataset, name_suffix, output_dir="sub_datasets"):
    """
    Args:
        dataset: 데이터셋의 리스트 형태. 예: (context, question, references, hypotheses)
        name_suffix: 데이터셋의 이름 (예: 'SQuAD_Full').
        output_dir: 파일이 저장될 디렉토리 경로 (기본값: "sub_datasets").
    """
    # 각 데이터셋별 폴더 생성
    dataset_dir = os.path.join(output_dir, name_suffix)
    os.makedirs(dataset_dir, exist_ok=True)

    # 데이터를 10개씩 나눔
    chunk_size = 10
    sub_datasets = list(zip(*dataset))
    sub_datasets = [sub_datasets[i:i + chunk_size] for i in range(0, len(sub_datasets), chunk_size)]

    # 각 서브 데이터셋을 폴더와 파일로 저장
    for idx, sub_dataset in enumerate(sub_datasets):
        sub_folder = os.path.join(dataset_dir, f"{name_suffix}_sub_{idx + 1}")
        os.makedirs(sub_folder, exist_ok=True)

        # 서브 데이터셋의 각 항목을 텍스트 파일로 저장
        for entry_idx, (context, question, reference, hypothesis) in enumerate(sub_dataset, start=1):
            entry_filename = f"{name_suffix}_Entry_{entry_idx}.txt"
            entry_path = os.path.join(sub_folder, entry_filename)
            with open(entry_path, "w") as f:
                f.write(f"{context}\n")  # 첫 번째 줄에 context 저장
                f.write(f"{question}\n")  # 두 번째 줄에 question 저장
                f.write(f"{reference}\n")  # 세 번째 줄에 reference 저장
                f.write(f"{hypothesis}\n")  # 네 번째 줄에 hypothesis 저장

        print(f"Sub-dataset {idx + 1} saved in {sub_folder}")

    print(f"{name_suffix} 데이터셋이 {len(sub_datasets)}개의 서브 데이터셋으로 저장되었습니다.")
    return dataset_dir
