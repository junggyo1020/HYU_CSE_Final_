import os
import pandas as pd


def convert_all_csv_to_xlsx_in_new_structure(root_directory, new_root_directory):
    """
    Traverse all subdirectories in a root directory, converting all CSV files
    to XLSX format with the same file names. The XLSX files are saved in a new
    directory structure that mirrors the original, without deleting the CSV files.

    Parameters:
    - root_directory: str, path to the root directory containing CSV files in subdirectories
    - new_root_directory: str, path to the new root directory where XLSX files will be saved
    """
    # 새 출력 디렉토리가 없으면 생성
    os.makedirs(new_root_directory, exist_ok=True)

    for directory_path, _, files in os.walk(root_directory):
        # 새 디렉토리 경로를 기존 구조를 유지하면서 생성
        relative_path = os.path.relpath(directory_path, root_directory)
        new_directory_path = os.path.join(new_root_directory, relative_path)
        os.makedirs(new_directory_path, exist_ok=True)

        for filename in files:
            if filename.endswith(".csv"):  # CSV 파일만 선택
                csv_path = os.path.join(directory_path, filename)
                xlsx_path = os.path.join(new_directory_path, filename.replace(".csv", ".xlsx"))

                try:
                    # CSV 파일을 DataFrame으로 로드
                    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")

                    # DataFrame을 새 디렉토리에 XLSX 파일로 저장
                    df.to_excel(xlsx_path, index=False, engine="xlsxwriter")
                    print(f"변환 완료: {csv_path} -> {xlsx_path}")

                except Exception as e:
                    print(f"오류 발생 {csv_path}: {e}")

# 예시 사용법
convert_all_csv_to_xlsx_in_new_structure("csv_files", "xlsx_files")
