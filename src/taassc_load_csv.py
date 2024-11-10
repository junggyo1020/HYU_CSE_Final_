import csv
import os
import numpy as np


# 모든 열에 대해 평균 계산 (filename 열 제외)
def calculate_column_averages(csv_file):
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        column_data = {col: [] for col in reader.fieldnames if col != 'filename'}  # 'filename' 열 제외

        for row in reader:
            for col in column_data.keys():
                try:
                    column_data[col].append(float(row[col]))
                except ValueError:
                    continue

        averages = {col: np.mean(column_data[col]) if column_data[col] else 0 for col in column_data.keys()}
        return averages


# 결과를 저장하는 함수
def save_average_to_csv(output_file, filename, averages):
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + [f"{col}_avg" for col in averages.keys()])
        writer.writerow([filename] + list(averages.values()))  # 파일명만 기록


# 전체 폴더에서 파일을 처리
def process_all_files(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory, f"avg_{filename}")

            # 파일명에서 경로 제외하고 이름만 남김
            short_filename = os.path.basename(filename)

            # 열 평균 계산 및 결과 저장
            averages = calculate_column_averages(input_file)
            if averages:
                save_average_to_csv(output_file, short_filename, averages)


# 실행
input_directory = "csv_files/taassc_result"
output_directory = "csv_files/taassc_average_results"
os.makedirs(output_directory, exist_ok=True)

process_all_files(input_directory, output_directory)
