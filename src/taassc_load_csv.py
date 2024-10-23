import csv
import os
import numpy as np

# 선택된 열에 대해 평균 계산
def calculate_column_averages(csv_file, selected_columns):
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        column_data = {col: [] for col in selected_columns}

        for row in reader:
            for col in selected_columns:
                try:
                    column_data[col].append(float(row[col]))
                except ValueError:
                    continue

        averages = {col: np.mean(column_data[col]) if column_data[col] else 0 for col in selected_columns}
        return averages

# 결과를 저장하는 함수
def save_average_to_csv(output_file, averages):
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename"] + [f"{col}_avg" for col in averages.keys()])
        writer.writerow([output_file] + list(averages.values()))

# 전체 폴더에서 파일을 처리
def process_all_files(input_directory, output_directory, selected_columns):
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory, f"avg_{filename}")
            averages = calculate_column_averages(input_file, selected_columns)
            if averages:
                save_average_to_csv(output_file, averages)

# 실행
input_directory = "taassc_result"
output_directory = "taassc_average_results"
os.makedirs(output_directory, exist_ok=True)

# 선택된 열 목록 수정
selected_columns = ['acomp_per_cl', 'advcl_per_cl', 'xcomp_per_cl', 'modal_per_cl', 'nsubj_per_cl']

process_all_files(input_directory, output_directory, selected_columns)
