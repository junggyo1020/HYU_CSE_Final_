# calculate_taassc_score.py

import os
import csv

# TAASSC 고정 가중치
taassc_weights = [0.3, 0.2, 0.2, 0.15, 0.15]

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

        averages = {col: sum(column_data[col]) / len(column_data[col]) if column_data[col] else 0 for col in selected_columns}
        return averages

# TAASSC_Score 계산 함수 (가중치 적용)
def calculate_taassc_score(averages, weights):
    # 가중치를 적용해 TAASSC_Score 계산
    score = sum(averages[col] * weight for col, weight in zip(averages, weights))
    return score

# 결과를 저장하는 함수
def save_score_to_csv(output_file, averages, score):
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename"] + [f"{col}_avg" for col in averages.keys()] + ["TAASSC_Score"])
        writer.writerow([output_file] + list(averages.values()) + [score])

# 전체 폴더에서 파일을 처리
def process_all_files(input_directory, output_directory, selected_columns, weights):
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory, f"taassc_score_{filename}")
            averages = calculate_column_averages(input_file, selected_columns)
            if averages:
                score = calculate_taassc_score(averages, weights)
                save_score_to_csv(output_file, averages, score)
                print(f"{filename}의 TAASSC_Score: {score}")

# 실행
input_directory = "taassc_average_results"
output_directory = "taassc_scores"
os.makedirs(output_directory, exist_ok=True)

# 선택된 열 목록 수정
selected_columns = ['acomp_per_cl_avg','advcl_per_cl_avg','xcomp_per_cl_avg','modal_per_cl_avg','nsubj_per_cl_avg']
process_all_files(input_directory, output_directory, selected_columns, taassc_weights)
