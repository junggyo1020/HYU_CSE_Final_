# calculate_taassc_score.py

import os
import csv

# 임의로 설정한 고정 가중치
taassc_weights = [0.3, 0.2, 0.2, 0.15, 0.15]

# 각 csv 파일의 평균값에 가중치를 적용해 TAASSC_Score를 계산하고 새로운 파일로 저장하는 함수
def calculate_taassc_score(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_directory, filename)
            dataset_name = filename.split("_")[1]  # 파일명에서 데이터셋 이름 추출 (예: cnn, squad, wmt)

            with open(file_path, newline='', encoding='utf-8') as csvfile:
                csvreader = csv.DictReader(csvfile)
                row = next(csvreader)  # 첫 번째 데이터를 가져옴 (이미 평균값으로 저장됨)

                # 선택된 열(지표)들의 평균값 추출
                selected_columns = ['acomp_per_cl_avg', 'advcl_per_cl_avg', 'xcomp_per_cl_avg', 'modal_per_cl_avg', 'nsubj_per_cl_avg']
                column_values = [float(row[col]) for col in selected_columns]

                # 가중치 적용
                weighted_score = sum([value * weight for value, weight in zip(column_values, taassc_weights)])

                # 새로운 csv 파일명 설정
                output_filename = f"{dataset_name}_taassc_score.csv"
                output_file_path = os.path.join(output_directory, output_filename)

                # 새로운 csv 파일에 가중치 적용 결과 저장
                with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(["Filename", "TAASSC_Score"])  # 헤더 추가
                    writer.writerow([filename, weighted_score])  # 파일명과 TAASSC_Score 기록

                print(f"{filename}의 TAASSC_Score: {weighted_score}")

# 실행 예시
input_directory = "taassc_average_results"  # 평균값 csv 파일들이 있는 폴더 경로
output_directory = "taassc_scores"  # 결과를 저장할 폴더 경로
calculate_taassc_score(input_directory, output_directory)
