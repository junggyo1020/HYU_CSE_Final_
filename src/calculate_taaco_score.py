# calculate_taaco_score.py

import os
import csv

# 임의로 설정한 고정 가중치
taaco_weights = [0.15, 0.15, 0.1, 0.1, 0.2, 0.2, 0.05, 0.05]

# 각 csv 파일의 지표 평균에 가중치를 적용해 TAACO_Score를 계산하고 새로운 파일로 저장하는 함수
def calculate_taaco_score(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_directory, filename)
            dataset_name = filename.split("_")[0]  # 파일명에서 데이터셋 이름 추출 (예: cnn, squad, wmt)

            with open(file_path, newline='', encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                headers = next(csvreader)  # 첫 줄은 헤더로 처리

                # 열의 평균값을 계산하고 가중치를 적용
                for row in csvreader:
                    column_values = [float(row[idx]) for idx in range(1, 9)]  # 첫 열(파일명)을 제외한 8개의 값 가져오기
                    weighted_score = sum([value * weight for value, weight in zip(column_values, taaco_weights)])

                    # 새로운 csv 파일명 설정
                    output_filename = f"{dataset_name}_taaco_score.csv"
                    output_file_path = os.path.join(output_directory, output_filename)

                    # 새로운 csv 파일에 가중치 적용 결과 저장
                    with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
                        writer = csv.writer(outfile)
                        writer.writerow(["Filename", "TAACO_Score"])  # 헤더 추가
                        writer.writerow([filename, weighted_score])  # 파일명과 TAACO_Score 기록

                    print(f"{filename}의 TAACO_Score: {weighted_score}")

# 실행 예시
input_directory = "taaco_average_results"  # 평균값 csv 파일들이 있는 폴더 경로
output_directory = "taaco_scores"  # 결과를 저장할 폴더 경로
calculate_taaco_score(input_directory, output_directory)
