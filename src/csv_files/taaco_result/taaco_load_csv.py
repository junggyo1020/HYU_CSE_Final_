import os
import csv


# 각 csv 파일의 지표 평균을 계산하고 새로운 파일로 저장하는 함수
def calculate_average_and_save(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_directory, filename)
            dataset_name = filename.split("_")[0]  # 파일명에서 데이터셋 이름 추출 (예: cnn, squad, wmt)

            with open(file_path, newline='', encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                headers = next(csvreader)  # 첫 줄은 헤더로 처리

                column_sums = [0] * (len(headers) - 1)  # 첫 번째 열(파일명)을 제외한 나머지 열들의 합계를 저장
                row_count = 0

                for row in csvreader:
                    row_count += 1
                    for idx in range(1, len(row)):  # 첫 열(파일명)을 제외한 나머지 값들의 합산
                        column_sums[idx - 1] += float(row[idx])

                # 각 열의 평균 계산
                column_averages = [column_sum / row_count for column_sum in column_sums]

                # 새로운 csv 파일명 설정
                output_filename = f"{dataset_name}_average.csv"
                output_file_path = os.path.join(output_directory, output_filename)

                # 새로운 csv 파일에 평균값 저장
                with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(["Filename"] + headers[1:])  # 첫 번째 열(파일명)을 제외한 헤더
                    writer.writerow([filename] + column_averages)  # 파일명과 평균값 기록


# 실행 예시
input_directory = "./"  # csv 파일들이 있는 폴더 경로
output_directory = "../taaco_average_results"  # 결과를 저장할 폴더 경로
calculate_average_and_save(input_directory, output_directory)
