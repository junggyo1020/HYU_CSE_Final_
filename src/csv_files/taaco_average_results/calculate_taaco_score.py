import os
import csv

# 데이터셋별 고정 가중치 벡터
wmt_weights = [0, 0, 0, 0, 0, 0, 0.6, 0.4, 0, 0]
squad_weights = [0, 0, 0, 0, 0, 0, 0.7, 0.3, 0, 0]
cnn_weights = [0, 0, 0, 0, 0, 0, 0.6, 0.4, 0, 0]
cnn_summary_weights = [0, 0, 0.6, 0.4, 0, 0, 0, 0, 0, 0]
sts_weights = [0, 0, 0, 0, 0, 0, 0.7, 0.3, 0, 0]

# 각 데이터셋 이름에 해당하는 가중치 매핑
weights_mapping = {
    'wmt': wmt_weights,
    'squad': squad_weights,
    'cnn': cnn_weights,
    'cnn_summary': cnn_summary_weights,
    'sts': sts_weights
}

# 각 csv 파일의 평균 지표 값에 가중치를 적용해 TAACO_Score를 계산하고 새로운 파일로 저장하는 함수
def calculate_taaco_score(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_directory, filename)
            dataset_name = filename.split("_")[0]  # 파일명에서 데이터셋 이름 추출 (예: cnn, squad, wmt)

            # 해당 데이터셋의 가중치 벡터 가져오기
            taaco_weights = weights_mapping.get(dataset_name, [0] * 10)  # 기본값은 0으로 채운 벡터

            with open(file_path, newline='', encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                headers = next(csvreader)  # 첫 줄은 헤더로 처리

                output_filename = f"{dataset_name}_taaco_score.csv"
                output_file_path = os.path.join(output_directory, output_filename)

                # 새로운 csv 파일에 가중치 적용 결과 저장
                with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(["Filename", "TAACO_Score"])  # 헤더 추가

                    # 열의 값에 가중치를 적용하여 TAACO_Score 계산
                    for row in csvreader:
                        column_values = [float(row[idx]) for idx in range(1, 11)]  # 첫 열(파일명)을 제외한 10개의 값 가져오기
                        weighted_score = sum([value * weight for value, weight in zip(column_values, taaco_weights)])
                        writer.writerow([row[0], weighted_score])  # 파일명과 TAACO_Score 기록

                        print(f"{row[0]}의 TAACO_Score: {weighted_score}")

# 실행 예시
input_directory = "./"  # 평균값 csv 파일들이 있는 폴더 경로
output_directory = "../taaco_scores"  # 결과를 저장할 폴더 경로
calculate_taaco_score(input_directory, output_directory)
