# cnn과 cnn 요약문의 taaco 점수를 평균 내어 합치는 모듈

import csv

# 두 CSV 파일의 파일 경로
taaco_file = 'cnn_taaco_score.csv'
summary_file = 'cnn_summary_score.csv'


# CSV 파일을 읽은 후 2행 2열의 점수를 가져오는 함수
def get_score_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        scores = list(reader)
        score = float(scores[1][1])
        return score


# 점수를 얻기
taaco_score = get_score_from_csv(taaco_file)
summary_score = get_score_from_csv(summary_file)

# 평균 점수 계산
average_score = (taaco_score + summary_score) / 2


# 평균 점수를 cnn_taaco_score.csv의 2행 2열에 덮어쓰는 함수
def overwrite_score_in_csv(file_path, score):
    with open(file_path, 'r') as file:
        reader = list(csv.reader(file))
    reader[1][1] = str(score)
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(reader)


# 점수 덮어쓰기
overwrite_score_in_csv(taaco_file, average_score)