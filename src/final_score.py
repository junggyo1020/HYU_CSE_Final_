# final_score.py

# TAASSC 점수는 사용하지 않기 때문에 관련 경로를 주석 처리

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 디렉토리 경로 설정
combined_scores_dir = 'csv_files/combined_scores'
taaco_scores_dir = 'csv_files/taaco_scores'
# taassc_scores_dir = 'csv_files/taassc_scores'

# 데이터셋 이름들
datasets = ['cnn', 'cnn_summary', 'squad', 'wmt', 'sts']

# 데이터셋별 고정 가중치 벡터 [combined, taaco]
wmt_weights = [0.7, 0.3]  # 번역 정확성 중점
squad_weights = [0.6, 0.4]  # 문맥적 연결성과 정확성 중점
cnn_weights = [0.5, 0.5]  # 정보 압축력과 문장 연결성 균등
cnn_summary_weights = [0.8, 0.2]  # 핵심 정보 전달 중점
sts_weights = [0.9, 0.1]  # 의미적 유사성 평가 중점

# 결과 저장을 위한 데이터프레임 리스트
final_scores = []

# 점수 파일을 읽고 2행 2열에서 점수를 가져와서 가중치 적용
# def calculate_final_score(combined_file, taaco_file, taassc_file):
def calculate_final_score(combined_file, taaco_file):
    combined_df = pd.read_csv(combined_file)
    taaco_df = pd.read_csv(taaco_file)
    # taassc_df = pd.read_csv(taassc_file)

    combined_score = combined_df['Score'].iloc[0]
    taaco_score = taaco_df['TAACO_Score'].iloc[0]
    # taassc_score = taassc_df['TAASSC_Score'].iloc[0]

    # 정규화 없이 가중치 적용
    # final_score = 0.4 * combined_score + 0.3 * taaco_score + 0.3 * taassc_score
    final_score = combined_weight * combined_score + taaco_weight * taaco_score
    return final_score



# 각 데이터셋에 대해 처리하고 결과 저장
for dataset in datasets:
    combined_file = os.path.join(combined_scores_dir, f'{dataset}_combined_score.csv')
    taaco_file = os.path.join(taaco_scores_dir, f'{dataset}_taaco_score.csv')
    # taassc_file = os.path.join(taassc_scores_dir, f'{dataset}_taassc_score.csv')

    final_score = calculate_final_score(combined_file, taaco_file)

    # 결과 저장
    final_scores.append({'Filename': f'{dataset}_result', 'Final_Score': final_score})

# 최종 결과 출력 또는 저장
final_scores_df = pd.DataFrame(final_scores)
print(final_scores_df)
