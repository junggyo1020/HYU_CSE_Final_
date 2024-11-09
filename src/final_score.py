# final_score.py

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 디렉토리 경로 설정
combined_scores_dir = 'combined_scores'
taaco_scores_dir = 'taaco_scores'
taassc_scores_dir = 'taassc_scores'

# 가중치 설정
combined_weight = 0.4
taaco_weight = 0.3
taassc_weight = 0.3

# 데이터셋 이름들
datasets = ['cnn', 'squad', 'wmt']

# 결과 저장을 위한 데이터프레임 리스트
final_scores = []


# 점수 파일을 읽고 2행 2열에서 점수를 가져와서 가중치 적용
def calculate_final_score(combined_file, taaco_file, taassc_file):
    combined_df = pd.read_csv(combined_file)
    taaco_df = pd.read_csv(taaco_file)
    taassc_df = pd.read_csv(taassc_file)

    combined_score = combined_df['Score'].iloc[0]
    taaco_score = taaco_df['TAACO_Score'].iloc[0]
    taassc_score = taassc_df['TAASSC_Score'].iloc[0]

    # 정규화 없이 가중치 적용
    final_score = 0.4 * combined_score + 0.3 * taaco_score + 0.3 * taassc_score

    return final_score



# 각 데이터셋에 대해 처리하고 결과 저장
for dataset in datasets:
    combined_file = os.path.join(combined_scores_dir, f'{dataset}_combined_score.csv')
    taaco_file = os.path.join(taaco_scores_dir, f'{dataset}_taaco_score.csv')
    taassc_file = os.path.join(taassc_scores_dir, f'{dataset}_taassc_score.csv')

    final_score = calculate_final_score(combined_file, taaco_file, taassc_file)

    # 결과 저장
    final_scores.append({'Filename': f'{dataset}_result', 'Final_Score': final_score})

# 최종 결과 출력 또는 저장
final_scores_df = pd.DataFrame(final_scores)
print(final_scores_df)