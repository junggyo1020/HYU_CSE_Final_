import os
import pandas as pd

# 디렉토리 경로 설정
combined_scores_dir = 'csv_files/combined_scores'
taaco_scores_dir = 'csv_files/taaco_scores'

# 데이터셋 이름들
datasets = ['cnn', 'cnnHighlights', 'squad', 'wmt', 'sts']

# 데이터셋별 고정 가중치 벡터 [combined, taaco]
weights = {
    'wmt': [0.7, 0.3],
    'squad': [0.6, 0.4],
    'cnn': [0.5, 0.5],
    'cnnHighlights': [0.5, 0.5],
    'sts': [0.9, 0.1]
}

# 결과 저장을 위한 데이터프레임 리스트
final_scores = []


# 점수 파일을 읽고 가중치 적용
def calculate_final_score(combined_file, taaco_file, weights):
    combined_df = pd.read_csv(combined_file)
    taaco_df = pd.read_csv(taaco_file)

    combined_score = combined_df['Score'].iloc[3]
    taaco_score = taaco_df['TAACO_Score'].iloc[0]

    # 가중치 적용
    combined_weight, taaco_weight = weights
    final_score = combined_weight * combined_score + taaco_weight * taaco_score
    return final_score


# 각 데이터셋에 대해 처리하고 결과 저장
for dataset in datasets:
    combined_file = os.path.join(combined_scores_dir, f'{dataset}_scores.csv')
    taaco_file = os.path.join(taaco_scores_dir, f'{dataset}_taaco_score.csv')

    # 각 데이터셋에 대해 적절한 가중치 가져오기
    dataset_weights = weights.get(dataset)

    try:
        # 최종 점수 계산
        final_score = calculate_final_score(combined_file, taaco_file, dataset_weights)
        # 결과 저장
        final_scores.append({'Dataset': dataset, 'Final_Score': final_score})
    except Exception as e:
        print(f"Error processing dataset {dataset}: {e}")

# 최종 결과를 데이터프레임으로 변환
final_scores_df = pd.DataFrame(final_scores)

# 결과 출력
print(final_scores_df)

# 결과를 CSV 파일로 저장
final_scores_df.to_csv('final_score.csv', index=False)