# main.py

import csv
import os
import logging
from datetime import datetime
from dataset_loader import load_wmt_dataset, load_squad_dataset, load_cnn_daily_dataset
from dataset_to_txt import save_dataset_to_folder
from evaluate_metrics import calculate_bleu, calculate_rouge, calculate_bert_score, calculate_combined_metric
from wmt_trans import translate_wmt_english_to_german

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

start_time = datetime.now()
logging.info("프로그램 실행 시작")

# WMT 데이터셋 불러오기
wmt_dataset = load_wmt_dataset()
wmt_english_data = [item['en'] for item in wmt_dataset]
wmt_references = translate_wmt_english_to_german(wmt_english_data)
wmt_hypotheses = [item['de'] for item in wmt_dataset]
wmt_saved_path = save_dataset_to_folder((wmt_references, wmt_hypotheses), "WMT")

# SQuAD 데이터셋 불러오기
squad_contexts, _, squad_answers = load_squad_dataset()
squad_references = [answer['text'][0] for answer in squad_answers]
squad_hypotheses = squad_contexts  # 문맥을 번역본으로 가정
squad_saved_path = save_dataset_to_folder((squad_references, squad_hypotheses), "SQuAD")

# CNN/DailyMail 데이터셋 불러오기
cnn_articles, cnn_highlights = load_cnn_daily_dataset()
cnn_references = cnn_highlights
cnn_hypotheses = cnn_articles
cnn_saved_path = save_dataset_to_folder((cnn_references, cnn_hypotheses), "CNN_DailyMail")

# 서브 데이터셋 경로와 이름
dataset_names = {
    "WMT": wmt_saved_path,
    "SQuAD": squad_saved_path,
    "CNN_DailyMail": cnn_saved_path
}

# Combined Score 저장할 디렉토리
output_dir = 'combined_scores'
os.makedirs(output_dir, exist_ok=True)

logging.info(f"WMT 데이터셋 저장 경로: {wmt_saved_path}")
logging.info(f"SQuAD 데이터셋 저장 경로: {squad_saved_path}")
logging.info(f"CNN 데이터셋 저장 경로: {cnn_saved_path}")


# CSV 저장 함수 정의
def save_scores_to_csv(dataset_name, sub_dataset_idx, bleu_score, rouge_score, bert_score, combined_score):
    output_file = os.path.join(output_dir, f"{dataset_name}_sub_{sub_dataset_idx + 1}_scores.csv")
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Score'])
        writer.writerow(['BLEU', bleu_score])
        writer.writerow(['ROUGE', rouge_score])
        writer.writerow(['BERTScore', bert_score])
        writer.writerow(['CombinedScore', combined_score])


# 각 데이터셋의 서브 데이터셋에 대해 평가 점수 계산 및 저장
for dataset_name, saved_path in dataset_names.items():
    for sub_dataset_idx in range(10):
        sub_dataset_folder = os.path.join(saved_path, f"{dataset_name}_sub_{sub_dataset_idx + 1}")

        # 서브 데이터셋의 텍스트 파일 로드
        references, hypotheses = [], []
        for idx in range(1, 11):  # 각 서브 데이터셋에 10개의 파일이 존재
            entry_file = os.path.join(sub_dataset_folder, f"{dataset_name}_Entry_{idx}.txt")
            with open(entry_file, 'r') as f:
                lines = f.readlines()
                references.append(lines[0].strip())  # 첫 번째 줄을 references로 사용
                hypotheses.append(lines[1].strip())  # 두 번째 줄을 hypotheses로 사용

        # 각 점수 계산
        bleu_score = calculate_bleu(references, hypotheses)
        rouge_score = calculate_rouge(references, hypotheses)
        bert_score = calculate_bert_score(references, hypotheses)
        combined_score = calculate_combined_metric(dataset_name, references, hypotheses)

        # 점수 저장
        save_scores_to_csv(dataset_name, sub_dataset_idx, bleu_score, rouge_score, bert_score, combined_score)
        logging.info(f"{dataset_name} 서브 데이터셋 {sub_dataset_idx + 1}의 점수 저장 완료.")

# 프로그램 종료 및 실행 시간 출력
end_time = datetime.now()
logging.info("프로그램 종료")
elapsed_time = end_time - start_time
logging.info(f"프로그램 총 실행 시간: {elapsed_time}")


# # 다양한 ngram_weight 값 설정
# ngram_weight_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#
# # 품사 가중치 조합
# pos_weights_list = [
#     {'NOUN': 1.5, 'VERB': 1.3, 'ADJ': 1.2},
#     {'NOUN': 1.5, 'VERB': 1.3, 'ADJ': 1.0},
#     {'NOUN': 1.2, 'VERB': 1.1, 'ADJ': 1.0},
#     {'NOUN': 1.4, 'VERB': 1.5, 'ADJ': 1.2},
#     {'NOUN': 1.2, 'VERB': 1.4, 'ADJ': 1.6},
# ]

# def run_ngram_weight_experiments(dataset_name, references, hypotheses, ngram_weight_list, pos_weights=None):
#     """
#     다양한 n-gram 가중치 값으로 실험을 수행하여 최적의 가중치를 찾는 함수.
#     """
#     combined_scores = []  # 실험 결과 저장
#
#     for i, ngram_weight in enumerate(ngram_weight_list):
#         print(f"실험 {i + 1}: n-gram weight = {ngram_weight}")
#
#         # Combined Metric 계산
#         combined_score = calculate_combined_metric(references, hypotheses, ngram_weight=ngram_weight, max_n=4,
#                                                    pos_weights=pos_weights)
#
#         # 결과 저장
#         combined_scores.append(combined_score)
#         print(f"Combined Score {i + 1}: {combined_score}\n")
#
#     # 실험 결과 시각화
#     visualize_ngram_weight_experiment_results(dataset_name, ngram_weight_list, combined_scores)
#
#
# def run_weight_experiments_with_visualization(dataset_name, references, hypotheses, pos_weights_list):
#     combined_scores = []  # 실험 결과 저장
#
#     for i, pos_weights in enumerate(pos_weights_list):
#         print(f"실험 {i + 1}: 품사별 가중치 {pos_weights}")
#
#         # Combined Metric 계산
#         combined_score = calculate_combined_metric(dataset_name, references, hypotheses, pos_weights=pos_weights)
#
#         # 결과 저장
#         combined_scores.append(combined_score)
#         print(f"Combined Score {i + 1}: {combined_score}\n")
#
#     # 실험 결과 시각화
#     visualize_weight_experiment_results(dataset_name, pos_weights_list, combined_scores)

# 품사 가중치 실험
# print("Running WMT weight experiments...")
# run_weight_experiments_with_visualization("WMT", wmt_references, wmt_hypotheses, pos_weights_list)
#
# print("Running SQuAD weight experiments...")
# run_weight_experiments_with_visualization("SQuAD", squad_references, squad_hypotheses, pos_weights_list)
#
# print("Running CNN/DailyMail weight experiments...")
# run_weight_experiments_with_visualization("CNN/DailyMail", cnn_references, cnn_hypotheses, pos_weights_list)

# n_gram_weight 가중치 실험
# print("Running WMT n-gram weight experiments...")
# run_ngram_weight_experiments("WMT", wmt_references, wmt_hypotheses, ngram_weight_list,
#                              pos_weights={'NOUN': 1.2, 'VERB': 1.1, 'ADJ': 1.0})
#
# print("Running SQuAD n-gram weight experiments...")
# run_ngram_weight_experiments("SQuAD", squad_references, squad_hypotheses, ngram_weight_list,
#                              pos_weights={'NOUN': 1.2, 'VERB': 1.1, 'ADJ': 1.0})
#
# print("Running CNN n-gram weight experiments...")
# run_ngram_weight_experiments("CNN/DailyMail", cnn_references, cnn_hypotheses, ngram_weight_list,
#                              pos_weights={'NOUN': 1.2, 'VERB': 1.1, 'ADJ': 1.0})