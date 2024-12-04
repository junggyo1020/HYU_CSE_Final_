# main.py

import pandas as pd
import csv
import os
import logging
from datetime import datetime
from dataset_loader import load_wmt_dataset, load_squad_dataset, load_cnn_daily_dataset, load_sts_dataset
from dataset_to_txt import save_dataset_to_folder, save_highlights_to_folder, save_full_squads_to_folder, \
    save_final_dataset_to_folder
from evaluate_metrics import calculate_bleu, calculate_rouge, calculate_bert_score, calculate_combined_metric
from wmt_trans import translate_wmt_german_to_english

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

start_time = datetime.now()
logging.info("프로그램 실행 시작")

# WMT 데이터셋 불러오기
wmt_dataset = load_wmt_dataset()
wmt_english_data = [item['de'] for item in wmt_dataset]
wmt_references = [item['en'] for item in wmt_dataset]
wmt_hypotheses = translate_wmt_german_to_english(wmt_english_data)

# SQuAD 데이터셋 불러오기
squad_references, squad_hypotheses, squad_contexts, squad_questions = load_squad_dataset()
squad_references = [answer['text'][0] for answer in squad_references]

# 인간평가를 위한 별도의 SQuAD 데이터셋 불러오기
# squad_full_saved_path = save_full_squads_to_folder((squad_contexts, squad_questions, squad_references, squad_hypotheses), "SQuAD_Full")

# CNN/DailyMail 데이터셋 불러오기
cnn_articles, cnn_highlights = load_cnn_daily_dataset()
cnn_references = cnn_articles
cnn_hypotheses = [" ".join(highlight.split("\n")) for highlight in cnn_highlights]  # 여러 문장을 하나로 묶음

# CNN highlights만 저장
# cnn_highlights_saved_path = save_highlights_to_folder(cnn_highlights, "CNN_Highlights")

# STS 데이터셋 불러오기
sts_sentence1, sts_sentence2, sts_scores = load_sts_dataset()
sts_references = sts_sentence1
sts_hypotheses = sts_sentence2
sts_scores = sts_scores

# Final Dataset 저장 경로
final_output_dir = "final_datasets"
os.makedirs(final_output_dir, exist_ok=True)

# Final Dataset 저장
wmt_final_path = save_final_dataset_to_folder((wmt_references, wmt_hypotheses), "WMT", output_dir=final_output_dir)
squad_final_path = save_final_dataset_to_folder((squad_references, squad_hypotheses), "SQuAD", output_dir=final_output_dir)
cnn_final_path = save_final_dataset_to_folder((cnn_references, cnn_hypotheses), "CNN_DailyMail", output_dir=final_output_dir)
sts_final_path = save_final_dataset_to_folder((sts_references, sts_hypotheses), "STS", output_dir=final_output_dir)
cnn_highlights_final_path = save_final_dataset_to_folder(cnn_highlights, "CNN_Highlights", output_dir=final_output_dir)

logging.info(f"WMT Final Dataset 저장 경로: {wmt_final_path}")
logging.info(f"SQuAD Final Dataset 저장 경로: {squad_final_path}")
logging.info(f"CNN/DailyMail Final Dataset 저장 경로: {cnn_final_path}")
logging.info(f"STS Final Dataset 저장 경로: {sts_final_path}")
logging.info(f"CNN_Highlights Final Dataset 저장 경로: {cnn_highlights_final_path}")


# 데이터셋 경로와 이름
dataset_names = {
    "WMT": wmt_final_path,
    "SQuAD": squad_final_path,
    # "SQuAD_Full": squad_full_final_path, # human evaluation 용
    "CNN_DailyMail": cnn_final_path,
    "CNN_Highlights": cnn_highlights_final_path,
    "STS": sts_final_path
}

# Combined Score 저장할 디렉토리
output_dir = 'csv_files/combined_scores'
os.makedirs(output_dir, exist_ok=True)


# CSV 저장 함수 정의
def save_scores_to_csv(dataset_name, bleu_score, rouge_score, bert_score, combined_score):
    output_file = os.path.join(output_dir, f"{dataset_name}_scores.csv")
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Score'])
        writer.writerow(['BLEU', bleu_score])
        writer.writerow(['ROUGE', rouge_score])
        writer.writerow(['BERTScore', bert_score])
        writer.writerow(['CombinedScore', combined_score])


# 각 데이터셋의 평가 점수 계산 및 저장(Final)
for dataset_name, saved_path in dataset_names.items():
    references, hypotheses = [], []
    for entry_file in sorted(os.listdir(saved_path)):  # 정렬된 Entry 파일 목록 가져오기
        if entry_file.endswith(".txt"):  # .txt 파일만 처리
            entry_path = os.path.join(saved_path, entry_file)
            with open(entry_path, 'r') as f:
                lines = f.readlines()
                references.append(lines[0].strip())  # 첫 번째 줄을 references로 사용
                hypotheses.append(lines[1].strip())  # 두 번째 줄을 hypotheses로 사용

    # 각 점수 계산
    bleu_score = calculate_bleu(references, hypotheses)
    rouge_score = calculate_rouge(references, hypotheses)
    bert_score = calculate_bert_score(references, hypotheses)
    combined_score = calculate_combined_metric(dataset_name, references, hypotheses)

    # 점수 저장
    save_scores_to_csv(dataset_name, bleu_score, rouge_score, bert_score, combined_score)
    logging.info(f"{dataset_name} 전체 데이터 점수 계산 완료 및 저장.")


# Human Evaluation - STS 점수 정규화 및 평균 계산
def normalize_and_average_scores(scores, min_score=0, max_score=5):
    """
    점수를 정규화하고 평균을 계산합니다.
    Args:
        scores: 원래의 점수 리스트 (0~5 범위).
        min_score: 정규화의 최소 값 (기본값: 0).
        max_score: 정규화의 최대 값 (기본값: 5).
    Returns:
        정규화된 점수의 평균 값 (0~1 범위).
    """
    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    return sum(normalized_scores) / len(normalized_scores)

# 데이터를 10개의 서브 데이터셋으로 나누기
chunk_size = 10
sub_datasets = [
    (sts_references[i:i + chunk_size], sts_hypotheses[i:i + chunk_size], sts_scores[i:i + chunk_size])
    for i in range(0, len(sts_references), chunk_size)
]

# 서브 데이터셋별로 정규화 및 평균 계산
sts_scores_results = []
for idx, (_, _, scores) in enumerate(sub_datasets):
    average_score = normalize_and_average_scores(scores)
    sts_scores_results.append(average_score)

# STS 평균 점수 요약 파일 생성
summary_file = os.path.join(output_dir, "STS_summary_scores.csv")
summary_data = {"SubDataset": [f"STS_sub_{i + 1}" for i in range(len(sts_scores_results))],
                "AverageScore": sts_scores_results}

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(summary_file, index=False)
logging.info("STS 전체 요약 점수 파일 저장 완료.")

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