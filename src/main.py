# main.py

import logging
from datetime import datetime
from dataset_loader import load_wmt_dataset, load_squad_dataset, load_cnn_daily_dataset
from dataset_to_txt import save_dataset_to_folder
from evaluate_metrics import calculate_bleu, calculate_rouge, calculate_bert_score, calculate_combined_metric
from visualize import visualize_scores, visualize_ngram_weight_experiment_results, visualize_weight_experiment_results
from wmt_trans import translate_wmt_english_to_german

# # 로깅 설정
# logging.basicConfig(
#     level=logging.INFO,  # 로그 레벨 설정 (INFO 레벨로 설정)
#     format='%(asctime)s - %(message)s',  # 로그 출력 형식
#     datefmt='%Y-%m-%d %H:%M:%S',  # 날짜/시간 형식
#     handlers=[logging.FileHandler("program_log.txt"),  # 로그를 파일에 저장
#               logging.StreamHandler()]  # 콘솔에도 출력
# )

# 다양한 ngram_weight 값 설정
ngram_weight_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 품사 가중치 조합
pos_weights_list = [
    {'NOUN': 1.5, 'VERB': 1.3, 'ADJ': 1.2},
    {'NOUN': 1.5, 'VERB': 1.3, 'ADJ': 1.0},
    {'NOUN': 1.2, 'VERB': 1.1, 'ADJ': 1.0},
    {'NOUN': 1.4, 'VERB': 1.5, 'ADJ': 1.2},
    {'NOUN': 1.2, 'VERB': 1.4, 'ADJ': 1.6},
]

def run_ngram_weight_experiments(dataset_name, references, hypotheses, ngram_weight_list, pos_weights=None):
    """
    다양한 n-gram 가중치 값으로 실험을 수행하여 최적의 가중치를 찾는 함수.
    """
    combined_scores = []  # 실험 결과 저장

    for i, ngram_weight in enumerate(ngram_weight_list):
        print(f"실험 {i + 1}: n-gram weight = {ngram_weight}")

        # Combined Metric 계산
        combined_score = calculate_combined_metric(references, hypotheses, ngram_weight=ngram_weight, max_n=4,
                                                   pos_weights=pos_weights)

        # 결과 저장
        combined_scores.append(combined_score)
        print(f"Combined Score {i + 1}: {combined_score}\n")

    # 실험 결과 시각화
    visualize_ngram_weight_experiment_results(dataset_name, ngram_weight_list, combined_scores)


def run_weight_experiments_with_visualization(dataset_name, references, hypotheses, pos_weights_list):
    combined_scores = []  # 실험 결과 저장

    for i, pos_weights in enumerate(pos_weights_list):
        print(f"실험 {i + 1}: 품사별 가중치 {pos_weights}")

        # Combined Metric 계산
        combined_score = calculate_combined_metric(dataset_name, references, hypotheses, pos_weights=pos_weights)

        # 결과 저장
        combined_scores.append(combined_score)
        print(f"Combined Score {i + 1}: {combined_score}\n")

    # 실험 결과 시각화
    visualize_weight_experiment_results(dataset_name, pos_weights_list, combined_scores)

# start_time = datetime.now()
# logging.info("프로그램 실행 시작")

# WMT 데이터셋 불러오기
wmt_dataset = load_wmt_dataset()
wmt_english_data = [item['en'] for item in wmt_dataset]
wmt_references = translate_wmt_english_to_german(wmt_english_data) # 영어 문장을 독일어로 번역
wmt_hypotheses = [item['de'] for item in wmt_dataset] # 독일어 문장
save_dataset_to_folder((wmt_references, wmt_hypotheses), "WMT")

# SQuAD 데이터셋 불러오기
squad_contexts, squad_questions, squad_answers = load_squad_dataset()
squad_references = [answer['text'][0] for answer in squad_answers]  # 정답 텍스트 리스트 추출
squad_hypotheses = squad_contexts  # 문맥이 번역본이라고 가정
save_dataset_to_folder((squad_contexts, squad_references, squad_hypotheses), "SQuAD")

# CNN/DAILYMAIL 데이터셋 불러오기
cnn_articles, cnn_highlights = load_cnn_daily_dataset()
cnn_references = cnn_highlights  # 요약문이 정답이라고 가정
cnn_hypotheses = cnn_articles  # 기사 본문이 번역본이라고 가정
save_dataset_to_folder((cnn_articles, cnn_references), "CNN_DailyMail")

# 데이터 양을 줄이기 위해 샘플링 (예: 처음 100개만 사용)
num_samples = 100
wmt_references = wmt_references[:num_samples]
wmt_hypotheses = wmt_hypotheses[:num_samples]
squad_references = squad_references[:num_samples]
squad_hypotheses = squad_hypotheses[:num_samples]
cnn_references = cnn_references[:num_samples]
cnn_hypotheses = cnn_hypotheses[:num_samples]

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
# run_ngram_weight_experiments("WMT", wmt_references, wmt_hypotheses, ngram_weight_list, pos_weights={'NOUN': 1.2, 'VERB': 1.1, 'ADJ': 1.0})
#
# print("Running SQuAD n-gram weight experiments...")
# run_ngram_weight_experiments("SQuAD", squad_references, squad_hypotheses, ngram_weight_list, pos_weights={'NOUN': 1.2, 'VERB': 1.1, 'ADJ': 1.0})
#
# print("Running CNN n-gram weight experiments...")
# run_ngram_weight_experiments("CNN/DailyMail", cnn_references, cnn_hypotheses, ngram_weight_list, pos_weights={'NOUN': 1.2, 'VERB': 1.1, 'ADJ': 1.0})


# WMT 점수 계산
print("Calculating WMT scores...")
wmt_bleu = calculate_bleu(wmt_references, wmt_hypotheses)
wmt_rouge = calculate_rouge(wmt_references, wmt_hypotheses)
wmt_bert = calculate_bert_score(wmt_references, wmt_hypotheses)
wmt_combined = calculate_combined_metric("WMT", wmt_references, wmt_hypotheses)

# SQuAD 점수 계산
print("Calculating SQuAD scores...")
squad_bleu = calculate_bleu(squad_references, squad_hypotheses)
squad_rouge = calculate_rouge(squad_references, squad_hypotheses)
squad_bert = calculate_bert_score(squad_references, squad_hypotheses)
squad_combined = calculate_combined_metric("SQuAD", squad_references, squad_hypotheses)

# CNN 점수 계산
print("Calculating CNN/DailyMail scores...")
cnn_bleu = calculate_bleu(cnn_references, cnn_hypotheses)
cnn_rouge = calculate_rouge(cnn_references, cnn_hypotheses)
cnn_bert = calculate_bert_score(cnn_references, cnn_hypotheses)
cnn_combined = calculate_combined_metric("CNN/DailyMail", cnn_references, cnn_hypotheses)

# 각 데이터셋의 점수 리스트
bleu_scores = [wmt_bleu, squad_bleu, cnn_bleu]
rouge_scores = [wmt_rouge, squad_rouge, cnn_rouge]
bert_scores = [wmt_bert, squad_bert, cnn_bert]
combined_scores = [wmt_combined, squad_combined, cnn_combined]

# CSV 파일 생성을 위한 모듈 추가
import csv

# Combined scores 리스트
dataset_names = ['wmt', 'squad', 'cnn']

# 파일을 저장할 디렉토리
output_dir = 'combined_scores'


# CSV 파일 생성 함수
def save_combined_score(dataset_name, score):
    # 파일 경로 설정
    output_file = f'{output_dir}/{dataset_name}_combined_score.csv'

    # CSV 파일 생성 및 2행 2열에 점수 저장
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 첫 번째 행과 두 번째 행을 채우기 위한 빈 리스트
        writer.writerow(['Filename', 'Score'])
        writer.writerow(['', score])


# 각 데이터셋에 대해 CSV 파일 생성
for dataset_name, score in zip(dataset_names, combined_scores):
    save_combined_score(dataset_name, score)

# 점수 시각화
visualize_scores(bleu_scores, rouge_scores, bert_scores, combined_scores)

# end_time = datetime.now()
# logging.info("프로그램 종료")
#
# # 실행 시간 계산 및 출력
# elapsed_time = end_time - start_time
# logging.info(f"프로그램 총 실행 시간: {elapsed_time}")
