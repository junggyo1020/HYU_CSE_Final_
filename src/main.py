# main.py

from dataset_loader import load_wmt_dataset, load_squad_dataset, load_cnn_daily_dataset
from dataset_to_txt import save_and_compress_dataset_to_zip
from evaluate_metrics import calculate_bleu, calculate_rouge, calculate_bert_score, calculate_combined_metric
from visualize import visualize_scores, visualize_ngram_weight_experiment_results, visualize_weight_experiment_results
from wmt_trans import translate_wmt_english_to_german

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
        combined_score = calculate_combined_metric(references, hypotheses, ngram_weight=0.3, max_n=4,
                                                   pos_weights=pos_weights)

        # 결과 저장
        combined_scores.append(combined_score)
        print(f"Combined Score {i + 1}: {combined_score}\n")

    # 실험 결과 시각화
    visualize_weight_experiment_results(dataset_name, pos_weights_list, combined_scores)

# WMT 데이터셋 불러오기
wmt_dataset = load_wmt_dataset()
wmt_english_data = [item['en'] for item in wmt_dataset]
wmt_references = translate_wmt_english_to_german(wmt_english_data) # 영어 문장을 독일어로 번역
wmt_hypotheses = [item['de'] for item in wmt_dataset] # 독일어 문장
# save_and_compress_dataset_to_zip((wmt_references, wmt_hypotheses), "wmt")  # WMT 데이터셋 텍스트 파일로 저장

# SQuAD 데이터셋 불러오기
squad_contexts, squad_questions, squad_answers = load_squad_dataset()
squad_references = [answer['text'][0] for answer in squad_answers]  # 정답 텍스트 리스트 추출
squad_hypotheses = squad_contexts  # 문맥이 번역본이라고 가정
# save_and_compress_dataset_to_zip((squad_contexts, squad_questions, squad_answers), "squad")  # SQuAD 데이터셋 텍스트 파일로 저장

# CNN/DAILYMAIL 데이터셋 불러오기
cnn_articles, cnn_highlights = load_cnn_daily_dataset()
cnn_references = cnn_highlights  # 요약문이 정답이라고 가정
cnn_hypotheses = cnn_articles  # 기사 본문이 번역본이라고 가정
# save_and_compress_dataset_to_zip((cnn_articles, cnn_highlights), "cnn")  # CNN/DailyMail 데이터셋 텍스트 파일로 저장

# 데이터 양을 줄이기 위해 샘플링 (예: 처음 100개만 사용)
num_samples = 100
wmt_references = wmt_references[:num_samples]
wmt_hypotheses = wmt_hypotheses[:num_samples]
squad_references = squad_references[:num_samples]
squad_hypotheses = squad_hypotheses[:num_samples]
cnn_references = cnn_references[:num_samples]
cnn_hypotheses = cnn_hypotheses[:num_samples]

# 품사 가중치 실험
print("Running WMT weight experiments...")
run_weight_experiments_with_visualization("WMT", wmt_references, wmt_hypotheses, pos_weights_list)

print("Running SQuAD weight experiments...")
run_weight_experiments_with_visualization("SQuAD", squad_references, squad_hypotheses, pos_weights_list)

print("Running CNN/DailyMail weight experiments...")
run_weight_experiments_with_visualization("CNN/DailyMail", cnn_references, cnn_hypotheses, pos_weights_list)

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
wmt_combined = calculate_combined_metric(wmt_references, wmt_hypotheses)

# SQuAD 점수 계산
print("Calculating SQuAD scores...")
squad_bleu = calculate_bleu(squad_references, squad_hypotheses)
squad_rouge = calculate_rouge(squad_references, squad_hypotheses)
squad_bert = calculate_bert_score(squad_references, squad_hypotheses)
squad_combined = calculate_combined_metric(squad_references, squad_hypotheses)

# CNN 점수 계산
print("Calculating CNN/DailyMail scores...")
cnn_bleu = calculate_bleu(cnn_references, cnn_hypotheses)
cnn_rouge = calculate_rouge(cnn_references, cnn_hypotheses)
cnn_bert = calculate_bert_score(cnn_references, cnn_hypotheses)
cnn_combined = calculate_combined_metric(cnn_references, cnn_hypotheses)

# 각 데이터셋의 점수 리스트
bleu_scores = [wmt_bleu, squad_bleu, cnn_bleu]
rouge_scores = [wmt_rouge, squad_rouge, cnn_rouge]
bert_scores = [wmt_bert, squad_bert, cnn_bert]
combined_scores = [wmt_combined, squad_combined, cnn_combined]

# 점수 시각화
visualize_scores(bleu_scores, rouge_scores, bert_scores, combined_scores)