# main.py

from dataset_loader import load_wmt_dataset, load_squad_dataset, load_cnn_daily_dataset
from evaluate_metrics import calculate_bleu, calculate_rouge, calculate_bert_score, calculate_combined_metric
from visualize import visualize_scores

# WMT 데이터셋 불러오기
wmt_dataset = load_wmt_dataset()
wmt_references = [item['en'] for item in wmt_dataset]
wmt_hypotheses = [item['de'] for item in wmt_dataset]  # 예시로 독일어 문장을 번역한 것이라고 가정

# SQuAD 데이터셋 불러오기
squad_contexts, squad_questions, squad_answers = load_squad_dataset()
squad_references = [answer['text'][0] for answer in squad_answers]  # 정답 텍스트 리스트 추출
squad_hypotheses = squad_contexts  # 문맥이 번역본이라고 가정

# CNN/DAILYMAIL 데이터셋 불러오기
cnn_articles, cnn_highlights = load_cnn_daily_dataset()
cnn_references = cnn_highlights  # 요약문이 정답이라고 가정
cnn_hypotheses = cnn_articles  # 기사 본문이 번역본이라고 가정

# 데이터 양을 줄이기 위해 샘플링 (예: 처음 100개만 사용)
num_samples = 100
wmt_references = wmt_references[:num_samples]
wmt_hypotheses = wmt_hypotheses[:num_samples]
squad_references = squad_references[:num_samples]
squad_hypotheses = squad_hypotheses[:num_samples]
cnn_references = cnn_references[:num_samples]
cnn_hypotheses = cnn_hypotheses[:num_samples]

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
