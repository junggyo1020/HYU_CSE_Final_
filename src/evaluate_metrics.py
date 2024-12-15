# evaluate_metrics.py

import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score
import numpy as np
from nltk.util import ngrams
from collections import Counter
import spacy

# spaCy 모델 불러오기 (영어)
nlp = spacy.load("en_core_web_sm")

# 품사별 가중치 실험 이후 최적의 가중치 설정 (명사: 1.2, 동사: 1.1, 형용사: 1.0)
POS_WEIGHTS = {'NOUN': 1.2, 'VERB': 1.1, 'ADJ': 1.0}

def calculate_bleu(references, hypotheses):
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score / 100 # 백분율로 변환

def calculate_rouge(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    rouge_l_f1 = np.mean([score['rougeL'].fmeasure for score in scores])
    return rouge_l_f1

def calculate_bert_score(references, hypotheses):
    P, R, F1 = score(hypotheses, references, lang="en", verbose=True)
    return F1.mean().item()

def calculate_weighted_ngram_match(reference, candidate, n, pos_weights):
    """
    품사별 가중치를 고려한 N-그램 일치율 계산
    일치하는 n-그램에 대해서만 해당 품사의 가중치를 적용하여 계산합니다.
    """
    ref_doc = nlp(reference)
    cand_doc = nlp(candidate)

    # 단어별 토큰화 및 품사 추출
    ref_tokens = [(token.text, token.pos_) for token in ref_doc]
    cand_tokens = [(token.text, token.pos_) for token in cand_doc]

    # 참조 문장과 후보 문장의 N-그램 생성
    ref_ngrams = list(ngrams(ref_tokens, n))
    cand_ngrams = list(ngrams(cand_tokens, n))


    ref_counter = Counter(ref_ngrams)
    cand_counter = Counter(cand_ngrams)

    overlap = 0
    total_weight = 0

    # 참조 문장의 n-그램과 후보 문장의 n-그램 일치 계산
    for ngram in ref_counter:
        weight = 1.0  # 기본 가중치는 1.0
        for _, pos in ngram:  # n-gram 내 각 단어의 품사를 확인
            if pos in pos_weights:
                weight *= pos_weights[pos]  # 품사에 해당하는 가중치 곱하기

        total_weight += ref_counter[ngram] * weight  # 전체 가중치 계산 (빈도 × 품사 가중치)

        # n-그램이 후보 문장에도 있을 경우
        if ngram in cand_counter:
            overlap += min(ref_counter[ngram], cand_counter[ngram]) * weight  # 일치하는 n-그램의 가중치를 포함한 점수

    if total_weight == 0:
        return 0.0

    return overlap / total_weight  # n-그램 일치율

def calculate_all_ngram_match_with_pos(references, hypotheses, max_n=4, pos_weights=None):
    """
    품사 가중치를 고려한 1-그램부터 N-그램까지의 평균 점수 계산
    """
    all_ngram_scores = []

    for n in range(1, max_n + 1):
        ngram_scores = [
            calculate_weighted_ngram_match(ref, hyp, n=n, pos_weights=pos_weights)
            for ref, hyp in zip(references, hypotheses)
        ]
        all_ngram_scores.append(np.mean(ngram_scores))

    return np.mean(all_ngram_scores)


def calculate_combined_metric(dataset_name, references, hypotheses, max_n=4, pos_weights=None):
    """
    N-그램 일치율 (품사 기반 가중치 포함)과 BERTScore를 결합한 평가 메트릭
    """
    # pos_weights가 None이면 기본 가중치 설정
    if pos_weights is None:
        pos_weights = POS_WEIGHTS

    # BERTScore 계산
    P, R, F1 = score(hypotheses, references, lang="en", verbose=True)
    bert_f1 = F1.mean().item()

    # 품사 가중치를 포함한 N-그램 일치율 계산
    combined_ngram_score = calculate_all_ngram_match_with_pos(references, hypotheses, max_n=max_n, pos_weights=pos_weights)

    # 데이터셋 별 가중치 조정
    if dataset_name == "WMT":
        ngram_weight = 0.4  # N-gram 40%, BERTScore 60%
    elif dataset_name == "SQuAD":
        ngram_weight = 0.1
    elif dataset_name == "CNN/DailyMail":
        ngram_weight = 0.1
    else:
        ngram_weight = 0.3  # STS dataset N-gram 가중치

    # 최종 결합 점수 계산
    combined_score = (1 - ngram_weight) * bert_f1 + ngram_weight * combined_ngram_score
    return combined_score