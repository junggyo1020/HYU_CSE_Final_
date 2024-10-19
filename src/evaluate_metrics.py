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

# 품사별 가중치 설정 (명사: 1.5, 동사: 1.3, 형용사: 1.2)
POS_WEIGHTS = {
    'NOUN': 1.5,
    'VERB': 1.3,
    'ADJ': 1.2
}

def calculate_bleu(references, hypotheses):
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


def calculate_rouge(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    rouge_l_f1 = np.mean([score['rougeL'].fmeasure for score in scores])
    return rouge_l_f1


def calculate_bert_score(references, hypotheses):
    P, R, F1 = score(hypotheses, references, lang="en", verbose=True)
    return F1.mean().item()

def calculate_weighted_ngram_match(reference, candidate, n):
    """
    품사별 가중치를 고려한 N-그램 일치율 계산
    """
    ref_doc = nlp(reference)
    cand_doc = nlp(candidate)

    # 참조 문장과 후보 문장의 N-그램 추출
    ref_ngrams = list(ngrams([token.text for token in ref_doc], n))
    cand_ngrams = list(ngrams([token.text for token in cand_doc], n))

    ref_counter = Counter(ref_ngrams)
    cand_counter = Counter(cand_ngrams)

    overlap = sum((cand_counter & ref_counter).values())
    total = sum(ref_counter.values())

    # 품사별 가중치 적용
    weighted_overlap = 0
    weighted_total = 0
    for ngram in ref_ngrams:
        ngram_pos_tags = [token.pos_ for token in nlp(' '.join(ngram))]
        weight = np.mean([POS_WEIGHTS.get(pos, 1.0) for pos in ngram_pos_tags])  # 기본 가중치는 1.0
        weighted_total += weight
        if ngram in cand_ngrams:
            weighted_overlap += weight

    if weighted_total == 0:
        return 0.0
    return weighted_overlap / weighted_total


def calculate_all_ngram_match_with_pos(references, hypotheses, max_n=4):
    """
    품사 가중치를 고려한 1-그램부터 N-그램까지의 평균 점수 계산
    """
    all_ngram_scores = []

    for n in range(1, max_n + 1):
        ngram_scores = []
        for ref, hyp in zip(references, hypotheses):
            ngram_scores.append(calculate_weighted_ngram_match(ref, hyp, n=n))
        all_ngram_scores.append(np.mean(ngram_scores))

    return np.mean(all_ngram_scores)


def calculate_combined_metric(references, hypotheses, ngram_weight=0.3, max_n=4):
    """
    N-그램 일치율 (품사 기반 가중치 포함)과 BERTScore를 결합한 평가 메트릭
    """
    # BERTScore 계산
    P, R, F1 = score(hypotheses, references, lang="en", verbose=True)
    bert_f1 = F1.mean().item()

    # 품사 가중치를 포함한 N-그램 일치율 계산
    combined_ngram_score = calculate_all_ngram_match_with_pos(references, hypotheses, max_n=max_n)

    # 문장의 길이에 따라 가중치 동적 조정
    avg_sentence_length = np.mean([len(ref.split()) for ref in references])
    if avg_sentence_length > 15:  # 문장이 길면 BERTScore에 더 높은 가중치 부여
        dynamic_ngram_weight = max(ngram_weight - 0.1, 0.1)
    else:  # 문장이 짧으면 N-그램에 더 높은 가중치 부여
        dynamic_ngram_weight = min(ngram_weight + 0.1, 0.5)

    # 최종 결합 점수 계산
    combined_score = (1 - dynamic_ngram_weight) * bert_f1 + dynamic_ngram_weight * combined_ngram_score
    return combined_score


# def calculate_ngram_match(reference, candidate, n):
#     """
#     특정 N-그램 크기에서의 일치율을 계산하는 함수
#     """
#     ref_ngrams = list(ngrams(reference.split(), n))
#     cand_ngrams = list(ngrams(candidate.split(), n))
#     ref_counter = Counter(ref_ngrams)
#     cand_counter = Counter(cand_ngrams)
#     overlap = sum((cand_counter & ref_counter).values())
#     total = sum(ref_counter.values())
#     if total == 0:
#         return 0.0
#     return overlap / total
#
#
# def calculate_all_ngram_match(references, hypotheses, max_n=4):
#     """
#     1-그램부터 N-그램까지 모두 계산하여 평균 점수를 반환하는 함수
#     """
#     all_ngram_scores = []
#
#     for n in range(1, max_n + 1):
#         ngram_scores = []
#         for ref, hyp in zip(references, hypotheses):
#             ngram_scores.append(calculate_ngram_match(ref, hyp, n=n))
#         all_ngram_scores.append(np.mean(ngram_scores))
#
#     # 1-그램, 2-그램, 3-그램, 4-그램의 평균 점수 반환
#     return np.mean(all_ngram_scores)
#
#
# def calculate_combined_metric(references, hypotheses, ngram_weight=0.3, max_n=4):
#     """
#        1-그램부터 4-그램까지 모두 활용한 N-그램 일치율과 BERTScore를 결합하는 함수
#        """
#     # BERTScore 계산
#     P, R, F1 = score(hypotheses, references, lang="en", verbose=True)
#     bert_f1 = F1.mean().item()
#
#     # 1-그램부터 max_n-그램까지의 평균 N-그램 일치율 계산
#     mean_ngram_score = calculate_all_ngram_match(references, hypotheses, max_n=max_n)
#
#     # 문장의 길이에 따라 가중치 동적 조정
#     avg_sentence_length = np.mean([len(ref.split()) for ref in references])
#     if avg_sentence_length > 15:  # 문장이 길면 BERTScore에 더 높은 가중치 부여
#         dynamic_ngram_weight = max(ngram_weight - 0.1, 0.1)
#     else:  # 문장이 짧으면 N-그램에 더 높은 가중치 부여
#         dynamic_ngram_weight = min(ngram_weight + 0.1, 0.5)
#
#     combined_score = (1 - dynamic_ngram_weight) * bert_f1 + dynamic_ngram_weight * mean_ngram_score
#     return combined_score