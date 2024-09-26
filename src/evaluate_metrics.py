import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# # spacy 모델 로드
# nlp = spacy.load("en_core_web_sm")

def calculate_bleu(references, hypotheses):
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score

# def calculate_rouge(references, hypotheses):
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#     scores = [scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
#     return scores

def calculate_rouge(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    # 평균 ROUGE-L F1 점수 계산
    rouge_l_f1 = np.mean([score['rougeL'].fmeasure for score in scores])
    return rouge_l_f1

def calculate_bert_score(references, hypotheses):
    P, R, F1 = score(hypotheses, references, lang="en", verbose=True)
    return F1.mean().item()

def calculate_weighted_bleu(references, hypotheses):
    """
    텍스트 임베딩과 코사인 유사도를 활용한 가중 BLEU 점수 계산
    """
    # BERT 모델과 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()  # 평가 모드로 설정

    # 모든 참조 및 가설 문장에 대해 임베딩 계산
    reference_embeddings = []
    hypothesis_embeddings = []

    with torch.no_grad():
        for ref, hyp in zip(references, hypotheses):
            # 토크나이즈 및 텐서 변환
            ref_tokens = tokenizer(ref, return_tensors='pt', truncation=True, padding=True)
            hyp_tokens = tokenizer(hyp, return_tensors='pt', truncation=True, padding=True)

            # 임베딩 계산
            ref_embedding = model(**ref_tokens).last_hidden_state.mean(dim=1)
            hyp_embedding = model(**hyp_tokens).last_hidden_state.mean(dim=1)

            reference_embeddings.append(ref_embedding)
            hypothesis_embeddings.append(hyp_embedding)

    # 코사인 유사도 계산
    similarities = []
    for ref_emb, hyp_emb in zip(reference_embeddings, hypothesis_embeddings):
        sim = cosine_similarity(ref_emb.numpy(), hyp_emb.numpy())[0][0]
        similarities.append(sim)

    # 평균 코사인 유사도 계산
    mean_similarity = np.mean(similarities)

    # 기존 BLEU 점수 계산
    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score

    # 가중치를 적용한 BLEU 점수 계산
    weighted_bleu = bleu * mean_similarity

    return weighted_bleu