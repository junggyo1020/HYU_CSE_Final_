# visualize.py

import matplotlib.pyplot as plt

pos_weights_tag_list = ['Default', 'Noun', 'Noun&Verb', 'Verb', 'Adjective']

def visualize_scores(bleu_scores, rouge_scores, bert_scores, combined_scores):
    """
    WMT, SQuAD, CNN 데이터셋에 대한 BLEU, ROUGE, BERTScore, Combined Metric을 막대 차트로 시각화
    """

    labels = ['WMT-BLEU', 'SQuAD-BLEU', 'CNN-BLEU',
              'WMT-ROUGE-L', 'SQuAD-ROUGE-L', 'CNN-ROUGE-L',
              'WMT-BERTScore', 'SQuAD-BERTScore', 'CNN-BERTScore',
              'WMT-Combined', 'SQuAD-Combined', 'CNN-Combined']

    scores = bleu_scores + rouge_scores + bert_scores + combined_scores

    plt.figure(figsize=(14, 6))
    plt.bar(labels, scores, color=['blue', 'green', 'orange'] * 4)
    plt.ylabel('Scores')
    plt.title('Comparison of Evaluation Metrics Across Datasets')

    for i, score in enumerate(scores):
        plt.text(i, score + 0.01, f'{score:.2f}', ha='center', fontsize=9)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def visualize_ngram_weight_experiment_results(dataset_name, ngram_weight_list, combined_scores):
    """
    n-gram 가중치 실험 결과를 시각화하는 함수.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(ngram_weight_list, combined_scores, marker='o', color='blue')
    plt.xlabel('N-gram Pos Weight Experiment')
    plt.ylabel('Combined Score')
    plt.title(f'{dataset_name} Dataset - Combined Score by n-gram Weight')

    for i, score in enumerate(combined_scores):
        plt.text(ngram_weight_list[i], score + 0.01, f'{score:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

def visualize_weight_experiment_results(dataset_name, pos_weights_list, combined_scores):
    """
    품사 가중치 실험 결과를 꺾은선 그래프로 시각화하는 함수.
    각 데이터셋에서 품사 가중치에 따른 Combined Score를 시각화.
    """
    labels = [f"Exp {i + 1}:{pos_weights_tag_list[i]}" for i in range(len(pos_weights_list))]  # 실험 번호 레이블

    # 데이터의 차이가 너무 적어서 10배로 확대
    scaled_scores = [score * 10 for score in combined_scores]

    plt.figure(figsize=(10, 6))
    plt.plot(labels, scaled_scores, marker='o', color='purple')
    plt.xlabel('Experiment Number')
    plt.ylabel('Scaled Combined Score (x10)')
    plt.title(f'{dataset_name} Dataset - Combined Score by POS Weights (Scaled)')

    # 각 데이터 포인트에 점수 표시
    for i, score in enumerate(scaled_scores):
        plt.text(i, score + 0.0005, f'{combined_scores[i]:.4f}', ha='center', fontsize=9)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
