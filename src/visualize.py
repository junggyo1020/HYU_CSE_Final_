# visualize.py

import matplotlib.pyplot as plt


def visualize_scores(bleu_scores, rouge_scores, bert_scores, combined_scores):
    """
    WMT, SQuAD, CNN 데이터셋에 대한 BLEU, ROUGE, BERTScore, Combined Metric을 막대 차트로 시각화
    """

    labels = ['WMT-BLEU', 'WMT-ROUGE-L', 'WMT-BERTScore', 'WMT-Combined',
              'SQuAD-BLEU', 'SQuAD-ROUGE-L', 'SQuAD-BERTScore', 'SQuAD-Combined',
              'CNN-BLEU', 'CNN-ROUGE-L', 'CNN-BERTScore', 'CNN-Combined']

    scores = bleu_scores + rouge_scores + bert_scores + combined_scores

    plt.figure(figsize=(14, 6))
    plt.bar(labels, scores, color=['blue', 'green', 'orange', 'purple'] * 3)
    plt.ylabel('Scores')
    plt.title('Comparison of Evaluation Metrics Across Datasets')

    for i, score in enumerate(scores):
        plt.text(i, score + 0.01, f'{score:.2f}', ha='center', fontsize=9)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
