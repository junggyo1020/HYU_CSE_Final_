import matplotlib.pyplot as plt


def visualize_scores(bleu_scores, rouge_scores, bert_scores, weighted_bleu_scores):
    """
    WMT, SQuAD, CNN 데이터셋에 대한 BLEU, ROUGE, BERTScore, Weighted BLEU를 막대 차트로 시각화

    :param bleu_scores: 각 데이터셋의 BLEU 점수 리스트
    :param rouge_scores: 각 데이터셋의 ROUGE-L (F1) 점수 리스트
    :param bert_scores: 각 데이터셋의 BERTScore 점수 리스트
    :param weighted_bleu_scores: 각 데이셋의 Weighted BLEU 점수 리스트
    """

    # 데이터셋별 점수 리스트 (순서: WMT, SQuAD, CNN)
    labels = ['WMT-BLEU', 'WMT-ROUGE-L', 'WMT-BERTScore', 'WMT-Weighted BLEU',
              'SQuAD-BLEU', 'SQuAD-ROUGE-L', 'SQuAD-BERTScore', 'SQuAD-Weighted BLEU',
              'CNN-BLEU', 'CNN-ROUGE-L', 'CNN-BERTScore', 'CNN-Weighted BLEU']

    scores = bleu_scores + rouge_scores + bert_scores + weighted_bleu_scores  # 각 점수 리스트 결합

    # 막대 차트 그리기
    plt.figure(figsize=(14, 6))
    plt.bar(labels, scores, color=['blue', 'green', 'orange', 'purple'] * 3)  # 각 데이터셋마다 다른 색상 적용
    plt.ylabel('Scores')
    plt.title('Comparison of Evaluation Metrics Across Datasets')

    # 각 막대 위에 점수 표시
    for i, score in enumerate(scores):
        plt.text(i, score + 0.01, f'{score:.2f}', ha='center', fontsize=9)

    plt.xticks(rotation=45, ha='right')  # x축 레이블을 회전하여 보기 쉽게 설정
    plt.tight_layout()  # 레이아웃을 자동으로 맞추기
    plt.show()
