# 각 기사를 개별 텍스트 파일로 저장하는 모듈
# 저장된 텍스트 파일은 후에 실행될 TAACO, TAASSC에서 사용될 예정

import os

def save_wmt_to_txt(dataset, output_dir="loaded_data_text"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "WMT_dataset.txt")

    with open(output_file, 'w') as f:
        for idx, data in enumerate(dataset):
            article_text = data.get('translation', {}).get('en', '')  # WMT 영어 번역 가져오기
            f.write(f"Translation {idx + 1}:\n")
            f.write(article_text + "\n\n")


def save_squad_to_txt(dataset, output_dir="loaded_data_text"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "SQuAD_dataset.txt")

    with open(output_file, 'w') as f:
        for idx, data in enumerate(dataset):
            article_text = data.get('context', '')  # SQuAD 문맥 텍스트 가져오기
            f.write(f"Context {idx + 1}:\n")
            f.write(article_text + "\n\n")


def save_cnn_to_txt(dataset, output_dir="loaded_data_text"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "CNN_dataset.txt")

    with open(output_file, 'w') as f:
        for idx, data in enumerate(dataset):
            article_text = data.get('article', '')  # CNN 기사 본문 가져오기
            f.write(f"Article {idx + 1}:\n")
            f.write(article_text + "\n\n")
