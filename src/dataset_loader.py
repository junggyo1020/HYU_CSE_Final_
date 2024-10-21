# dataset_loader.py

from datasets import load_dataset

def load_wmt_dataset():
    print("Loading WMT dataset...")
    dataset = load_dataset("wmt14", "de-en", split="test[:1%]")  # 데이터의 1%만 불러오기 : 테스트용
    print("WMT dataset loaded.")
    return dataset['translation']

def load_squad_dataset():
    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad", split="validation[:10%]")  # 데이터의 10%만 불러오기
    print("SQuAD dataset loaded.")
    return dataset['context'], dataset['question'], dataset['answers']

def load_cnn_daily_dataset():
    print("Loading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:10%]")  # 데이터의 10%만 불러오기
    print("CNN/DailyMail dataset loaded.")
    return dataset['article'], dataset['highlights']
