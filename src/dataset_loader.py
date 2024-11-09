# dataset_loader.py

from datasets import load_dataset

def load_wmt_dataset():
    print("Loading WMT dataset...")
    dataset = load_dataset("wmt14", "de-en", split="test[:100]")  # 데이터 100개만 불러오기
    print("WMT dataset loaded.")
    return dataset['translation']

def load_squad_dataset():
    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad", split="validation[:100]")
    print("SQuAD dataset loaded.")
    return dataset['context'], dataset['question'], dataset['answers']

def load_cnn_daily_dataset():
    print("Loading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:100]")
    print("CNN/DailyMail dataset loaded.")
    return dataset['article'], dataset['highlights']
