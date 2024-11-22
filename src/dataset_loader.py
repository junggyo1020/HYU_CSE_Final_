# dataset_loader.py

from datasets import load_dataset
from transformers import pipeline

# QA 모델 로드
qa_model = pipeline("question-answering")

def load_wmt_dataset():
    print("Loading WMT dataset...")
    dataset = load_dataset("wmt14", "de-en", split="test[:100]")  # 데이터 100개만 불러오기
    print("WMT dataset loaded.")
    return dataset['translation']

def load_squad_dataset():
    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad", split="validation[:100]")
    print("SQuAD dataset loaded.")

    contexts = dataset['context']
    questions = dataset['question']
    original_answers = dataset['answers']

    generated_answers = []
    for context, question in zip(contexts, questions):
        result = qa_model(question=question, context=context)
        generated_answers.append(result['answer'])

    print("SQuAD dataset proceed with QA model.")
    return original_answers, generated_answers, contexts, questions

def load_cnn_daily_dataset():
    print("Loading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:100]")
    print("CNN/DailyMail dataset loaded.")
    return dataset['article'], dataset['highlights']

def load_sts_dataset():
    print("Loading STS dataset...")
    dataset = load_dataset("stsb_multi_mt", name="en", split="test[:100]")
    print("STS dataset loaded.")
    return dataset['sentence1'], dataset['sentence2']

# # 여러 개의 질문-지문 쌍을 일괄 처리
# data = [
#     {"context": "Stanford University is located in California.", "question": "Where is Stanford University located?"},
#     {"context": "The Eiffel Tower is in Paris.", "question": "Where is the Eiffel Tower?"},
#     # SQuAD 데이터셋의 다른 항목들도 추가 가능
# ]
#
# results = [qa_model(question=item["question"], context=item["context"])['answer'] for item in data]
# print(results)