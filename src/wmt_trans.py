# wmt_trans.py
# WMT 데이터셋을 번역하는 모듈

# googletrans를 이용하여 영어 데이터를 독일어로 번역하는 함수
from googletrans import Translator

# Google Translate 객체 생성
translator = Translator()

def translate_wmt_german_to_english(wmt_data):
    translated_sentences = []

    for sentence in wmt_data:
        # 독일어를 영어로 번역
        translated = translator.translate(sentence, src='de', dest='en')
        translated_sentences.append(translated.text)

    return translated_sentences