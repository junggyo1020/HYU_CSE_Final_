# wmt_trans.py
# WMT 데이터셋을 번역하는 모듈

# googletrans를 이용하여 영어 데이터를 독일어로 번역하는 함수
from googletrans import Translator

# Google Translate 객체 생성
translator = Translator()

def translate_wmt_english_to_german(wmt_english_data):
    translated_sentences = []

    for sentence in wmt_english_data:
        # 영어를 독일어로 번역
        translated = translator.translate(sentence, src='en', dest='de')
        translated_sentences.append(translated.text)

    return translated_sentences