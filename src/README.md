TAACO와 TAASSC에서는 개별 텍스트 데이터를 전부 한 텍스트 파일에 모아서 분석하면, 텍스트의 응집성과 다양성 평가에서 의도치 않은 결과를 낳게 됨.
예를 들어서 설명하면, CNN의 경우 각각의 기사에서, 기사의 원문과 요약문이 얼마나 잘 응집되어 있는 지(유사한지), 한 기사 안에서 얼마나 다양한
어휘들이 사용되었는 지를 평가하는 것이 중요함. 그러나, 이러한 평가를 위해서는 각각의 기사를 개별적으로 분석해야 함. 그렇기 때문에 데이터들을
개별의 텍스트로 분리하여 저장해야 함.

데이터셋을 1%만 사용했는데도, main.py에서만 1분 이상의 시간이 소요됨. 이후 교수님께서 자동화에 대해 지적한다면, TAACO와 TASSC을 수정하여
자동화를 진행할 예정임.

main.py에서 일단 전환한 텍스트 데이터들을 저장하고 zip으로 압축함. 하지만 TAACO와 TAASSC에서는 압축된 파일을 읽지 못함. TAAACO와
TAASSC에서는 압축된 파일을 읽을 수 있도록 코드를 수정하거나, 압축된 파일을 풀어서 읽을 수 있도록 해야 함. 또는 main.py를 수정하여
텍스트 파일들을 저장할 때, 압축하지 않고 저장할 수도 있음.

수정 전, 기존의 WMT에 대한 평가의 결과 점수가 낮게 나온 이유는, BERT를 포함한 평가 지표들은 일반적으로 같은 언어 데이터에 대해서 유사성을
평가하는 것이기 때문임. 그러나, 기존의 코드에서는 다른 언어에 대한 번역을 평가하는 것이기 때문에, 점수가 낮게 나온 것임. 이를 해결하기 위해서,
기존의 코드를 수정하여, 번역된 데이터에 대한 평가(독일어로 번역된 WMT의 영어와 WMT의 독일어를 비교)를 진행함. 그런데 BLEU와 Combined Score
의 점수는 5배 가까이 높아졌지만, 다른 지표들의 값은 전혀 변하지 않았음. 이유는 아직 모르겠음.

회귀 분석을 하기 위해선, TAACO와 TAASSC의 CSV 파일의 결과 값들을 수학적으로 산술 평균을 내야함. CSV 파일에서 셀을 이용하여 직접 값을
계산하든가, 결과로 나온 CSV 파일을 처리하여 수학적 산술 평균을 진행하는 개별 파이썬 코드를 작성해야 함. 아니면 TAACO와 TAASSC의 코드를
수정하여, 자동으로 수학적 산술 평균을 계산하게끔 할 수도 있음.

TAACO와 TAASSC의 GUI에서 선택해야할 옵션들의 리스트 :

TAACO
Lemma tokens to analyze for lexical overlap and TTR: 'All'
Lexical overlap: 'Sentence', 'Paragraph', 'Adjacent', 'Adjacent 2'
Other indices: 'TTR'

TAASSC
Clause Complexity, Phrase Complexity, Syntactic Sophistication, Syntactic Components

TAACO 가중치 설정 관련 : 서브 데이터셋(전체 데이터셋의 1/100)의 결과값을 평균을 냄. 하나의 지표 벡터가 나옴. 설정한 우선순위에 맞춰 임의로
(고정된 수치의) 가중치를 부여함.

TAACO의 지표 벡터 : 1. Lexical Overlap, 2. TTR, 3. Sentence Overlap, 4. Paragraph Overlap, 5. Adjacent Overlap, 6. Adjacent 2 Overlap

cobined_Score, TAACO_Score, TAASSC_Score;

final_score = a * combined_Score + b * TAACO_Score + c * TAASSC_Score
final_score = 0.4 * combined_Score + 0.3 * TAACO_Score + 0.3 * TAASSC_Score

지금 해야할 것 :
1. 텍스트 파일들의 디렉토리를 압축하지 않게 수정 -정교
2. TAACO와 TAASSC의 결과인 각각의 csv 파일들에서 각 지표별 평균값 추출하는 load_csv.py 구현 -지우
3. 우선순위에 따라서 부여한 임의의 고정 가중치를 곱한, TAACO_Score, TAASSC_Score를 구하는 파이썬 모듈 구현
4. cobined_Score, TAACO_Score, TAASSC_Score의 임의의 고정된 가중치를 곱하여 final_score를 구하는 파이썬 모듈 구현

TAASSC 결과 지표의 F, H, T, Y, AF
