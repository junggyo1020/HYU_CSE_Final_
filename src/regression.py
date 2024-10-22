import pandas as pd

# 데이터 로드
data = pd.read_csv('evaluation_metrics.csv') # evaluation_metrics.csv : 평가 지표 데이터
X = data[['BLEU', 'ROUGE', 'BERTScore', 'Combined', 'TAACO', 'TAASSC']]  # 독립 변수들
y = data['Target']  # 종속 변수 (타겟 값)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 데이터셋을 훈련용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 초기화 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 각 평가 지표에 대한 가중치 (회귀 계수) 출력
print("Weights for each metric: ", model.coef_)
