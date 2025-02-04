# 필수 라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek
from collections import Counter

# 1. CSV 파일에서 데이터 로드
data_path = './svm_data.csv'  # CSV 파일 경로를 입력하세요.
df = pd.read_csv(data_path)

# 데이터 로드 확인
print(f"데이터 shape: {df.shape}")

# 2. 데이터 전처리
# 레이블 인코딩 (up → 1, down → 0)
df['label'] = df['label'].map({'up': 1, 'down': 0})

# 레이블 인코딩 확인
print("고유 레이블:", df['label'].unique())

# 특징과 레이블 분리
X = df[['feature1', 'feature2', 'feature3']]
y = df['label']

# 클래스 분포 확인
print("원본 클래스 분포:")
print(Counter(y))

# 데이터 품질 검증
MIN_SAMPLES = 50
if len(X) < MIN_SAMPLES:
    print(f"경고: 데이터셋 크기가 너무 작습니다. 현재 {len(X)}개, 최소 {MIN_SAMPLES}개 권장")
    print("소규모 데이터셋에 대한 처리를 진행합니다.")

    # 부트스트랩 리샘플링
    n_bootstrap = max(MIN_SAMPLES, len(X) * 2)
    bootstrap_indices = np.random.choice(len(X), size=n_bootstrap, replace=True)
    X = X.iloc[bootstrap_indices]
    y = y.iloc[bootstrap_indices]

    print(f"부트스트랩 리샘플링 후 데이터 크기: {len(X)}")

# 클래스 불균형 처리
if len(set(y)) > 1:  # 클래스가 2개 이상인 경우에만 실행
    min_class_count = min(Counter(y).values())
    if min_class_count < 2:
        print("RandomOverSampler를 사용하여 데이터 균형을 맞춥니다.")
        ros = RandomOverSampler(random_state=44)
        X, y = ros.fit_resample(X, y)
    else:
        print("SMOTETomek를 사용하여 데이터 균형을 맞춥니다.")
        smote_tomek = SMOTETomek(random_state=44)
        X, y = smote_tomek.fit_resample(X, y)

    print("리샘플링 후 클래스 분포:")
    print(Counter(y))
else:
    print("경고: 단일 클래스만 존재합니다. 분류 작업을 진행할 수 없습니다.")
    exit()

# 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

# 4. 랜덤 포레스트 모델 학습
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=44, class_weight='balanced')
model.fit(X_train, y_train)

# 5. 예측 및 정확도 계산
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# 분류 보고서 출력
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# 7. 특성 중요도 시각화
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12, 6))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Model')
plt.show()

# 8. 교차 검증 점수 계산
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"\nCross-validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# 9. 학습 곡선 시각화
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5,
         label='Training accuracy')
plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15,
                 color='blue')
plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')
plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15,
                 color='green')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend(loc='lower right')
plt.show()

print("\n주의: 이 결과는 소규모 데이터셋에 기반하므로 신뢰성이 제한적일 수 있습니다.")
print("더 신뢰할 수 있는 결과를 위해 더 많은 데이터를 수집하는 것을 권장합니다.")
