import csv

# 샘플 데이터
data = [
    [1.2, 3.4, 5.6, 0],
    [2.3, 4.5, 6.7, 1],
    [3.4, 5.6, 7.8, 0],
    # ... 더 많은 데이터 행 추가
]

# CSV 파일로 저장
with open('svm_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['feature1', 'feature2', 'feature3', 'label'])  # 헤더 추가
    writer.writerows(data)
