# 🚦 YOLOv8 기반 도로 표지판 객체 탐지 (Road Sign Detection)

이 프로젝트는 Kaggle의 도로 표지판 데이터셋을 활용하여, 인공지능 모델인 **YOLOv8**이 실시간으로 표지판을 얼마나 잘 찾아내는지 학습하고 테스트한 결과물입니다.

---

## 📌 과제 개요
- **목표**: YOLOv8 모델을 활용한 도로 표지판 객체 탐지(Object Detection) 프로세스 완수
- **데이터셋**: [Kaggle Road Sign Detection](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection)
- **주요 내용**: 데이터 전처리, YOLOv8 모델 학습, 학습된 가중치를 이용한 예측(Predict)

---

## 🛠 실습 과정

### 1. 데이터셋 준비 (Dataset Preparation)
Kaggle에서 다운로드한 데이터를 YOLOv8 형식에 맞춰 `images`와 `labels` 폴더로 구분하고, 학습용(train)과 검증용(val) 데이터로 나누어 정리했습니다.

* **폴더 구조**:
    ```text
    /content/pascal_datasets/
    ├── VOC/
    │   ├── images/ (train2007, val2007)
    │   └── labels/ (train2007, val2007)
    └── data.yaml
    ```

### 2. 모델 학습 (Training)
초보 로봇(Pre-trained YOLOv8s)에게 도로 표지판 사진과 정답지를 주고 100번 반복 학습을 시켰습니다.

```python
from ultralytics import YOLO

# 1. 모델 로드 (yolov8s 사용)
model = YOLO("yolov8s.pt")

# 2. 학습 시작
model.train(
    data="/content/pascal_datasets/custom_voc.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    name="road_sign_custom"
)

3. 객체 탐지 테스트 (Inference)
학습 결과물인 best.pt를 사용하여 모델이 한 번도 본 적 없는 새로운 도로 사진에서 표지판을 찾게 했습니다.

# 가장 성적이 좋았던 지식(가중치) 불러오기
model = YOLO("runs/detect/road_sign_custom/weights/best.pt")

# 예측 수행
results = model.predict(
    source="/content/pascal_datasets/VOC/images/custom2007",
    conf=0.25,
    save=True
)

📊 결과 분석
✅ 탐지 결과 시각화

<img src="객체탐지이미지.png">
