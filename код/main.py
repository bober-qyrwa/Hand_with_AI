# Обучение на hand-keypoints.yaml (Ultralytics YOLO) [web:1][web:7]
from ultralytics import YOLO

# Загрузка предобученной модели
model_hand = YOLO('yolov8n-pose.pt')

# Обучение на датасете hand-keypoints (27k изображений, 21 keypoints)
results = model_hand.train(
    data='hand-keypoints.yaml',  # 27000 изображений кистей [web:1][web:8]
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)

# Инференс на видео
results = model_hand.predict('video.mp4', save=True)
# Выход: 21 точка на кисти (запястье + 20 суставов пальцев)


# Обучение на COCO-pose (17 keypoints тела) [web:3][web:9]
model_coco = YOLO('yolov8n-pose.pt')

results = model_coco.train(
    data='coco-pose.yaml',  # 60k изображений, фокус на локоть/плечо [web:3]
    epochs=100,
    imgsz=640,
    task='pose'
)

# Комбинированный инференс
results_hand = model_hand('frame.jpg')
results_body = model_coco('frame.jpg')
# Извлечение локтевого/плечевого сустава (keypoints 5,6,7)
elbow_pos = results_body[0].keypoints.xy[0][5]  # COCO elbow [web:9]
