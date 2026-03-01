#запуск полной нейросети
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO

# =================== 1. ЗАГРУЗКА ВСЕХ МОДЕЛЕЙ ===================

device = 'cpu'
print("🚀 ЗАПУСК СИСТЕМЫ ОТСЛЕЖИВАНИЯ ВЗГЛЯДА")
print("="*60)

# Модель для детекции глаз (YOLO)
print("\n📦 Загрузка YOLO модели для глаз...")
eye_model = YOLO('best.pt')
eye_model.model.to(device)
print("✅ YOLO модель загружена")

# Модель для детекции зрачков (EyesNet)
class Reshaper(nn.Module):
    def __init__(self, target_shape):
        super(Reshaper, self).__init__()
        self.target_shape = target_shape
    def forward(self, input):
        return torch.reshape(input, (-1, *self.target_shape))

class EyesNet(nn.Module):
    def __init__(self):
        super(EyesNet, self).__init__()
        self.features_left = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            Reshaper([64])
        )
        self.features_right = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            Reshaper([64])
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )
    def forward(self, x_left, x_right):
        x_left = self.features_left(x_left)
        x_right = self.features_right(x_right)
        x = torch.cat((x_left, x_right), 1)
        x = self.fc(x)
        return x

print("\n📦 Загрузка модели для детекции зрачков...")
state_dict = torch.load('epoch_299.pth', map_location=device)
pupil_model = EyesNet().to(device)
pupil_model.load_state_dict(state_dict)
pupil_model.eval()
print("✅ Модель зрачков загружена")

# НАША ОБУЧЕННАЯ ResNet МОДЕЛЬ для предсказания точки
print("\n📦 Загрузка обученной ResNet модели...")

def create_resnet_model():
    model = models.resnet18(weights=None)  # без предобученных весов
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2),
        nn.Sigmoid()
    )
    return model

# Загружаем нашу обученную модель
try:
    gaze_model = torch.load('gaze_resnet_full.pth', map_location=device)
    gaze_model.eval()
    print("✅ ResNet модель загружена (полная)")
except:
    # Если не получилось загрузить полную модель, пробуем загрузить веса
    gaze_model = create_resnet_model()
    gaze_model.load_state_dict(torch.load('gaze_resnet_weights.pth', map_location=device))
    gaze_model.eval()
    print("✅ ResNet модель загружена (веса)")

# =================== 2. ФУНКЦИИ ДЛЯ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ ===================

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def filter_duplicate_boxes(boxes, iou_threshold=0.5):
    if boxes is None or len(boxes) == 0:
        return []
    boxes_np = []
    for box in boxes:
        coords = box.xyxy[0].cpu().numpy()
        boxes_np.append((coords, box.conf[0].cpu().numpy(), box.cls[0].cpu().numpy(), box))
    boxes_np.sort(key=lambda x: x[1], reverse=True)
    filtered_boxes = []
    used_indices = set()
    for i, (coords_i, conf_i, cls_i, box_i) in enumerate(boxes_np):
        if i in used_indices:
            continue
        filtered_boxes.append(box_i)
        used_indices.add(i)
        for j, (coords_j, conf_j, cls_j, box_j) in enumerate(boxes_np):
            if j in used_indices or i == j:
                continue
            iou = calculate_iou(coords_i, coords_j)
            if iou > iou_threshold:
                used_indices.add(j)
    return filtered_boxes

def preprocess_eye_for_pupil(eye_roi):
    try:
        if len(eye_roi.shape) == 3:
            gray = cv2.cvtColor(eye_roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = eye_roi
        resized = cv2.resize(gray, (32, 16))
        normalized = resized / 255.0
        tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(device), resized
    except Exception as e:
        return None, None

def detect_pupil_simple(eye_roi):
    try:
        if len(eye_roi.shape) == 3:
            gray = cv2.cvtColor(eye_roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = eye_roi
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 10:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return cx, cy
        return eye_roi.shape[1] // 2, eye_roi.shape[0] // 2
    except Exception as e:
        return eye_roi.shape[1] // 2, eye_roi.shape[0] // 2

def detect_pupil_neural(left_eye_roi, right_eye_roi):
    try:
        left_tensor, _ = preprocess_eye_for_pupil(left_eye_roi)
        right_tensor, _ = preprocess_eye_for_pupil(right_eye_roi)
        if left_tensor is None or right_tensor is None:
            return detect_pupil_simple(left_eye_roi), detect_pupil_simple(right_eye_roi)
        with torch.no_grad():
            pupils_pred = pupil_model(left_tensor, right_tensor)
        pupil_y, pupil_x = pupils_pred[0].cpu().numpy()
        left_pupil_x = pupil_x * left_eye_roi.shape[1]
        left_pupil_y = pupil_y * left_eye_roi.shape[0]
        right_pupil_x = pupil_x * right_eye_roi.shape[1]
        right_pupil_y = pupil_y * right_eye_roi.shape[0]
        return (left_pupil_x, left_pupil_y), (right_pupil_x, right_pupil_y)
    except Exception as e:
        return detect_pupil_simple(left_eye_roi), detect_pupil_simple(right_eye_roi)

def extract_features_from_frame(frame):
    """Извлекает 12 признаков из кадра"""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    
    # Детекция глаз через YOLO
    results = eye_model.predict(frame, conf=0.01, iou=0.4, verbose=False)
    
    if results[0].boxes is None:
        return None
    
    filtered_boxes = filter_duplicate_boxes(results[0].boxes)
    
    if len(filtered_boxes) < 2:
        return None
    
    # Сортируем глаза по X координате
    objects_info = []
    for box in filtered_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        objects_info.append({
            'coords': [x1, y1, x2, y2],
            'center_x': (x1 + x2) / 2
        })
    
    objects_info.sort(key=lambda x: x['center_x'])
    
    # Берем левый и правый глаз
    left_eye = objects_info[0]
    right_eye = objects_info[-1]
    
    left_coords = [int(c) for c in left_eye['coords']]
    right_coords = [int(c) for c in right_eye['coords']]
    
    # Вырезаем области глаз
    left_eye_roi = frame[left_coords[1]:left_coords[3], left_coords[0]:left_coords[2]]
    right_eye_roi = frame[right_coords[1]:right_coords[3], right_coords[0]:right_coords[2]]
    
    if left_eye_roi.size == 0 or right_eye_roi.size == 0:
        return None
    
    # Детекция зрачков
    left_pupil, right_pupil = detect_pupil_neural(left_eye_roi, right_eye_roi)
    
    # Абсолютные координаты зрачков
    left_pupil_abs = (left_coords[0] + left_pupil[0], left_coords[1] + left_pupil[1])
    right_pupil_abs = (right_coords[0] + right_pupil[0], right_coords[1] + right_pupil[1])
    
    # Извлекаем 12 признаков
    features = []
    
    # 1. Центры глаз
    left_center_x = (left_coords[0] + left_coords[2]) / 2
    left_center_y = (left_coords[1] + left_coords[3]) / 2
    features.append(left_center_x / w)
    features.append(left_center_y / h)
    
    right_center_x = (right_coords[0] + right_coords[2]) / 2
    right_center_y = (right_coords[1] + right_coords[3]) / 2
    features.append(right_center_x / w)
    features.append(right_center_y / h)
    
    # 2. Относительные положения зрачков
    left_width = left_coords[2] - left_coords[0]
    left_height = left_coords[3] - left_coords[1]
    features.append((left_pupil_abs[0] - left_coords[0]) / left_width)
    features.append((left_pupil_abs[1] - left_coords[1]) / left_height)
    
    right_width = right_coords[2] - right_coords[0]
    right_height = right_coords[3] - right_coords[1]
    features.append((right_pupil_abs[0] - right_coords[0]) / right_width)
    features.append((right_pupil_abs[1] - right_coords[1]) / right_height)
    
    # 3. Размеры глаз
    features.append(left_width / w)
    features.append(left_height / h)
    features.append(right_width / w)
    features.append(right_height / h)
    
    return np.array(features, dtype=np.float32)

# =================== 3. ФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЯ ЧЕРЕЗ RESNET ===================

def prepare_for_resnet(features):
    """Преобразует 12 признаков в формат для ResNet (3, 224, 224)"""
    img_size = 224
    # Расширяем признаки до нужного размера
    features_expanded = np.repeat(features, img_size * img_size * 3 // 12 + 1)
    features_expanded = features_expanded[:3 * img_size * img_size]
    # Нормализуем
    features_expanded = (features_expanded - features_expanded.mean()) / (features_expanded.std() + 1e-8)
    return features_expanded.reshape(1, 3, img_size, img_size)

def predict_gaze_point(features, screen_width=1920, screen_height=1080):
    """Предсказывает точку взгляда на экране"""
    if features is None:
        return None
    
    # Подготавливаем данные для ResNet
    input_tensor = torch.FloatTensor(prepare_for_resnet(features))
    
    # Предсказываем
    with torch.no_grad():
        normalized_point = gaze_model(input_tensor)[0].numpy()
    
    # Конвертируем в пиксели
    screen_x = int(normalized_point[0] * screen_width)
    screen_y = int(normalized_point[1] * screen_height)
    
    return (screen_x, screen_y)

# =================== 4. ОСНОВНОЙ ЦИКЛ С ВЕБ-КАМЕРОЙ ===================

def main():
    print("\n📷 ЗАПУСК ВЕБ-КАМЕРЫ")
    print("="*60)
    
    # Открываем веб-камеру
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Не удалось открыть веб-камеру")
        return
    
    print("✅ Веб-камера открыта")
    print("👁️ Смотрите в камеру - точка покажет, куда вы смотрите")
    print("🛑 Нажмите 'q' для выхода")
    print("="*60)
    
    # Настройки окна
    cv2.namedWindow('Gaze Tracking with ResNet', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gaze Tracking with ResNet', 1024, 768)
    
    # Для сглаживания предсказаний
    history = []
    history_size = 5
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Копируем кадр для отрисовки
        display = frame.copy()
        
        # Извлекаем признаки
        features = extract_features_from_frame(frame)
        
        if features is not None:
            # Предсказываем точку
            gaze_point = predict_gaze_point(features)
            
            if gaze_point is not None:
                # Добавляем в историю для сглаживания
                history.append(gaze_point)
                if len(history) > history_size:
                    history.pop(0)
                
                # Усредняем историю
                avg_x = int(np.mean([p[0] for p in history]))
                avg_y = int(np.mean([p[1] for p in history]))
                
                # Конвертируем в координаты на кадре (для визуализации)
                display_x = int(avg_x * w / 1920)
                display_y = int(avg_y * h / 1080)
                
                # Рисуем точку взгляда
                cv2.circle(display, (display_x, display_y), 15, (0, 0, 255), -1)
                cv2.circle(display, (display_x, display_y), 20, (255, 255, 255), 2)
                
                # Показываем координаты
                cv2.putText(display, f"Screen: ({avg_x}, {avg_y})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display, "TRACKING ACTIVE", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Рисуем bounding boxы глаз
                results = eye_model.predict(frame, conf=0.01, iou=0.4, verbose=False)
                if results[0].boxes is not None:
                    filtered = filter_duplicate_boxes(results[0].boxes)
                    for box in filtered[:2]:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.putText(display, "NO GAZE PREDICTION", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display, "NO EYES DETECTED", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Информация
        cv2.putText(display, f"Frame: {frame_count}", 
                   (display.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "Press 'q' to quit", 
                   (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Показываем кадр
        cv2.imshow('Gaze Tracking with ResNet', display)
        
        # Выход по 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Программа завершена")

# =================== 5. ЗАПУСК ===================

if __name__ == "__main__":
    main()