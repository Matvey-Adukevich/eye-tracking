import cv2
import os
import time

video_path = "pictures/eye_tracker_video.mp4"
output_folder = "output/output_images"
frame_interval = 1

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Ошибка открытия видеофайла: {video_path}")
    exit()

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        timestamp = int(time.time() * 1000)
        output_path = os.path.join(output_folder, f'{timestamp}_frame_{saved_count:05d}.jpg')
        cv2.imwrite(output_path, frame)
        print(f"Сохранено: {output_path}")
        saved_count += 1

    frame_count += 1

cap.release()
print("Разделение видео на фотографии завершено.")