import cv2
import numpy as np
from ultralytics import YOLO
import time

model = YOLO('yolov8n.pt')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

STATES = {
    'normal': {'color': (0, 255, 0), 'label': 'Awake'},
    'sleeping': {'color': (0, 0, 255), 'label': 'Sleeping'},
    'drinking': {'color': (255, 0, 0), 'label': 'Drinking'},
    'phone': {'color': (255, 255, 0), 'label': 'Phone'},
    'bottle': {'color': (0, 165, 255), 'label': 'Bottle'},
    'cup': {'color': (255, 165, 0), 'label': 'Cup'},
    'charger': {'color': (128, 0, 128), 'label': 'Charger'},
    'ball': {'color': (0, 128, 255), 'label': 'Ball'}
}

CUSTOM_OBJECTS = {
    'cup': ['cup', 'mug', 'glass', 'wine glass'],
    'bottle': ['bottle', 'water bottle', 'drink bottle'],
    'charger': ['charger', 'mobile charger', 'phone charger', 'usb charger'],
    'ball': ['ball', 'football', 'basketball', 'tennis ball']
}

cap = cv2.VideoCapture(0)
eye_closed_start = None
EYE_CLOSED_SECONDS_THRESHOLD = 1.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # ✨ NEW LOGIC: Eye state tracking with time
    face_detected = False
    state = 'normal'

    for (x, y, w, h) in faces:
        face_detected = True
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face_roi_gray)

        if len(eyes) >= 2:
            eye_closed_start = None
            state = 'normal'
            for (ex, ey, ew, eh) in eyes[:2]:
                cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
        else:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            elif time.time() - eye_closed_start >= EYE_CLOSED_SECONDS_THRESHOLD:
                state = 'sleeping'

        cv2.rectangle(frame, (x, y), (x+w, y+h), STATES[state]['color'], 2)
        cv2.putText(frame, STATES[state]['label'], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, STATES[state]['color'], 2)

    # 🔁 Reset timer if no face detected
    if not face_detected:
        eye_closed_start = None

    # 🔍 Object detection with YOLOv8
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls)]
                conf = float(box.conf)

                detected = False
                for obj_type, keywords in CUSTOM_OBJECTS.items():
                    if any(keyword in label.lower() for keyword in keywords):
                        if obj_type in STATES:
                            display_label = STATES[obj_type]['label']
                            color = STATES[obj_type]['color']
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{display_label} {conf:.2f}",
                                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, color, 2)
                            detected = True
                            break

                if not detected and label.lower() in ['cell phone', 'mobile phone']:
                    display_label = STATES['phone']['label']
                    color = STATES['phone']['color']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{display_label} {conf:.2f}",
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, color, 2)
            except Exception:
                continue

    cv2.imshow("Activity Monitor", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
