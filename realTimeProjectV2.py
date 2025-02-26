import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# TFLite modelini yükle
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# MediaPipe modüllerini başlat
mp_holistic = mp.solutions.holistic  # El ve yüz takibi için Holistic kullan
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


# Landmark görselleştirme fonksiyonu
def visualize_landmarks(landmarks, landmark_type, width, height):
    # Matplotlib figürü oluştur
    fig, ax = plt.subplots(figsize=(6, 6))

    # El kenarları
    hand_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (0, 17), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                  (11, 12),
                  (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20)]

    # Landmark noktalarını x,y koordinatlarına dönüştür
    x_coords = []
    y_coords = []
    z_coords = []

    for landmark in landmarks.landmark:
        x_coords.append(landmark.x)
        y_coords.append(landmark.y)
        z_coords.append(landmark.z if hasattr(landmark, 'z') else 0)

    # Noktaları çiz
    ax.scatter(x_coords, y_coords, color='dodgerblue')

    # Noktaları numaralandır
    for i in range(len(x_coords)):
        ax.text(x_coords[i], y_coords[i], str(i))

    # Kenarları çiz - sadece el için
    if landmark_type == "hand":
        for edge in hand_edges:
            if edge[0] < len(x_coords) and edge[1] < len(x_coords):
                ax.plot([x_coords[edge[0]], x_coords[edge[1]]],
                        [y_coords[edge[0]], y_coords[edge[1]]],
                        color='salmon')

    ax.set_title(f"{landmark_type.capitalize()} Landmarks")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # y-ekseni ters çevir
    ax.set_aspect('equal')
    ax.axis('off')

    # Figure'ı numpy array'e dönüştür
    canvas = FigureCanvas(fig)
    canvas.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)
    img_array = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    img_array = img_array.reshape(height, width, 4)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

    plt.close(fig)
    return img_array


# Modelin işlemi için landmark'ları hazırlama fonksiyonu
def prepare_landmarks_for_model(results):
    # ROWS_PER_FRAME değeri kodunuzda 543 olarak tanımlanmış
    landmarks = np.zeros((543, 3))

    # Yüz landmark'larını ekle
    if results.face_landmarks:
        for i, landmark in enumerate(results.face_landmarks.landmark):
            landmarks[i] = [landmark.x, landmark.y, landmark.z]

    # Sol el landmark'larını ekle
    if results.left_hand_landmarks:
        for i, landmark in enumerate(results.left_hand_landmarks.landmark):
            # Orijinal kodunuzdaki indekslemeye göre uyarlanmalı
            landmarks[468 + i] = [landmark.x, landmark.y, landmark.z]

    # Sağ el landmark'larını ekle
    if results.right_hand_landmarks:
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            # Orijinal kodunuzdaki indekslemeye göre uyarlanmalı
            landmarks[489 + i] = [landmark.x, landmark.y, landmark.z]

    # Kodunuzdaki LANDMARK_IDX'e göre filtreleme
    LANDMARK_IDX = [0, 9, 11, 13, 14, 17, 117, 118, 119, 199, 346, 347, 348] + list(range(468, 543))
    filtered_landmarks = np.zeros((len(LANDMARK_IDX), 3))

    for i, idx in enumerate(LANDMARK_IDX):
        if idx < landmarks.shape[0]:
            filtered_landmarks[i] = landmarks[idx]

    return filtered_landmarks


# Webcam'i başlat
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Görüntüyü RGB'ye dönüştür
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # Orijinal görüntüde landmark'ları çiz
    image_with_landmarks = image.copy()

    # Yüz landmark'larını çiz
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image_with_landmarks,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )

        # Yüz landmark'larını ayrı pencerede göster
        face_visualization = visualize_landmarks(results.face_landmarks, "face", image.shape[1], image.shape[0])
        cv2.imshow('Face Landmarks', face_visualization)

    # Sol el landmark'larını çiz
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image_with_landmarks,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )

        # Sol el landmark'larını ayrı pencerede göster
        left_hand_visualization = visualize_landmarks(results.left_hand_landmarks, "hand", image.shape[1],
                                                      image.shape[0])
        cv2.imshow('Left Hand Landmarks', left_hand_visualization)

    # Sağ el landmark'larını çiz
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image_with_landmarks,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # Sağ el landmark'larını ayrı pencerede göster
        right_hand_visualization = visualize_landmarks(results.right_hand_landmarks, "hand", image.shape[1],
                                                       image.shape[0])
        cv2.imshow('Right Hand Landmarks', right_hand_visualization)

    # Landmark'ları model için hazırla
    landmarks = prepare_landmarks_for_model(results)

    # Modele gönder ve tahmin al
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # TFLite model için giriş verilerini hazırla
    model_input = np.expand_dims(landmarks, axis=0).astype(np.float32)

    # TF-Lite kodunuzdan:
    # ragged_batch = tf.gather(ragged_batch, LANDMARK_IDX, axis=2)
    # x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    # x = tf.concat([x[...,i] for i in range(3)], -1)

    # NaN değerleri 0 ile değiştir
    model_input = np.nan_to_num(model_input)

    # xyz koordinatlarını düzleştir - orijinal kodunuzdan:
    # x = tf.concat([x[...,i] for i in range(3)],-1)
    flattened_input = np.zeros((1, len(LANDMARK_IDX) * 3), dtype=np.float32)
    for i in range(3):
        flattened_input[0, i * len(LANDMARK_IDX):(i + 1) * len(LANDMARK_IDX)] = model_input[0, :, i]

    # Modele gönder
    interpreter.set_tensor(input_details[0]['index'], flattened_input)
    interpreter.invoke()

    # Model çıktısını al
    prediction = interpreter.get_tensor(output_details[0]['index'])
    sign_index = np.argmax(prediction)

    # Sonucu ekranda göster
    confidence = np.max(prediction) * 100
    cv2.putText(
        image_with_landmarks,
        f"Sign: {sign_index} ({confidence:.1f}%)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Ana görüntüyü göster
    cv2.imshow('Holistic Landmark Detection', image_with_landmarks)

    # ESC tuşu ile çıkış
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()