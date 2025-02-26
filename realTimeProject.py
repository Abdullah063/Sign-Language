import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# TFLite modelini yükle
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# MediaPipe hands modülünü başlat
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


# El landmark'larını görselleştirme fonksiyonu
def visualize_landmarks(hand_landmarks, width, height):
    # Matplotlib figürü oluştur
    fig, ax = plt.subplots(figsize=(6, 6))

    # El kenarları (original kodunuzdan)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (0, 17), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
             (11, 12),
             (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20)]

    # Landmark noktalarını x,y koordinatlarına dönüştür
    x_coords = []
    y_coords = []

    for landmark in hand_landmarks.landmark:
        x_coords.append(landmark.x)
        y_coords.append(landmark.y)

    # Noktaları çiz
    ax.scatter(x_coords, y_coords, color='dodgerblue')

    # Noktaları numaralandır
    for i in range(len(x_coords)):
        ax.text(x_coords[i], y_coords[i], str(i))

    # Kenarları çiz
    for edge in edges:
        ax.plot([x_coords[edge[0]], x_coords[edge[1]]],
                [y_coords[edge[0]], y_coords[edge[1]]],
                color='salmon')

    ax.set_title("Hand Landmarks")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # y-ekseni ters çevir (görüntü koordinat sistemi)
    ax.set_aspect('equal')
    ax.axis('off')

    # Figure'ı numpy array'e dönüştür
    canvas = FigureCanvas(fig)
    canvas.draw()

    # Doğrudan matplotlib figürünü OpenCV görüntüsüne dönüştür
    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)
    img_array = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    img_array = img_array.reshape(height, width, 4)
    # ARGB'yi BGR'ye dönüştür (OpenCV için)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

    plt.close(fig)
    return img_array


# Modelin işlemi için landmark'ları hazırlama fonksiyonu
def prepare_landmarks_for_model(multi_hand_landmarks):
    # ROWS_PER_FRAME değeri kodunuzda 543 olarak tanımlanmış
    landmarks = np.zeros((543, 3))

    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:
            for i, landmark in enumerate(hand_landmarks.landmark):
                # MediaPipe'dan gelen landmark'ları doğru indekslere yerleştir
                # Bu kısım örnek kodunuza göre uyarlanmalıdır
                if i < 21:  # Bir elde 21 landmark vardır
                    landmarks[i] = [landmark.x, landmark.y, landmark.z]

    # Not: Gerçek uygulamada, bu kısmı modelin beklediği formata göre uyarlamanız gerekebilir
    # Landmark indekslerini LANDMARK_IDX ile filtreleme gibi

    return landmarks


# Webcam'i başlat
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Görüntüyü RGB'ye dönüştür (MediaPipe RGB bekler)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Orijinal görüntüde el landmark'larını çiz
    image_with_hands = image.copy()
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                image_with_hands,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

            # Ayrı bir pencerede landmark noktalarını göster
            h, w, _ = image.shape
            landmark_visualization = visualize_landmarks(hand_landmarks, w, h)

            # Landmark görselleştirmesini göster
            window_name = f'Hand Landmarks {i + 1}'
            cv2.imshow(window_name, landmark_visualization)

        # Hazırlanan landmark'ları model için kullan
        landmarks = prepare_landmarks_for_model(results.multi_hand_landmarks)

        # Modele gönder ve tahmin al (TFLite için)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Model giriş şekli doğru mu kontrol et ve uygun şekilde ayarla
        input_shape = input_details[0]['shape']
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(landmarks, axis=0).astype(np.float32))
        interpreter.invoke()

        # Model çıktısını al
        prediction = interpreter.get_tensor(output_details[0]['index'])
        sign_index = np.argmax(prediction)

        # Sonucu ekranda göster
        confidence = np.max(prediction) * 100
        cv2.putText(
            image_with_hands,
            f"Sign: {sign_index} ({confidence:.1f}%)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # Ana görüntüyü göster
    cv2.imshow('Video Feed with Hands', image_with_hands)

    # ESC tuşu ile çıkış
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()