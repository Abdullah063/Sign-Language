import os
import cv2
import time
import uuid

# Klasör ve etiket tanımlamaları
IMAGE_PATH = 'CollectedImages'
labels = ['Hello', 'Yes', 'No', 'Thanks', 'IloveYou', 'Please']
number_of_images = 2

# Kullanıcıya seçim yaptır
mode = input("Otomatik mod için 'a', manuel mod için 'm' tuşuna basın: ")

timer_delay = 2  # Otomatik modda her görüntü arasında gecikme süresi

for label in labels:
    img_path = os.path.join(IMAGE_PATH, label)
    os.makedirs(img_path, exist_ok=True)  # Eğer klasör yoksa oluştur
    cap = cv2.VideoCapture(0)
    print(f'Collecting images for {label}')
    time.sleep(3)

    img_count = 0
    while img_count < number_of_images:
        ret, frame = cap.read()
        if not ret:
            print("Kamera açılırken hata oluştu!")
            break

        cv2.imshow('frame', frame)

        if mode == 'a':  # Otomatik mod
            time.sleep(timer_delay)
            save_image = True
        elif mode == 'm':  # Manuel mod (Space tuşuna basınca kaydeder)
            key = cv2.waitKey(1) & 0xFF
            save_image = key == ord(' ')
        else:
            print("Geçersiz mod! Lütfen programı yeniden başlatın.")
            break

        if save_image:
            imagename = os.path.join(img_path, f'{label}.{uuid.uuid1()}.jpg')
            cv2.imwrite(imagename, frame)
            print(f"Kaydedildi: {imagename}")
            img_count += 1

        # Çıkış kontrolü
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
