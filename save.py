import cv2
import os
import time

def create_dataset(words, save_path, num_samples=3):
    cap = cv2.VideoCapture(0)

    for word in words:
        word_path = os.path.join(save_path, word)

        if not os.path.exists(word_path):
            os.makedirs(word_path)

        existing_files = len([name for name in os.listdir(word_path) if name.endswith(".jpg")])
        count = existing_files

        mode = input(f"{word} için veri toplama modu seçin (otomatik: 'a', manuel: 'm'): ")
        capturing = False
        capturing_time = time.time()  # Otomatik mod için zamanlayıcıyı başlatıyoruz

        print(f"{word} için {num_samples} görüntü kaydedilecek. Başlamak için 's' tuşuna basın, çıkmak için 'q'.")

        while count < existing_files + num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(frame, f"{word}: {count}/{existing_files + num_samples}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Dataset Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                capturing = True
                # Otomatik modda ilk görüntü kaydedildikten sonra zaman başlatılır
                if mode == 'a' and capturing_time == time.time():
                    capturing_time = time.time()  # Zaman başlatılır

            elif key == ord('q'):
                break

            if capturing:
                if mode == 'a':
                    # Otomatik modda her 3 saniyede bir görüntü kaydet
                    if time.time() - capturing_time >= 3:
                        img_name = os.path.join(word_path, f"{word}_{count:04d}.jpg")
                        cv2.imwrite(img_name, frame)
                        count += 1
                        capturing_time = time.time()  # Son kaydedilen zaman güncelleniyor
                elif mode == 'm' and key == ord('c'):
                    # Manuel modda 'c' tuşuna basarak fotoğraf çekilebilir
                    img_name = os.path.join(word_path, f"{word}_{count:04d}.jpg")
                    cv2.imwrite(img_name, frame)
                    count += 1

        print(f"{word} için veri toplama tamamlandı!")

    cap.release()
    cv2.destroyAllWindows()
    print("Veri seti tamamlandı!")

if __name__ == "__main__":
    save_path = "/Users/altun/Desktop/tübitak/VeriSetim"  # Türkçe karakter uyumlu yazıldı
    words = ["Evet", "Hayır", "Teşekkürler", "Merhaba", "Tamam", "Güle güle", "Sevgi", "Affet", "Lütfen", "İyi",
             "Kötü", "Mutlu", "Üzgün", "Yorgun", "Hasta", "Soğuk", "Sıcak", "Su", "Yemek", "Bekle",
             "Dur", "Hızlı", "Yavaş", "Büyük", "Küçük", "Ben", "Sen", "Biz", "Onlar", "Güzel"]
    create_dataset(words, save_path)