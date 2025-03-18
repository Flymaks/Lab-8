import time
import numpy as np
import cv2

def image_processing():
    img = cv2.imread('images/variant-5.jpg')
    cv2.imshow('image', img)
    # Создание шума с помощью случайных чисел и перевод его в формат  uint8
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8) 
    noisy_image = cv2.add(img, noise)     # Добавление шума к изображению
    cv2.imshow('noisy_image',  noisy_image)
  

def video_processing():
    cap = cv2.VideoCapture(0)
    h_e, w_e = 320, 320 # размер экрана
    down_points = (h_e, w_e)
    i = 0
    fly = cv2.imread('fly64.png')
    fly = cv2.resize(fly, (160,160))
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(thresh,
                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if i % 2 == 0:
                a = x + (w // 2)
                b = y + (h // 2)
                print(a, b)
                if a < 50 and b < 50:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                if a > h_e - 50 and a > w_e - 50:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)                           
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)
        i += 1

    cap.release()


if __name__ == '__main__':
    # image_processing()
    video_processing()

cv2.waitKey(0)
cv2.destroyAllWindows()