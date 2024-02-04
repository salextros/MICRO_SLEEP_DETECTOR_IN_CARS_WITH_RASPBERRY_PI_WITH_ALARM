#PROJECT_CREATED_BY_SALEXTROS

import RPi.GPIO as GPIO
import cv2
import mediapipe as mp
import numpy as np
import time
from threading import Thread
import threading
from collections import deque

time.sleep(60)

pin = 7
GPIO.setmode(GPIO.BOARD)
GPIO.setup(pin, GPIO.OUT)

# Variable para rastrear si el hilo está en ejecución
led_thread_running = False

n = 0

# Función para encender y apagar el LED en segundo plano
def toggle_led():
    global led_thread_running
    if not led_thread_running:
        led_thread_running = True
        for i in range(5):
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(pin, GPIO.LOW)
            time.sleep(1)
        led_thread_running = False

# Crear un hilo para el bucle de encendido y apagado del LED
led_thread = Thread(target=toggle_led)

def drawing_output(frame, coordinates_left_eye, coordinates_right_eye, blink_counter):
      aux_image = np.zeros(frame.shape, np.uint8)
    contours1 = np.array([coordinates_left_eye])
    contours2 = np.array([coordinates_right_eye])
    cv2.fillPoly(aux_image, pts=[contours1], color=(255, 0, 0))
    cv2.fillPoly(aux_image, pts=[contours2], color=(255, 0, 0))
    output = cv2.addWeighted(frame, 1, aux_image, 0.7, 1)

    # Crear una ventana para mostrar el número de parpadeos
    cv2.rectangle(output, (0, 0), (200, 50), (255, 0, 0), -1)
    cv2.rectangle(output, (202, 0), (265, 50), (255, 0, 0),2)
    cv2.putText(output, "Num. Parpadeos:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(output, "{}".format(blink_counter), (220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 0, 250), 2)

    return output


def eye_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))

    return (d_A + d_B) / (2 * d_C)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

mp_face_mesh = mp.solutions.face_mesh
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]
EAR_THRESH = 0.26
NUM_FRAMES = 2
aux_counter = 0
blink_counter = 0
closed_eyes_duration = 0
asleep_threshold = 5

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        coordinates_left_eye = []
        coordinates_right_eye = []

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                for index in index_left_eye:
                    # ... Código para obtener las coordenadas de los ojos izquierdos ...
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_left_eye.append([x, y])
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                    cv2.circle(frame, (x, y), 1, (128, 0, 250), 1)

                for index in index_right_eye:
                    # ... Código para obtener las coordenadas de los ojos derechos ...
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_right_eye.append([x, y])
                    cv2.circle(frame, (x, y), 2, (128, 0, 250), 1)
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

                ear_left_eye = eye_aspect_ratio(coordinates_left_eye)
                ear_right_eye = eye_aspect_ratio(coordinates_right_eye)
                ear = (ear_left_eye + ear_right_eye) / 2

                # Ojos cerrados
                if ear < EAR_THRESH:
                    aux_counter += 1
                    closed_eyes_duration += 1

                    # Verificar si los ojos están cerrados durante el umbral de tiempo
                    if closed_eyes_duration >= asleep_threshold:
                        n += 1
                        print("Se quedo dormido!" + str(n))
                        print()
                        # Iniciar el hilo para controlar el LED
                        print("ya_llegue")
                        led_thread = Thread(target=toggle_led)
                        led_thread.start()
                 
                else:
                    if aux_counter >= NUM_FRAMES:
                        aux_counter = 0
                        blink_counter += 1
                    closed_eyes_duration = 0

            frame = drawing_output(frame, coordinates_left_eye, coordinates_right_eye, blink_counter)

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
