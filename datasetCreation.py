import copy
import csv
import itertools
import cv2
import mediapipe as mp
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)

def cal_landmarks(image, landmarks):
    img_width, img_height = image.shape[1], image.shape[0]

    landmark_points = []

    for hand_landmarks in landmarks:
        for landmark in hand_landmarks.landmark:
            landmark_x = min(int(landmark.x * img_width), img_width-1)
            landmark_y = min(int(landmark.y * img_height), img_height-1)
            landmark_points.append(landmark_x)
            landmark_points.append(landmark_y)

    return landmark_points

def normalizeData(landmark_list):
    list_copy = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0

    for index, landmark_point in enumerate(list_copy):
        if index % 2 == 0:
            if index == 0:
                base_x = landmark_point
            list_copy[index] = landmark_point - base_x
        else:
            if index == 1:
                base_y = landmark_point
            list_copy[index] = landmark_point - base_y

    list_copy = list(map(lambda x: x / max(list(map(abs, list_copy))), list_copy))
    return list_copy

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    class_name = "eight"

    if results.multi_hand_landmarks:
        landmark_points = cal_landmarks(img, results.multi_hand_landmarks)
        normalized_data = normalizeData(landmark_points)

        with open(r'finalDataset.csv', mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([class_name] + normalized_data)

        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,
                                  hand_landmarks,
                                  mpHands.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                  connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

    cv2.imshow("Image", img)
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
