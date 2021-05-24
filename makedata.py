import cv2
import mediapipe as mp
import numpy as np
import pyautogui

width = pyautogui.size().width
height = pyautogui.size().height


max_num_hands = 1
gesture = {
    0:'click', 1:'default', 2:'rock'
}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
# file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
# angle = file[:,:-1].astype(np.float32)
# label = file[:, -1].astype(np.float32)
# knn = cv2.ml.KNearest_create()
# knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(1)
import os
index = input('제스쳐 번호를 입력해주세요 >>')
for i in range(1,1000):
    if os.path.isfile(index+'_'+str(i)+'.csv'):
        continue
    f = open(index+'_'+str(i)+'.csv','w')
    break
import time
while cap.isOpened():
    time.sleep(0.2)
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:

            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            print(data)
            for i in data[0]:
                f.write(str(i)+',')
            f.write(str(index)+'\n')

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break