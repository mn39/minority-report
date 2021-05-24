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
file = np.genfromtxt('data/new_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(1)

import time
endtime = time.time()
def click():
    global endtime
    if(time.time() <= endtime):
        return
    print("click")
    x = pyautogui.position().x
    y = pyautogui.position().y

    pyautogui.leftClick(x, y)
    endtime = time.time()+0.5

isDragging = True
def dragdown():
    global isDragging
    if(isDragging):
        return
    isDragging = True
    print("dragdown")
    x = pyautogui.position().x
    y = pyautogui.position().y
    pyautogui.mouseDown(x, y, button='left')

def dragup():
    global isDragging
    isDragging = False
    # print("dragup")
    x = pyautogui.position().x
    y = pyautogui.position().y
    pyautogui.mouseUp(x, y, button='left')

def scrolldown():
    x = pyautogui.position().x
    y = pyautogui.position().y
    pyautogui.scroll(500, x=x, y=y)
def scrollup():
    x = pyautogui.position().x
    y = pyautogui.position().y
    pyautogui.scroll(-500, x=x, y=y)

lastGesture = [None]*10
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            mainpoint = res.landmark[0]
            x = min(1,max(0,mainpoint.x-0.2)/0.6)
            y = min(1,max(0,mainpoint.y-0.4)/0.4)
            pyautogui.moveTo(x*width,y*height)
            


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
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # Draw gesture result
            # if idx in rps_gesture.keys():
            #     cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            # Other gestures
            for i in range(8,-1,-1):
                lastGesture[i+1] = lastGesture[i]
                
            lastGesture[0] = gesture[idx]
            
            rockcnt = 0
            for i in range(5):
                if(lastGesture[i] == 'rock'): rockcnt+=1
            if(rockcnt>=3):
                dragdown()
            else:
                dragup()
                if(lastGesture[0] == 'click' and lastGesture[1] == 'click'):
                    click()
                elif(gesture[idx] == 'after-scrolldown' and lastGesture[1] == 'after-scrolldown'):
                    if(lastGesture.count('pre-scrolldown')>=2):
                        scrolldown()
                elif(gesture[idx] == 'after-scrollup' and lastGesture[1] == 'after-scrollup'):
                    if(lastGesture.count('pre-scrollup')>=2):
                        scrollup()
                
            cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break
