import time
import cv2
import numpy as np
import pyautogui
import mediapipe as mp

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

# STEP 2: Create an HandLandmarker object.


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.


max_num_hands = 1

cap = cv2.VideoCapture(1)

endtime = time.time()


def click():
    global endtime
    if (time.time() <= endtime):
        return
    print("click")
    x = pyautogui.position().x
    y = pyautogui.position().y

    pyautogui.leftClick(x, y)
    endtime = time.time()+0.5


isDragging = True


def dragdown():
    global isDragging
    if (isDragging):
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

    # result = hands.process(img)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    # STEP 4: Detect hand landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the classification result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(
        image.numpy_view(), detection_result)
    cv2.imshow('Game', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    # width = pyautogui.size().width
    # height = pyautogui.size().height

    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # if result.multi_hand_landmarks is not None:
    #     for res in result.multi_hand_landmarks:
    #         mainpoint = res.landmark[0]
    #         x = min(1,max(0,mainpoint.x-0.2)/0.6)
    #         y = min(1,max(0,mainpoint.y-0.4)/0.4)
    #         pyautogui.moveTo(x*width,y*height)

    #         joint = np.zeros((21, 3))
    #         for j, lm in enumerate(res.landmark):
    #             joint[j] = [lm.x, lm.y, lm.z]

    #         # Compute angles between joints
    #         v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
    #         v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
    #         v = v2 - v1 # [20,3]
    #         # Normalize v
    #         v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    #         # Get angle using arcos of dot product
    #         angle = np.arccos(np.einsum('nt,nt->n',
    #             v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
    #             v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

    #         angle = np.degrees(angle) # Convert radian to degree

    #         # Inference gesture
    #         data = np.array([angle], dtype=np.float32)
    #         ret, results, neighbours, dist = knn.findNearest(data, 3)
    #         idx = int(results[0][0])

    #         # Draw gesture result
    #         # if idx in rps_gesture.keys():
    #         #     cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    #         # Other gestures
    #         for i in range(8,-1,-1):
    #             lastGesture[i+1] = lastGesture[i]

    #         lastGesture[0] = gesture[idx]

    #         rockcnt = 0
    #         for i in range(5):
    #             if(lastGesture[i] == 'rock'): rockcnt+=1
    #         if(rockcnt>=3):
    #             dragdown()
    #         else:
    #             dragup()
    #             if(lastGesture[0] == 'click' and lastGesture[1] == 'click'):
    #                 click()
    #             elif(gesture[idx] == 'after-scrolldown' and lastGesture[1] == 'after-scrolldown'):
    #                 if(lastGesture.count('pre-scrolldown')>=2):
    #                     scrolldown()
    #             elif(gesture[idx] == 'after-scrollup' and lastGesture[1] == 'after-scrollup'):
    #                 if(lastGesture.count('pre-scrollup')>=2):
    #                     scrollup()

    # cv2.imshow('Game', img)
    # if cv2.waitKey(1) == ord('q'):
    # break
