import mediapipe as mp
import cv2
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

arr = np.full([480,640,3],255,dtype=np.uint8)
with mp_holistic.Holistic() as holistic:
    vid = cv2.VideoCapture(0)
    while vid.isOpened():
        _,frame = vid.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = holistic.process(image)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.face_landmarks,mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color = (0,80,250),thickness =1,circle_radius = 1),
                                  mp_drawing.DrawingSpec(color = (0,80,250),thickness =1,circle_radius = 1))
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (0,0,0),thickness =2,circle_radius = 3),
                                  mp_drawing.DrawingSpec(color = (0,0,0),thickness =2,circle_radius = 3))
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (0,0,0),thickness =2,circle_radius = 3),
                                  mp_drawing.DrawingSpec(color = (0,0,0),thickness =2,circle_radius = 3))
        mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (255,80,0),thickness =2,circle_radius = 3),
                                  mp_drawing.DrawingSpec(color = (240,80,0),thickness =2,circle_radius = 3))
        cv2.imshow("Model Detection",image)
        arr.fill(255)
        if (cv2.waitKey(10) & 0xFF == ord('q')):
            break
        
vid.release()
cv2.destroyAllWindows()