import mediapipe as mp
import cv2
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# using with to avoid writing the whole function 
with mp_holistic.Holistic() as holistic:
    # taking the video feed from the webcam. the number 0 may vary depending on the system
    vid = cv2.VideoCapture(0)
    
    while vid.isOpened():
        # Taking each fream read by vid and converting it to RGB format
        _,frame = vid.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Processsing the image and converting it to BGR format
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Drawing all the face points and lines on the image 
        mp_drawing.draw_landmarks(image, results.face_landmarks,mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color = (0,80,250),thickness =1,circle_radius = 1),
                                  mp_drawing.DrawingSpec(color = (0,80,250),thickness =1,circle_radius = 1))
        # Drawing all the face points and lines on the image 
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (0,0,0),thickness =2,circle_radius = 3),
                                  mp_drawing.DrawingSpec(color = (0,0,0),thickness =2,circle_radius = 3))
        # Drawing all the face points and lines on the image 
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (0,0,0),thickness =2,circle_radius = 3),
                                  mp_drawing.DrawingSpec(color = (0,0,0),thickness =2,circle_radius = 3))
        # Drawing all the face points and lines on the image 
        mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (255,80,0),thickness =2,circle_radius = 3),
                                  mp_drawing.DrawingSpec(color = (240,80,0),thickness =2,circle_radius = 3))
        # Displaying the final image 
        cv2.imshow("Model Detection",image)
        # code to exit the while loop
        if (cv2.waitKey(10) & 0xFF == ord('q')):
            break
# to release the webcam and destroy all windows displaying the images       
vid.release()
cv2.destroyAllWindows()