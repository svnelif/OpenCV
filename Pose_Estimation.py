import cv2
import mediapipe as mp
import time 

mpPorse = mp.solutions.pose
pose = mpPorse.Pose()

mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture("Video3.mp4")

pTime = 0

while True:
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)


    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPorse.POSE_CONNECTIONS)
    
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, _ = img.shape
        cx, cy = int(lm.x*w), int(lm.y*h)
        
        if id == 13:
            cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
            
        
      
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
      
    cv2.putText(img, "FPS: "+ str(int(fps)), (10,65), cv2.FONT_HERSHEY_PLAIN, 2, (100,5,8),4)
    
    cv2.imshow("Video", img)
    cv2.waitKey(40)
    