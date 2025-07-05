import cv2
import time
import mediapipe as mp


cap = cv2.VideoCapture("Video3.mp4")

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

pTime=0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
            
            
        for id, lm in enumerate(faceLms.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            print([id,cx,cy])
    
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    ptime = cTime
    cv2.putText(img, "FPS: "+str(int(fps)), (10,65), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    
    
    cv2.imshow("Video", img)
    cv2.waitKey(50)
    