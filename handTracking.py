import os
import json
import cv2
import mediapipe as mp

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=1,trackCon=1):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self,img, handNo=0,draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c=img.shape
                cx, cy=int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        return lmList

def lectureVideo():
    mapMot = {}
    compteur = 0
    for video in os.listdir(f"videos"):
        listeMain = []
        print(video)
        if compteur == 20:
            break
        cap = cv2.VideoCapture(f"videos/{video}")
        detector = handDetector()
        while True and cap.isOpened():
            success, img= cap.read()
            if img is None:
                break
            img = detector.findHands(img)
            if detector.results.multi_hand_landmarks == None:
                continue
            elif len(detector.results.multi_hand_landmarks) == 1:
                lmList = detector.findPosition(img,0)
                print(lmList)
                listeMain.append(lmList)
            elif len(detector.results.multi_hand_landmarks) == 2:
                lmList = detector.findPosition(img,0)
                lmList2 = detector.findPosition(img,1)
                print(lmList)
                print(lmList2)
                listeMain.append((lmList,lmList2))
            
            cv2.imshow("Image",img)
            if cv2.waitKey(1) == ord('q'):
                break
        video = video.split(".")[0]
        mapMot.update({video:listeMain})
        compteur += 1
    return mapMot

def main():
    mapMot = lectureVideo()
    print(mapMot)
    with open("coordonnees_mots.json", "w") as f:
        json.dump(mapMot, f)

if __name__ == "__main__":
    main()