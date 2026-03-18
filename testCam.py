import os
import cv2
import mediapipe as mp
import handTracking as ht  # Assure-toi que ce module existe et contient sp_noise

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape
            for id, lm in enumerate(myHand.landmark):
                # Coordonnées normalisées (déjà entre 0 et 1)
                cx_norm = lm.x
                cy_norm = lm.y
                # Coordonnées en pixels (pour affichage)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx_norm, cy_norm])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Largeur
    cap.set(4, 480)  # Hauteur
    detector = handDetector()
    while True and cap.isOpened():
        success, img = cap.read()
        success2, img2 = cap.read()
        if img is None or img2 is None:
            break

        img2 = ht.sp_noise(img2, 0.05)
        img2 = detector.findHands(img2)
        img = detector.findHands(img)

        if detector.results.multi_hand_landmarks is None:
            continue
        elif len(detector.results.multi_hand_landmarks) == 1:
            lmList = detector.findPosition(img, 0)
            print("Main 1 (normalisé) :", lmList)
        elif len(detector.results.multi_hand_landmarks) == 2:
            lmList = detector.findPosition(img, 0)
            lmList2 = detector.findPosition(img, 1)
            print("Main 1 (normalisé) :", lmList)
            print("Main 2 (normalisé) :", lmList2)

        cv2.imshow("Image", img)
        cv2.imshow("Image2", img2)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()
