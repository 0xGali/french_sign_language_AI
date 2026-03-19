import os
import json
import cv2
import mediapipe as mp
import random
import numpy as np


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

    def findPosition(self, img, handNo=0, draw=True, w=1, h=1):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Normalisation
                cx_norm = lm.x
                cy_norm = lm.y
                lmList.append([id, cx_norm, cy_norm])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


def sp_noise(image, prob):
    '''
    Ajoute du bruit à l'image (points blancs et noir, salt and pepper noise)
    prob: Probabilité du bruit
    A utiliser entre 0.01 and 0.5 du bruit total (avec 0.05 étant bien)
    '''
    imageModif = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                imageModif[i][j] = 0
            elif rdn > thres:
                imageModif[i][j] = 255
            else:
                imageModif[i][j] = image[i][j]
    return imageModif


def lectureVideo():
    mapMot = {}

    # On parcourt le dossier "videos" pour récupérer les vidéos à traiter
    for video in os.listdir(f"videos"):
        listeMain = []
        print(video)
        cap = cv2.VideoCapture(f"videos/{video}")
        detector = handDetector()

        # récupération de la taille de la video
        success, first_frame = cap.read()
        if success:
            h, w, _ = first_frame.shape
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Revenir au début de la vidéo
        else:
            print(f"Impossible de lire la vidéo {video}")

        # La boucle continue tant que la vidéo est lancée
        while True and cap.isOpened():

            # On récupère une image de la vidéo
            success, img = cap.read()
            if img is None:
                break

            # Redimentionnement de l'image pour accélérer le processus de détection
            if img.shape[0] > 300:
                img = cv2.resize(img, (int(img.shape[1] // 1.5), int(img.shape[0] // 1.5)))

            # Détection des mains sur l'image et l'affiche sur l'image
            img = detector.findHands(img)

            # En fonction du nombre de mains détectées, on récupère les coord et on les stocke dans 1 ou 2 listes
            if detector.results.multi_hand_landmarks == None:
                continue
            elif len(detector.results.multi_hand_landmarks) == 1:
                lmList = detector.findPosition(img, 0)
                lmList2 = []
                for i in range(len(lmList)):
                    lmList2.append([i, 0, 0])
                print(lmList2)
                listeMain.append((lmList, lmList2))
            elif len(detector.results.multi_hand_landmarks) == 2:
                lmList = detector.findPosition(img, 0)
                lmList2 = detector.findPosition(img, 1)
                listeMain.append((lmList, lmList2))

            # Affiche les images
            # cv2.imshow("Image",img)

            # Permet de finir la lecture de la vidéo en appuyant sur "q"
            if cv2.waitKey(1) == ord('q'):
                break

        video = video.split(".")[0]
        # Crée un map en associant le mot à la liste de coordonnées des mains
        mapMot.update({video: listeMain})
    return mapMot


def lectureVideoBruit():
    mapMot = {}

    for video in os.listdir(f"videos"):
        listeMain = []
        print(video)
        cap = cv2.VideoCapture(f"videos/{video}")
        detector = handDetector()

        # Récupérer la taille de la vidéo
        success, first_frame = cap.read()
        if success:
            h, w, _ = first_frame.shape
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Revenir au début
        else:
            print(f"Impossible de lire la vidéo {video}")
            continue

        while True and cap.isOpened():
            success, img = cap.read()
            if img is None:
                break

            if img.shape[0] > 300:
                img = cv2.resize(img, (int(img.shape[1] // 1.5), int(img.shape[0] // 1.5)))

            # On applique du bruit à l'image avant de faire la détection des mains
            img = sp_noise(img, 0.01)
            img = detector.findHands(img)

            if detector.results.multi_hand_landmarks == None:
                continue
            elif len(detector.results.multi_hand_landmarks) == 1:
                lmList = detector.findPosition(img, 0)
                lmList2 = []
                for i in range(len(lmList)):
                    lmList2.append([i,0,0])
                print(lmList2)
                listeMain.append((lmList,lmList2))
            elif len(detector.results.multi_hand_landmarks) == 2:
                lmList = detector.findPosition(img, 0)
                lmList2 = detector.findPosition(img, 1)
                listeMain.append((lmList, lmList2))

            # cv2.imshow("Image",img)
            if cv2.waitKey(1) == ord('q'):
                break
        video = video.split(".")[0]
        mapMot.update({video: listeMain})
    return mapMot


def main():
    mapMot = lectureVideo()
    mapMotBruit = lectureVideoBruit()
    mapFusion = {**mapMot, **mapMotBruit}

    # On crée le json à partir de la fusion des 2 maps (avec et sans bruit)
    with open("coordonnees_mots.json", "w") as f:
        json.dump(mapFusion, f)


if __name__ == "__main__":
    main()
