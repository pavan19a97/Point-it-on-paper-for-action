import cv2
import numpy as np
from textRecognizer import textRecognition
class detectPaper:

    def getPaper( img):
        per = 30
        imgQ = cv2.imread('resources/handpaper.jpg')
        imgQ = cv2.resize(imgQ,(1920,1080))
        h, w, c = imgQ.shape
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(imgQ, None)
        kp2, des2 = orb.detectAndCompute(img, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des2, des1)
        matches.sort(key=lambda x: x.distance)
        good = matches[:int(len(matches) * (per / 100))]
        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
        try:
            imgScan = cv2.warpPerspective(img, M, (w, h))
        except Exception as e:
            print(e)
            return "e",M

        return imgScan, M

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    cap.set(3, 1920)
    cap.set(4, 1080)

    while cap.isOpened():
        suc, img = cap.read()
        img = cv2.resize(img, (1920 , 1080 ))
        cv2.imshow("imageres", img)
        # img = cv2.rotate(img, cv2.cv2.ROTATE_180)
        # img = cv2.flip(img, -1)
        imgS, h =detectPaper.getPaper(img)

        # cv2.imshow("image", img)
        if cv2.waitKey(10) & 0xFFF == ord("q"):
            break