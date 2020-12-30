import cv2
import numpy as np
from pyzbar.pyzbar import decode
import mediapipe as mp
from math import floor
import pytesseract

class detectPaper:
    def __init__(self, img):
        self.img = img
        self.width, self.height, _ = img.shape

    def getBarcodeData(self, imgScan):
        resizedImg = cv2.resize(imgScan, (self.width//2, self.height//2))
        if decode(resizedImg):
            myData = decode(resizedImg).data.decode('utf-8')
            return myData
        else:
            return False


    def getPaper(self, img):

        per = 30
        imgQ = cv2.imread('resources/handpaper.jpg')

        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(imgQ, None)
        kp2, des2 = orb.detectAndCompute(img, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des2, des1)
        matches.sort(key=lambda x: x.distance)
        good = matches[:int(len(matches) * (per / 100))]
        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
        imgScan = cv2.warpPerspective(img, M, (self.width, self.height))

        return imgScan, M

    def imageData(self, img):

        imgScan, M = self.getPaper(img)
        barcodeData = self.getBarcodeData(imgScan)

        if barcodeData:
            points, imageMediaPipe = pointer.getIndexFingerPoints(imgScan)
            if points == "handsNotFound":
                print("hands not  founded")
                return "handsNotFounded", imageMediaPipe
            else:
                pointedText, imgWithText = textRecognition.getPointedText(imgScan, points)
                if(pointedText == "pointIsNotInTextRegion"):
                    return pointedText, imgWithText
                else:
                    return  pointedText, imgWithText
        else:
            return "barcodeNotFound"

class pointer:

    def getIndexFinerPoints(self, imgScan):

        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5, )

        image = imgScan
        frame = imgScan
        w, he, c = frame.shape

        image = cv2.cvtColor(cv2.flip(image, -1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h = hand_landmarks.landmark[8]

                a = float(h.x)
                b = float(h.y)
                x_px = min(floor(a * w), w - 1)
                y_px = min(floor(b * he), he - 1)
                points = [x_px, y_px]

                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # cv2.imshow('MediaPipe Hands', image)
        if points:
            return points, image
        else:
            return "handsNotFound", image


class textRecognition:
    def getPointedText(self, imgScan, points):

        roi = [[(566, 726), (846, 896)], [(873, 730), (1150, 910)], [(1160, 733), (1453, 906)],
               [(1463, 746), (1743, 913)], [(1760, 740), (2016, 893)], [(563, 926), (836, 1096)],
               [(866, 946), (1160, 1093)], [(1173, 953), (1446, 1093)], [(1470, 940), (1750, 1090)],
               [(1763, 943), (2030, 1103)], [(573, 1113), (836, 1286)], [(873, 1126), (1143, 1300)],
               [(1176, 1130), (1440, 1290)], [(1460, 1120), (1730, 1290)], [(1750, 1120), (2033, 1293)],
               [(543, 1313), (826, 1496)], [(853, 1330), (1130, 1493)], [(1170, 1333), (1436, 1486)],
               [(1456, 1320), (1730, 1493)], [(1753, 1323), (1993, 1470)], [(563, 1526), (816, 1663)],
               [(856, 1523), (1130, 1680)], [(1160, 1516), (1443, 1673)], [(1473, 1516), (1746, 1690)],
               [(1760, 1533), (2010, 1670)]]

        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

        imgShow = imgScan.copy()
        imgMask = np.zeros_like(imgShow)
        x_px,y_px = points[0], points[1]
        myData = []

        pointsAreInRegion = 0
        b = 30 # roi with extented region space

        for x, r in enumerate(roi):
            if (x_px > r[0][0] & x_px < r[0][1]) & (y_px > r[0][1] & y_px < r[1][1]):
                pointsAreInRegion = 1
                cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
                imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
                imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
                myData.append(pytesseract.image_to_string(imgCrop))
                cv2.putText(imgShow, str(myData[x]), (r[0][0], r[0][1]),
                            cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 4)
        if pointsAreInRegion:
            return myData, imgShow
        else:
            return "pointIsNotInTextRegion", imgShow
