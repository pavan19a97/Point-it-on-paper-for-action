import cv2
import pytesseract
import numpy as np
class textRecognition:
    def getPointedText(imgScan, points):

        roi = [[(358, 362), (530, 440)], [(550, 366), (722, 440)], [(744, 366), (916, 442)], [(928, 366), (1102, 442)], [(1116, 372), (1282, 442)], [(356, 458), (528, 534)], [(546, 454), (724, 534)], [(742, 456), (898, 538)], [(1120, 460), (1278, 538)], [(364, 554), (520, 630)], [(554, 554), (710, 630)], [(748, 552), (908, 632)], [(930, 558), (1102, 628)], [(1124, 564), (1286, 626)], [(364, 664), (514, 722)], [(550, 650), (712, 728)], [(742, 648), (904, 732)], [(1114, 648), (1278, 732)], [(366, 752), (510, 818)], [(554, 744), (708, 816)], [(1130, 756), (1270, 812)]]

        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

        imgShow = imgScan.copy()
        imgMask = np.zeros_like(imgShow)
        x_px ,y_px = points[0], points[1]
        myData = []

        pointsAreInRegion = 0
        b = 30 # roi with extented region space

        for x, r in enumerate(roi):
            if (x_px > r[0][0] & x_px < r[0][1]) & (y_px > r[1][0] & y_px < r[1][1]):
                pointsAreInRegion = 1
                cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
                imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
                imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
                myData.append(pytesseract.image_to_string(imgCrop))
                # cv2.putText(imgShow, str(myData[x]), (r[0][0], r[0][1]),
                #             cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 4)
        if pointsAreInRegion:
            return myData, imgShow
        else:
            return "pointIsNotInTextRegion", imgShow