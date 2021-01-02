import cv2
from handDetector import pointer
from textRecognizer import textRecognition
from barcodeDetectionAndRecognizer import barcodeDecoder
from paperDetection import detectPaper

def imageData(img):
    imgScan, M = detectPaper.getPaper(img)
    if imgScan=="e":
        return "e", M
    cv2.imshow("scanned Image", cv2.resize(imgScan, (1920//3,1080//3)))
    barcodeData = barcodeDecoder.getBarcodeData(imgScan)

    if barcodeData:
        points, imageMediaPipe = pointer.getIndexFingerPoints(imgScan)
        if points == "handsNotFound":
            print("hands not  founded")
            return "handsNotFounded", imageMediaPipe
        else:
            cv2.imshow("Hands With Media Pipe", cv2.resize(imageMediaPipe,(1920//3,1080//3)))
            pointedText, imgWithText = textRecognition.getPointedText(imgScan, points)
            if (pointedText == "pointIsNotInTextRegion"):
                return pointedText, imgWithText
            else:
                cv2.imshow("With Text", cv2.resize(imgWithText,(1920//3,1080//3)))
                cv2.waitKey(100 )
                return pointedText, imgWithText
    else:
        return "barcodeNotFound", img


def action(img):
    data, imgRceived = imageData(img)
    if data == "e":
        return "e"
    print(data)
    return data


def main():
    cap = cv2.VideoCapture(1)
    cap.set(3,1920)
    cap.set(4,1080)

    while cap.isOpened():
        suc, img = cap.read()
        image = cv2.resize(img, (1920//3, 1080//3))
        cv2.imshow("imageres", image )
        # img = cv2.rotate(img, cv2.cv2.ROTATE_180)
        img = cv2.flip(img, 1)

        key = action(img)
        if key=="e":
            pass

        # cv2.imshow("image", img)
        if cv2.waitKey(10) & 0xFFF == ord("q"):
            break



if __name__ =="__main__":
    main()