import cv2
import mediapipe as mp
from math import floor


class pointer:

    def getIndexFingerPoints(imgScan):

        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5, )

        image = imgScan
        frame = imgScan
        he, w, c = frame.shape

        # image = cv2.cvtColor(cv2.flip(image, -1), cv2.COLOR_BGR2RGB)


        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.cvtColor(imgScan, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        points = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h = hand_landmarks.landmark[8]
                # points = (h.x,h.y)
                a = float(h.x)
                b = float(h.y)
                x_px = min(floor(a * w), w - 1)
                y_px = min(floor(b * he), he - 1)
                points = [x_px, y_px]
                print(points)
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # cv2.imshow('MediaPipe Hands', image)
        if points:
            return points, image
        else:
            return "handsNotFound", image

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)

    while True:
        suc, img = cap.read()
        img = cv2.flip(img, 1)

        points, image = pointer.getIndexFingerPoints(img)
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX

            # org
            org = (50, 50)

            # fontScale
            fontScale = 1

            # Blue color in BGR
            color = (255, 0, 0)

            # Line thickness of 2 px
            thickness = 20

            # Using cv2.putText() method
            # image = cv2.putText(image, 'OpenCV', org, font,
            #                     fontScale, color, thickness, cv2.LINE_AA)
            # imageShow = cv2.putText(cv2.imread("output.jpg"),points, org, font, fontScale, color, thickness, cv2.LINE_AA)
            imageShow = cv2.circle(image, (points[0], points[1]), 10, color, thickness, cv2.LINE_AA)
            cv2.imshow("output", cv2.resize(imageShow, (1920//3, 1080//3)))
            if cv2.waitKey(100) & 0xFFF == ord("q"):
                break
        except Exception as e:
            print(e)
if __name__ =="__main__":
    main()