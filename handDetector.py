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
        w, he, c = frame.shape

        # image = cv2.cvtColor(cv2.flip(image, -1), cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(imgScan, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        points = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h = hand_landmarks.landmark[8]

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