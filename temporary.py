import cv2

img = cv2.imread("resources/rema.jpg")
img  = cv2.resize(img,(1920,1080))
cv2.imwrite("output.jpg", img)