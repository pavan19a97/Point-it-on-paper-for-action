import cv2
from pyzbar.pyzbar import decode
class barcodeDecoder:
    def getBarcodeData(imgScan):
        width = 1920
        height = 1080
        resizedImg = cv2.resize(imgScan, (width//2, height//2))
        for barcode in decode(resizedImg):
            return barcode.data.decode('utf-8')
        else:
            return False