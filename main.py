import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder

cap = cv2.VideoCapture(1) #because we want to start off with a webcam. We are going to capture from device number 0.
cap.set(3,640) #setting the width and the height TBC
cap.set(4,480) #setting the width and the height TBC

totalMoney = 0

myColorFinder = ColorFinder(False)
# Custom Orange Color
hsvVals = {'hmin': 0, 'smin': 186, 'vmin': 180, 'hmax': 30, 'smax': 255, 'vmax': 255}

def empty(a):
    pass

cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 640, 240)
cv2.createTrackbar("Threshold1", "Settings", 196, 255, empty)
cv2.createTrackbar("Threshold2", "Settings", 71, 255, empty)

def preProcessing(img):

    imgPre = cv2.GaussianBlur(img,(5,5),3) #Use this to blur it.
    thresh1 = cv2.getTrackbarPos("Threshold1", "Settings")
    thresh2 = cv2.getTrackbarPos("Threshold2", "Settings")
    imgPre = cv2.Canny(imgPre, thresh1,thresh2) #Use this to find the edges. Using trackbars to select the level of noise we'll accept
    kernel = np.ones((3,3),np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations = 1)
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel)
    return imgPre

while True:
    success, img = cap.read() # This line reads a frame from the video capture object. success is a boolean variable indicating if the frame was read successfully. img is a NumPy array containing the captured image data.
    imgPre = preProcessing(img)
    imgContours, conFound = cvzone.findContours(img, imgPre, minArea=20) #essentially analyzes the preprocessed image to identify closed shapes (objects) based on changes in pixel intensity. By applying specific thresholds and filtering techniques in the pre-processing steps, this function aims to identify the boundaries of potential objects in the image.

    totalMoney =0
    imgCount = np.zeros((480, 640, 3), np.uint8)

    if conFound:
        for count, contour in enumerate(conFound):
            peri = cv2.arcLength(contour['cnt'], True)
            approx = cv2.approxPolyDP(contour['cnt'], 0.02 * peri, True)
            if len(approx) > 5:
               area = contour['area']
               x, y, w, h = contour['bbox']
               imgCrop = img[y:y+h,x:x+w]
               #cv2.imshow(str(count), imgCrop)
               imgColor, mask = myColorFinder.update(imgCrop, hsvVals)
               whitePixelCount = cv2.countNonZero(mask)
               print("NUMBER OF WHITE PIXELS", whitePixelCount)
               cv2.imshow("imgColor",imgColor)  # This line displays the captured image on a window named "Image" using the OpenCV imshow function.
               print("AREA OF COIN", contour['area'])
               if area<2100 and whitePixelCount < 100:
                   totalMoney +=5
               elif 2250<area<2900 and whitePixelCount < 100:
                   totalMoney +=20
               elif 1800<area<2700 and whitePixelCount > 100:
                   totalMoney +=1
               elif 2500<area<3700 and whitePixelCount < 100:
                   totalMoney +=10
               elif 2700<area<5000:
                   totalMoney +=2

    #print(totalMoney)
    cvzone.putTextRect(imgCount,f'Gbp.{totalMoney}',(100,200), scale=10, offset=30, thickness=7 )
    imgStacked = cvzone.stackImages([img, imgPre, imgContours, imgCount], 2,1)
    cvzone.putTextRect(imgStacked,f'Gbp.{totalMoney}',(50,50) )
    cv2.imshow("Image", imgStacked) # This line displays the captured image on a window named "Image" using the OpenCV imshow function.

    cv2.waitKey(1) # This line waits for a key press for 1 millisecond. If a key is pressed within this time, the loop will break. This allows you to stop the program by pressing any key.
