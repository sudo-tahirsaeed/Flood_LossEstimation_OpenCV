import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, pyplot as plt
from matplotlib import colors
# cv2.imshow("Img", canny)
def estimatation():
    image = cv2.imread("flood2.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    canny = cv2.Canny(gray, 30, 550)
    cv2.imshow("ok",canny)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 2)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    houses = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000 and area < 50000:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            if cv2.isContourConvex(box):
                houses.append(box)
    for house in houses:
        cv2.drawContours(image, [house], 0, (0, 0, 255), 2)




    cv2.waitKey()
    # edged = cv2.Canny(blurred, 30, 225)

estimatation()


def detection():
    cap= cv2.VideoCapture('floodvid4.mp4')
    i=0
    while(cap.isOpened()):
        ret, image = cap.read()
        ori=image;

        width = int(image.shape[1] * 70 / 100)
        height = int(image.shape[0] * 40 / 100)
        dim = (width, height)

        # resize image
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        ori = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        if ret == False:
            break
        image = cv2.GaussianBlur(image, (3, 3), 10)

        thresh, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #red
        BGR = np.array([4,4,252])
        upper = BGR + 100
        lower = BGR - 20
        range=cv2.inRange(binary_image, lower, upper)
    #yellow
        BGR = np.array([4, 252, 252])
        upper = BGR + 100
        lower = BGR - 30
        range1 = cv2.inRange(binary_image, lower, upper)

    #magenta
        BGR = np.array([255, 255, 0])
        upper = BGR + 100
        lower = BGR - 30
        range2 = cv2.inRange(binary_image, lower, upper)
    #blue
        BGR = np.array([255, 255, 4])
        upper = BGR + 100
        lower = BGR - 30
        range3 = cv2.inRange(binary_image, lower, upper)
        canny = cv2.Canny(hsv, 125, 175)
        # cv2.imshow('Canny Edges', canny)

        # # ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
        # # cv.imshow('Thresh', thresh)
        blank = np.zeros(image.shape, dtype='uint8')


        contours, hierarchies = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print(f'{len(contours)} contour(s) found!')
        mask = np.zeros(image.shape[:2], dtype=image.dtype)
        for c in contours:
            if cv2.contourArea(c) > 10:
                x, y, w, h = cv2.boundingRect(c)
                cv2.drawContours(mask, c, -1, (0,0,255), 1)

        result = cv2.bitwise_and(image, image, mask=mask)
        # cv2.imshow('Contours Drawn', result)
        # bitwiseOr = cv2.bitwise_and(range,range1,range2,range3,mask=)

        # cv2.imshow("Flooded",range)
        image[range>0]=(0,0,255)
        image[range1>0]=(0,0,255)
        image[range2 > 0] = (0, 0, 255)
        image[range3 > 0] = (0, 0, 255)

        cc = cv2.vconcat([image, ori])
        cv2.imshow('framed', cc)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
