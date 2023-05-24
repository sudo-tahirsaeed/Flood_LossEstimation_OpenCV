import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, pyplot as plt
from matplotlib import colors
def data():
    img=cv2.imread("flood2.png")
    hsv_image=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # Clean flood water color range in HSV format
    clean_min = np.array([90, 50, 50])
    clean_max = np.array([110, 255, 255])

    # Dirty flood water color range in HSV format
    dirty_min = np.array([0, 0, 0])
    dirty_max = np.array([180, 50, 50])
    # Create mask for clean flood water
    clean_mask = cv2.inRange(hsv_image, clean_min, clean_max)

    # Create mask for dirty flood water
    dirty_mask = cv2.inRange(hsv_image, dirty_min, dirty_max)
    # Apply clean mask to original image
    clean_water = cv2.bitwise_and(img, img, mask=clean_mask)

    # Apply dirty mask to original image
    dirty_water = cv2.bitwise_and(img, img, mask=dirty_mask)
    cv2.imshow('Clean Flood Water', clean_water)
    cv2.imshow('Dirty Flood Water', dirty_water)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
data()
    # img = cv2.imread('flood2.png')
    # gray =cv2.GaussianBlur(img,(3,3),8)
    # hsv=cv2.cvtColor(gray,cv2.COLOR_BGR2HSV)
    # cv2.imshow("pak",hsv)
    # canny = cv2.Canny(hsv, 255, 85)
    # # Convert the image to grayscale
    #
    #
    # # Find the contours in the image
    # contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # Create a numpy array filled with ones
    # mask = np.ones(gray.shape, np.uint8)
    # for c in contours:
    #     if cv2.contourArea(c)>2:
    #         pass
    # cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    # # cv2.imshow("m",mask)
    # result = np.zeros(img.shape, np.uint8)
    # # cv2.imshow("a",img)
    # result[mask == 255] = (img[mask == 255])
    # mask_inv = cv2.bitwise_not(result)
    # # Draw the contours on the mask as pixels with a value of 0 (black)
    # # cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    # cv2.imshow("a",mask_inv)
    # # Show the mask
    # #
    # cv2.imshow("Contour Mask", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()