import cv2
import numpy as np

image = cv2.imread("newflood.jpg")
image = cv2.GaussianBlur(image, (3, 3), 2)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh, binary_image = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
canny = cv2.Canny(binary_image, 125, 175)
cv2.imshow("a",canny)
contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image,contours,-1,(0,255,0),1)
cv2.imshow("12",image)

cv2.imshow("1",binary_image)
cv2.waitKey()
