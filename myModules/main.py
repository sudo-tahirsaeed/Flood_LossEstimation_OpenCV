
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, pyplot as plt
from matplotlib import colors

def test():
    image = cv2.imread("flood.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 4)
    edged = cv2.Canny(blurred, 30, 225)
    #the more the high more accuracy for homes
    # define a (3, 3) structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # apply the dilation operation to the edged image
    dilate = cv2.dilate(edged, kernel, iterations=3)
    # find the contours in the dilated image

    nemo = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    contours, _ = cv2.findContours(nemo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # area = cv2.contourArea(contours)
    # print(area)
    nemo = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_copy = image.copy()
    # draw the contours on a copy of the original image
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    print(len(contours), "objects were found in this image.")
    # cv2.imshow("Dilated image", dilate)
    cv2.imshow("Flood", image_copy)

    cv2.imshow("RGB",nemo)
    cv2.waitKey(0)
def flood():
    image=cv2.imread("flood3.png")
    image=cv2.GaussianBlur(image,(3,3),3)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    canny = cv2.Canny(hsv, 155, 205)
    contours, hierarchies = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    # cv2.imshow("Img",image)
    # create emtpy mask
    mask = np.zeros(image.shape[:2], dtype="uint8")

    for c in contours:
        if cv2.contourArea(c) > 10:
            x, y, w, h = cv2.boundingRect(c)
            cv2.drawContours(mask, c, -1, (0,0,255), 1)

    # result = cv2.bitwise_or(image, image,mask=mask)
    cv2.imshow('Contours Drawn', mask)
    cv2.waitKey(0)
img=cv2.imread("green.png")
thresh, binary_image = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
cv2.imshow('Dirty Flood Water',binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# flood()
# # Load the image
# img = cv2.imread('flood2.png')
# canny = cv2.Canny(img, 125, 175)
# # Convert the image to grayscale
# gray =cv2.GaussianBlur(img,(3,3),3)
#
# # Find the contours in the image
# contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# # Create a numpy array filled with ones
# mask = np.ones(gray.shape, np.uint8)
# cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
#
# result = np.zeros(img.shape, np.uint8)
# result[mask == 255] = img[mask == 255]
# # Draw the contours on the mask as pixels with a value of 0 (black)
# cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
#
# # Show the mask
#
# cv2.imshow("Contour Mask", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#### NEW CODE
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower = np.array([252,104,6])
    # upper = np.array([209,85,10])
    #
    # mask = cv2.inRange(hsv, lower, upper)
    #
    # res = cv2.bitwise_and(hsv, hsv, mask=mask)
    # gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #
    # for i in contours:
    #     cnt = cv2.contourArea(i)
    #     if cnt > 1000:
    #         cv2.drawContours(hsv, [i], 0, (0, 0, 0), -1)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cnt = max(contours, key=cv2.contourArea)
    # area = cv2.contourArea(cnt)
    # cv2.putText(img, 'Gray area =' + str(area), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # cv2.imshow('img', hsv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def plot():
    image = cv2.imread("flood2.png",cv2.IMREAD_UNCHANGED)
    nemo = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image= cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # plt.imshow(nemo)
    #
    # # display that image
    # plt.show()
    # # 877 87
    #
    color = image[482,854]

    blue = int(color[0])
    green = int(color[1])
    red = int(color[2])
    print(blue,", ",green,",",red)
    #
    # nemo = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # r, g, b = cv2.split(nemo)
    # fig = plt.figure()
    # axis = fig.add_subplot(1, 1, 1)
    # pixel_colors = nemo.reshape((np.shape(nemo)[0] * np.shape(nemo)[1], 3))
    # norm = colors.Normalize(vmin=-1., vmax=1.)
    # norm.autoscale(pixel_colors)
    # pixel_colors = norm(pixel_colors).tolist()
    # axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    # axis.set_xlabel("Red")
    # axis.set_ylabel("Green")
    # # axis.set_zlabel("Blue")
    # plt.show()
    # cv2.imshow("Flooded", )
    # cv2.waitKey(0)

