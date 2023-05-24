# import numpy as np
# img = cv.imread("flood.jpg")
# blank = np.zeros(img.shape, dtype='uint8')
# # Displaying the image
# cv.imshow('image', img)
# gauss = cv.GaussianBlur(img, (3,3), 0)
# cv.imshow('Gaussian Blur', gauss)
# # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# # cv.imshow('HSV', hsv)
# canny = cv.Canny(gauss, 125, 175)
# cv.imshow('Canny Edges', canny)
#
# # ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# # cv.imshow('Thresh', thresh)
#
# contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# print(f'{len(contours)} contour(s) found!')
#
# cv.drawContours(blank, contours, -1, (0,0,255), 1)
# cv.imshow('Contours Drawn', blank)
#
# cv.waitKey(0)
# importing the module
import cv2


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color = image[y, x]
        blue = int(color[0])
        green = int(color[1])
        red = int(color[2])
        print(blue, ", ", green, ",", red)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)


        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)


# driver function
if __name__ == "__main__":
    # reading the image
    img = cv2.imread('flood3.png')
    img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # displaying the image
    cv2.imshow('image', img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()