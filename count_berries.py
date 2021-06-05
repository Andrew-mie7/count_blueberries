from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import ipdb
import sys


def img_show(img):
    cv2.imshow('img', img)
    while 1:
        k = cv2.waitKey(0)
        if k == 27:     # Esc key to exit the whole progress
            cv2.destroyAllWindows()
            sys.exit()
        elif k == 32:
            cv2.destroyAllWindows()
            break       # Space key to keep going
        else:
            print(k)    # else print its value


def main():
    original_img = cv2.imread('./test3.jpg')
    img_show(original_img)

    GrayImage = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    GrayImage = cv2.medianBlur(GrayImage, 5)
    ret, th = cv2.threshold(GrayImage, 140, 255, cv2.THRESH_BINARY_INV)
    img_show(th)

    kernel = np.ones((5, 5), np.uint8)

    dilation = cv2.dilate(th, kernel, iterations=2)
    img_show(dilation)

    erosion = cv2.erode(dilation, kernel, iterations=5)
    img_show(erosion)

    denoised = erosion
    img_show(denoised)

    boundary = cv2.Canny(denoised, 30, 100)
    img_show(boundary)

    MIN_RADIUS = 100
    MAX_RADIUS = round(MIN_RADIUS*1.5)
    DISTANCE = round(MIN_RADIUS*1.5)
    circles = cv2.HoughCircles(boundary, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=10,
                               minRadius=100, maxRadius=150)
    circles = np.uint16(np.around(circles))

    # plt_bg = original_img.copy()
    plt_bg = denoised.copy()
    plt_bg = cv2.cvtColor(plt_bg, cv2.COLOR_GRAY2BGR)


    count = 0
    for idx, i in enumerate(circles[0, :]):
        # Skip when the center of the circle is on white
        isOnBg = plt_bg[i[1], i[0]] == np.array([0, 0, 0])
        if isOnBg.all():
            continue

        count += 1
        cv2.circle(plt_bg, (i[0], i[1]), i[2], (0, 255, 0), 5)
        print(f"{idx:>02}: {i}")

        # draw the center of the circle
        cv2.circle(plt_bg, (i[0], i[1]), 2, (0, 0, 255), 3)

    img_show(plt_bg)

    print(f'There are {count} berries.')

    # ipdb.set_trace()

    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
