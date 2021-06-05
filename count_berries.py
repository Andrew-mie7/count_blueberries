from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import ipdb
import sys

def wait_key():
    while 1:
        k = cv2.waitKey(0)
        if k == 27:    # Esc key to stop
            break
        elif k == 32:
            pass  # Space to keep going
        else:
            print(k)  # else print its value

def main():
    img = cv2.imread('./test3.jpg')
    cv2.imshow(winname='img', mat=img)
    wait_key()

    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    GrayImage = cv2.medianBlur(GrayImage, 5)
    ret, th = cv2.threshold(GrayImage, 140, 255, cv2.THRESH_BINARY)
    cv2.imshow(winname='th', mat=th)
    wait_key()


    kernel = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(th, kernel, iterations=2)
    cv2.imshow('erosion', erosion)
    wait_key()

    dilation = cv2.dilate(erosion, kernel, iterations=10)
    cv2.imshow('dilation', dilation)
    wait_key()

    denoised = dilation
    cv2.imshow('denoised', denoised)
    wait_key()

    boundry = cv2.Canny(denoised, 30, 100)
    cv2.imshow(winname='boundry', mat=boundry)
    wait_key()

    MIN_RADIUS = 100
    MAX_RADIUS = round(MIN_RADIUS*1.5)
    DISTANCE = round(MIN_RADIUS*1.5)
    circles = cv2.HoughCircles(boundry, cv2.HOUGH_GRADIENT, 1, 100,
                            param1=50, param2=10,
                            minRadius=100, maxRadius=150)
    circles = np.uint16(np.around(circles))

    plt_bg = img.copy()
    # plt_bg = denoised.copy()
    # plt_bg = cv2.cvtColor(plt_bg, cv2.COLOR_GRAY2BGR)


    cv2.destroyAllWindows()


    count = 0
    for idx, i in enumerate(circles[0, :]):
        # Skip when the center of the circle is on white
        isWhite = plt_bg[i[1], i[0]] == np.array([255, 255, 255])
        if isWhite.all():
            continue

        count += 1
        cv2.circle(plt_bg, (i[0], i[1]), i[2], (0, 255, 0), 5)
        print(f"{idx:>02}: {i}")

        # draw the center of the circle
        cv2.circle(plt_bg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', plt_bg)

    print(f'There are {count} berries.')


    # ipdb.set_trace()

    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()