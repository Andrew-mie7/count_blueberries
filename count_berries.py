from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import ipdb
import sys


def img_show(img, img_name=None):
    if img_name is None:
        img_name = 'img'
    cv2.imshow(img_name, img)
    while 1:
        k = cv2.waitKey(0)
        if k == 27:     # Esc key to exit the whole progress
            cv2.destroyAllWindows()
            sys.exit()
        elif k == 32:
            # cv2.destroyAllWindows()
            break       # Space key to keep going
        else:
            print(k)    # else print its value


def main():
    original_img = cv2.imread('./test2.jpg')
    img_show(original_img)

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.medianBlur(gray_img, 5)
    ret, th = cv2.threshold(gray_img, 140, 255, cv2.THRESH_BINARY_INV)
    img_show(th)

    kernel_d = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(th, kernel_d, iterations=2)
    img_show(dilation)

    source = dilation
    # distance transform
    ''' ref:
    https://stackoverflow.com/questions/26932891/detect-touching-overlapping-circles-ellipses-with-opencv-and-python

    https://docs.opencv.org/3.0-rc1/d2/dbd/tutorial_distance_transform.html
    '''
    dt = cv2.distanceTransform(source, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist = cv2.normalize(dt, None, 0, 1, cv2.NORM_MINMAX)
    img_show(dist)

    BORDER = 100
    PADDING = 20
    dist_bordered = cv2.copyMakeBorder(
        dist, BORDER, BORDER, BORDER, BORDER,
        cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    kernel_template = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*(BORDER-PADDING)+1, 2*(BORDER-PADDING)+1))
    kernel_template = cv2.copyMakeBorder(
        kernel_template, PADDING, PADDING, PADDING, PADDING,
        cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    dist_templ = cv2.distanceTransform(
        kernel_template, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    matched = cv2.matchTemplate(
        dist_bordered, dist_templ, cv2.TM_CCOEFF_NORMED)
    img_show(matched)

    matched = cv2.normalize(matched, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img_show(matched)

    ret, th_matched = cv2.threshold(matched, 180, 255, cv2.THRESH_BINARY)
    img_show(th_matched)

"""
    ret, th_dist = cv2.threshold(dist, 120, 255, cv2.THRESH_BINARY)
    img_show(th_dist)

    kernel_e = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(dilation, kernel_e, iterations=30)
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
    circles = np.uint16(np.floor(circles))

    # TEST plt_bg = original_img.copy()
    plt_bg = denoised.copy()
    plt_bg = cv2.cvtColor(plt_bg, cv2.COLOR_GRAY2BGR)

    count = 0
    for idx, i in enumerate(circles[0, :]):
        # Skip when the center of the circle is on bg (black)
        isOnBg = plt_bg[i[1], i[0]] == np.array([0, 0, 0])
        if isOnBg.all():
            continue

        count += 1
        # draw the the circle
        cv2.circle(plt_bg, (i[0], i[1]), i[2], (0, 255, 0), 5)
        print(f"{idx:>02}: {i}")

        # draw the center of the circle
        cv2.circle(plt_bg, (i[0], i[1]), 2, (0, 0, 255), 3)

    img_show(plt_bg)

    print(f'There are {count} blueberries.')

    # ipdb.set_trace()

    # cv2.destroyAllWindows()

"""
if __name__ == '__main__':
    main()
