from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import ipdb
import sys
from pprint import pprint


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
    original_img = cv2.imread('./test4.jpg')
    img_show(original_img)

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.medianBlur(gray_img, 5)
    ret, th = cv2.threshold(gray_img, 140, 255, cv2.THRESH_BINARY_INV)
    img_show(th)

    kernel_dilation = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(th, kernel_dilation, iterations=1)
    img_show(dilation)

    # distance transform
    ''' ref:
    https://stackoverflow.com/questions/26932891/detect-touching-overlapping-circles-ellipses-with-opencv-and-python

    https://docs.opencv.org/3.0-rc1/d2/dbd/tutorial_distance_transform.html
    '''
    dist = cv2.distanceTransform(dilation, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)
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

    # matched = cv2.normalize(matched, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # img_show(matched)

    # ret, th_matched = cv2.threshold(matched, 160, 255, cv2.THRESH_BINARY)
    # img_show(th_matched)

    mn, mx, _, _ = cv2.minMaxLoc(matched)
    THRESH = mx*0.25
    _, th_matched = cv2.threshold(matched, THRESH, 255, cv2.THRESH_BINARY)
    th_matched = cv2.normalize(th_matched, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    kernel_erosion = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(th_matched, kernel_erosion, iterations=1)
    img_show(erosion)

    source_BW = erosion

    source_BGR = cv2.cvtColor(source_BW, cv2.COLOR_GRAY2BGR)
    combined = cv2.addWeighted(original_img, .5, source_BGR, .5, 0.0)
    img_show(combined)


    boundary = cv2.Canny(source_BW, 30, 100)
    img_show(boundary)

    (contours, _) = cv2.findContours(
        boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(cnt) for cnt in contours]
    areas.sort()
    pprint(areas)
    pprint(sum(areas))
    pprint(sum(areas) / 13000)
    pprint(len(areas))

"""
    MIN_RADIUS = 40
    MAX_RADIUS = round(MIN_RADIUS*2)
    DISTANCE = round(MIN_RADIUS*3)
    ROUNDNESS = 3
    circles = cv2.HoughCircles(boundary, cv2.HOUGH_GRADIENT, 1, DISTANCE,
                               param1=50, param2=ROUNDNESS,
                               minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)
    circles = np.uint16(np.floor(circles))

    # original_img_bordered = cv2.copyMakeBorder(
    #     original_img, BORDER, BORDER, BORDER, BORDER,
    #     cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    plt_bg = combined.copy() # TEST

    # plt_bg = cv2.cvtColor(plt_bg, cv2.COLOR_GRAY2BGR)

    count = 0
    for idx, i in enumerate(circles[0, :]):
        # Skip when the center of the circle is on bg (black)
        isOnBg = source_BW[i[1], i[0]] == np.array([0, 0, 0])
        if isOnBg.all():
            continue

        count += 1
        # draw the the circle
        cv2.circle(plt_bg, (i[0], i[1]), i[2], (0, 255, 0), 5)
        print(f"{count:>02}: {i}")

        # draw the center of the circle
        cv2.circle(plt_bg, (i[0], i[1]), 2, (0, 0, 255), 3)

    img_show(plt_bg)

    print(f'There are {count} blueberries.')

    # ipdb.set_trace()

    # cv2.destroyAllWindows()

"""
if __name__ == '__main__':
    main()
