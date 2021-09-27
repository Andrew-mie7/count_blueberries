import cv2
import numpy as np
import sys
import time

INPUT_IMG = './test5.jpg'
count = 0

# TODO use arrow key to move forward and backward
def img_show(img, img_name=None):
    global count
    if img_name is None:
        img_name = 'img'
    cv2.imshow(img_name, img)
    # cv2.imwrite(f'{count:02}_{img_name}.jpg', img)
    count += 1
    while True:
        k = cv2.waitKey(0)
        if k == 27:     # Esc key to exit the whole progress
            cv2.destroyAllWindows()
            sys.exit()
        elif k == 32:   # Space key to move on
            # cv2.destroyAllWindows()
            break
        else:
            time.sleep(0.1)


def main():
    original_img = cv2.imread(INPUT_IMG)
    img_show(original_img, 'original')

    # binarize
    grayed = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(grayed, 5)

    _, threshed = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY_INV)
    img_show(threshed, 'thresholded')

    kernel_dilate = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(threshed, kernel_dilate, iterations=1)

    binarized = dilated
    img_show(binarized, 'dilated')

    # distance transform
    ''' ref:
    https://stackoverflow.com/questions/26932891/detect-touching-overlapping-circles-ellipses-with-opencv-and-python

    https://docs.opencv.org/3.0-rc1/d2/dbd/tutorial_distance_transform.html
    '''
    dist = cv2.distanceTransform(binarized, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
    img_show(dist, 'distance transform')

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
    img_show(dist_templ, 'dist_templ')


    matched = cv2.matchTemplate(
        dist_bordered, dist_templ, cv2.TM_CCOEFF_NORMED)
    matched = matched.clip(min=0) # set neg nums to zero
    matched = cv2.normalize(matched, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img_show(matched, 'find lightest parts')



    mn, mx, _, _ = cv2.minMaxLoc(matched)
    THRESH = mx*0.25
    _, th_matched = cv2.threshold(matched, THRESH, 255, cv2.THRESH_BINARY)
    th_matched = cv2.normalize(th_matched, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    kernel_erosion = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(th_matched, kernel_erosion, iterations=1)
    img_show(erosion, 'binarized')

    source_BW = erosion

    source_BGR = cv2.cvtColor(source_BW, cv2.COLOR_GRAY2BGR)
    combined = cv2.addWeighted(original_img, .5, source_BGR, .5, 0.0)
    img_show(combined, 'overlayed')


    boundary = cv2.Canny(source_BW, 30, 100)
    img_show(boundary, 'with boundary')

    (contours, _) = cv2.findContours(
        boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(cnt) for cnt in contours]
    areas.sort()

    median = np.median(areas)
    std = np.std(areas)
    # pprint(f'{areas = }')
    print(f'{median = }')
    print(f'{std = }')

    avg_area = np.mean([area for area in areas if abs(area - median) < std ])
    count_by_area = sum(areas) / avg_area
    print(f'{avg_area = }')
    print(f'{count_by_area = }')
    img_show(combined, 'overlayed')


    MIN_RADIUS = 40
    MAX_RADIUS = round(MIN_RADIUS*2)
    DISTANCE = round(MIN_RADIUS*3)
    ROUNDNESS = 3
    circles = cv2.HoughCircles(boundary, cv2.HOUGH_GRADIENT, 1, DISTANCE,
                               param1=50, param2=ROUNDNESS,
                               minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)
    circles = np.uint16(np.floor(circles))

    plt_bg = combined.copy() # TEST

    # plt_bg = cv2.cvtColor(plt_bg, cv2.COLOR_GRAY2BGR)

    count_by_circle = 0
    for idx, i in enumerate(circles[0, :]):
        # Skip when the center of the circle is on bg (black)
        isOnBg = source_BW[i[1], i[0]] == np.array([0, 0, 0])
        if isOnBg.all():
            continue

        count_by_circle += 1
        # draw the circle as green
        cv2.circle(plt_bg, (i[0], i[1]), i[2], (0, 255, 0), 5)

        # draw the center of the circle as red
        cv2.circle(plt_bg, (i[0], i[1]), 2, (0, 0, 255), 3)

    print(f'{count_by_circle = }')
    img_show(plt_bg, 'with circles')


    # ipdb.set_trace()

    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
