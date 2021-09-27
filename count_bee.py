import sys

import cv2
import numpy as np

def q():
    cv2.destroyAllWindows()
    sys.exit()




def process(im: np.ndarray):
    SCALAR = 1
    im = cv2.resize(im, (im.shape[1]*SCALAR, im.shape[0]*SCALAR), cv2.INTER_AREA)

    TH = 120
    ERODE = 3

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)

    blurred = cv2.medianBlur(gray, 9*SCALAR)
    # cv2.imshow('blurred', blurred)
    light = cv2.blur(gray, (99*SCALAR, 99*SCALAR))
    light = cv2.bitwise_not(light)
    
    light = cv2.normalize(light, None, 0, 1, cv2.NORM_MINMAX)
    cv2.imshow('light', light)

    th, bw = cv2.threshold(blurred, TH, 255, cv2.THRESH_BINARY)
    # cv2.imshow('bw', bw)

    kernel_erosion = np.ones((ERODE, ERODE), np.uint8)
    eroded = cv2.erode(bw, kernel_erosion, iterations=1)
    cv2.imshow('eroded', eroded)

    dist = cv2.distanceTransform(eroded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)
    cv2.imshow('dist', dist)

    # dist[dist < 0.3] = 0
    # dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
    # dist = dist.astype(np.uint8)
    # cv2.imshow('dist-', dist)

    borderSize = 10
    distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize, 
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    # cv2.imshow('distborder', distborder)


    gap = 1
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(borderSize-gap)+1, 2*(borderSize-gap)+1))
    kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap, 
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)

    distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # cv2.imshow('distTempl', distTempl)

    matched = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
    cv2.imshow('matched', matched)
    kernel_erosion = np.ones((ERODE, ERODE), np.uint8)
    matched = cv2.erode(matched, kernel_erosion, iterations=1)
    cv2.imshow('matched2', matched)

    mn, mx, _, _ = cv2.minMaxLoc(matched)
    THRESH = mx*0.25
    _, th_matched = cv2.threshold(matched, THRESH, 255, cv2.THRESH_BINARY)
    th_matched = cv2.normalize(th_matched, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow('th_matched', th_matched)


    final = th_matched

    contours, hierarchy = cv2.findContours(final, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
        # to use as mask

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 10*SCALAR*SCALAR:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        _, mx, _, mxloc = cv2.minMaxLoc(final[y:y+h, x:x+w], final[y:y+h, x:x+w])
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.drawContours(im, contours, i, (0, 0, 255), 1)
    
    return im
    # cv2.imshow('circles', im)
    # cv2.waitKey(0)



# Reading Videos
capture = cv2.VideoCapture('./video.mp4')
while True:
    isTrue, frame = capture.read()
    if isTrue == True:
        
        frame_resized = frame[150:-150, 300:-500]
        cv2.imshow('Video Resized', process(frame_resized))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.waitKey(-1)
    else:
        break
capture.release()

# im = cv2.imread('./bee_test.png')
# done = process(im)
# cv2.imshow('', done)


cv2.waitKey(0)
cv2.destroyAllWindows()