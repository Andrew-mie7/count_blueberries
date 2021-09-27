import sys

import cv2
import numpy as np

def q():
    cv2.destroyAllWindows()
    sys.exit()




def process(im: np.ndarray):
    SCALAR = 5
    im = cv2.resize(im, (im.shape[1]*SCALAR, im.shape[0]*SCALAR), cv2.INTER_AREA)

    TH = 120
    TH_matched = 120

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)

    blurred = cv2.medianBlur(gray, 9*SCALAR)
    # cv2.imshow('blurred', blurred)

    th, bw = cv2.threshold(blurred, TH, 255, cv2.THRESH_BINARY)
    # cv2.imshow('bw', bw)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3*SCALAR, 3*SCALAR))
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('morph', morph)

    # masked = cv2.bitwise_and(gray, gray, mask=morph)
    # cv2.imshow('masked', masked)

    dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)
    cv2.imshow('dist', dist)

    dist[dist < 0.3] = 0
    # dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
    # dist = dist.astype(np.uint8)
    # cv2.imshow('dist-', dist)

    borderSize = 50
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
    kernel_erosion = np.ones((5, 5), np.uint8)
    matched = cv2.erode(matched, kernel_erosion, iterations=1)
    cv2.imshow('matched', matched)

    mn, mx, _, _ = cv2.minMaxLoc(matched)
    THRESH = mx*0.25
    _, th_matched = cv2.threshold(matched, THRESH, 255, cv2.THRESH_BINARY)
    th_matched = cv2.normalize(th_matched, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow('th_matched', th_matched)


    peaks8u = cv2.convertScaleAbs(th_matched)


    # boundary = cv2.Canny(peaks8u, 30, 100)
    # cv2.imshow('boundary', boundary)


    contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
        # to use as mask

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 50*SCALAR*SCALAR:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        _, mx, _, mxloc = cv2.minMaxLoc(morph[y:y+h, x:x+w], peaks8u[y:y+h, x:x+w])
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.drawContours(im, contours, i, (0, 0, 255), 1)
    
    return im
    # cv2.imshow('circles', im)
    # cv2.waitKey(0)



# # Reading Videos
# capture = cv2.VideoCapture('./video_2.mp4')
# while True:
#     isTrue, frame = capture.read()
#     if isTrue == True:
        
#         frame_resized = frame[150:-150, 300:-500]
#         cv2.imshow('Video Resized', process(frame_resized))

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cv2.waitKey(-1)
#     else:
#         break
# capture.release()

im = cv2.imread('./count_bee.png')
done = process(im)
cv2.imshow('', done)


cv2.waitKey(0)
cv2.destroyAllWindows()