from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import ipdb
import sys


img = cv2.imread('./test.jpg')
cv2.imshow(winname='img' ,mat=img)
cv2.waitKey(0)

GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
GrayImage= cv2.medianBlur(GrayImage,5)
ret,th1 = cv2.threshold(GrayImage,140,255,cv2.THRESH_BINARY_INV)


th = th1

cv2.imshow(winname='th' ,mat=th)
cv2.waitKey(0)
kernel = np.ones((5,5),np.uint8)

dilation = cv2.dilate(th1,kernel,iterations=2)
cv2.imshow('dilation',dilation)
cv2.waitKey(0)

erosion = cv2.erode(dilation,kernel,iterations=5)
cv2.imshow('erosion',erosion)
cv2.waitKey(0)

boundry=cv2.Canny(erosion,30,100)
cv2.imshow(winname='boundry' ,mat=boundry)
cv2.waitKey(0)

circles = cv2.HoughCircles(boundry,cv2.HOUGH_GRADIENT,1,100,
                            param1=50,param2=15,minRadius=50,maxRadius=200)
circles = np.uint16(np.around(circles))
print(circles)

erosion = cv2.cvtColor(erosion, cv2.COLOR_GRAY2BGR)
print(type(erosion[10,10]))
plt_bg = erosion.copy()

count = 0
for idx, i in enumerate(circles[0,:]):
    # Skip when the center of the circle is on black
    isBlack = erosion[i[1], i[0]] == np.array([0,0,0])
    if isBlack.all():
        continue
    
    count += 1
    cv2.circle(plt_bg,(i[0],i[1]),i[2],(0,255,0),5)
    print(idx)
    print(i)
    
    # draw the center of the circle
    cv2.circle(plt_bg,(i[0],i[1]),2,(0,0,255),3)
    cv2.destroyAllWindows()
    cv2.imshow('detected circles',plt_bg)

    k = cv2.waitKey(0)
    if k==27:    # Esc key to stop
        break
    elif k==32:
        pass # Space to keep going
    else:
        print(k) # else print its value

print(f'There are {count} berries.')

# cv2.destroyAllWindows()