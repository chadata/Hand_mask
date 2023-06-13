import os
import cv2 as cv
import numpy as np
from  matplotlib import pyplot as plt

dir_list=os.listdir("C:\Hands\\")

for i in dir_list:
    #imgpath='C:\Hands\\'+str(i)+'.jpg'
    resultpath="C:\Result\\"+str(i)
    originalspath = "C:\Originals\\" + str(i)

    img = cv.imread('C:\Hands\\'+str(i))
    img = cv.resize(img, dsize=(1920, 1280))
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    img_hand = cv.inRange(img_hsv, lower, upper)
    #ycrcb보다는 hsv가 더 성능이 좋다

    # 경계선 찾음
    contours, hierarchy = cv.findContours(img_hand, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # 가장 큰 영역 찾기
    max = 0
    maxcnt = None
    for cnt in contours:
        area = cv.contourArea(cnt)
        if (max < area):
            max = area
            maxcnt = cnt

    # maxcontours의 각 꼭지점 다각선 만들기
    hull = cv.convexHull(maxcnt)

    # img 다 0으로 만들기
    mask = np.zeros(img.shape).astype(img.dtype)

    color = [255, 255, 255]

    # 경계선 내부 255로 채우기
    cv.fillPoly(mask, [maxcnt], color)
    img_hand = cv.bitwise_and(img, mask)
    cv.drawContours(img_hand, [maxcnt], 0, (255, 0, 0), 3)
    cv.drawContours(img_hand, [hull], 0, (0, 255, 0), 3)

    mask = cv.resize(mask, dsize=(1920, 1280))

    cv.imwrite(resultpath, mask)
    cv.imwrite(originalspath, img)

