import cv2
import numpy as np
    
def display(img):
    print("In python....")
    cv2.imshow('py', img)
    cv2.waitKey(0)
    return tuple([12, 23, 34, 45])
