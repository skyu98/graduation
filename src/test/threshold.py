import cv2
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import argrelextrema

def getThreshold(src):
    print("In python....")
    list = []
    gray = cv2.resize(src, ((int)(src.shape[0] / 3), (int)(src.shape[1] / 3)))
    for x in range(0, gray.shape[0]):
        for y in range(0, gray.shape[1]):
            list.append(gray[x, y]) 
    # data = array(list).reshape(-1, 1)
    # kde = KernelDensity(kernel='gaussian', bandwidth=9).fit(data)
    # x = linspace(0, max(list))
    # e = kde.score_samples(x.reshape(-1, 1))
    # # mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]

    bandwidth = 1.05 * np.std(list) * (len(list)**(-1/5))

    x_array = np.linspace(min(list), max(list), 40)
    print(111)
    y_array = get_kde(x_array, list, bandwidth)

    mi = argrelextrema(y_array, np.less)[0]
    print("Minimam res:", x_array[mi][0]) 
    return (int)(x_array[mi][0])
    
def get_kde(x_array, data_array, bandwidth=0.1):
    def gauss(x):
        import math
        return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * (x**2))
    
    y = []
    N = len(data_array)
    for x in x_array:
        res = 0
        if len(data_array) == 0:
            return 0
        for i in range(len(data_array)):
            res += gauss((x - data_array[i]) / bandwidth)
        res /= (N * bandwidth)
        y.append(res)
    return np.array(y)
    
def display(img):
    print("In python....")
    cv2.imshow('py', img)
    cv2.waitKey(0)
    return 23
