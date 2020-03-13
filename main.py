import cv2 as cv
import matplotlib.pylab as plt
import numpy as np


img = cv.imread('lane.jpeg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

h = img.shape[0]
w = img.shape[1]


print(h, w)

plt.imshow(img)
plt.show()