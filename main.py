import cv2 as cv
import matplotlib.pylab as plt
import numpy as np

img = cv.imread('lane.jpeg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

h = img.shape[0]
w = img.shape[1]

print(h, w)

region_of_interest_vertex = np.array([[
    (0, h),
    (w/2, h/2 + 30),
    (w, h)
]], np.int32)

mask = np.zeros_like(img)
channel_count = img.shape[2]
match_mask = (255,) * channel_count
cv.fillPoly(mask, region_of_interest_vertex, match_mask)
masked_image = cv.bitwise_and(img, mask)

plt.imshow(masked_image)
plt.show()