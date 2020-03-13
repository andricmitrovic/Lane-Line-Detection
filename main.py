import cv2 as cv
import matplotlib.pylab as plt
import numpy as np


def get_region(img, vertices):
    mask = np.zeros_like(img)
    match_mask = 255
    cv.fillPoly(mask, vertices, match_mask)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines):
    img = np.copy(img)
    line_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=5)

    img = cv.addWeighted(img, 0.7, line_image, 1, 0.0)
    return img


if __name__ == "__main__":

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

    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    canny_img = cv.Canny(gray_img, 100, 200)

    region_cropped = get_region(canny_img, region_of_interest_vertex)

    lines = cv.HoughLinesP(region_cropped,
                           rho=6,
                           theta=np.pi/60,
                           threshold=160,
                           lines=np.array([]),
                           minLineLength=5,
                           maxLineGap=15)

    img_lines = draw_lines(img, lines)

    plt.imshow(img_lines, cmap='gray')
    plt.show()
