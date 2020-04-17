import cv2 as cv
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
            cv.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=8)

    img = cv.addWeighted(img, 0.8, line_image, 1, 0.0)
    return img


def proccess_img(img):

    h = img.shape[0]
    w = img.shape[1]

    region_of_interest_vertex = np.array([[
        (0, h),
        (w/2, 1.2*h/2),
        (w, h)
        # (0+w/6, h-h/10),
        # (w/2, 3*h/5),
        # (w-w/6, h-h/10)
    ]], np.int32)

    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    canny_img = cv.Canny(gray_img, 100, 120)

    region_cropped = get_region(canny_img, region_of_interest_vertex)

    lines = cv.HoughLinesP(region_cropped,
                           rho=2,
                           theta=np.pi/180,
                           threshold=50,
                           lines=np.array([]),
                           minLineLength=40,
                           maxLineGap=100)

    if lines is not None:
        img_lines = draw_lines(img, lines)
        return img_lines
    else:
        return None


if __name__ == "__main__":

    cap = cv.VideoCapture('driving.mp4')

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # frame = cv.resize(frame, (1080, 720))
        new_frame = proccess_img(frame)
        if new_frame is None:
            cv.imshow('Lane Line Detection', frame)
        else:
            cv.imshow('Lane Line Detection', new_frame)
        cv.moveWindow('Lane Line Detection', 200, 200)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


