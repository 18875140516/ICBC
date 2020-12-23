import numpy as np


def checkPoint(point, contours):
    c = False
    for i in range(len(contours)):
        ii = (i + 1) % len(contours)
        if ((contours[i][1] > point[1]) != (contours[ii][1] > point[1])) \
                and point[0] - contours[i][0] < (contours[i][0] - contours[ii][0]) * (point[1] - contours[i][1]) \
                / (contours[i][1] - contours[ii][1]):
            c = not c
    return c
