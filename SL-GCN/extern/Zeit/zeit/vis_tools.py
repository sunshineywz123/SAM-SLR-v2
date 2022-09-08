import numpy as np
import cv2


row_col_ = {
    # 2: (2, 1),
    2: (1, 2),
    7: (2, 4),
    8: (2, 4),
    9: (3, 3),
    26: (4, 7)
}


def get_row_col(l):
    if l in row_col_.keys():
        return row_col_[l]
    else:
        from math import sqrt
        row = int(sqrt(l) + 0.5)
        col = int(l/ row + 0.5)
        if row*col<l:
            col = col + 1
        if row > col:
            row, col = col, row
        return row, col


def merge(images, row=-1, col=-1, resize=False, ret_range=False, **kwargs):
    if row == -1 and col == -1:
        row, col = get_row_col(len(images))
    height = images[0].shape[0]
    width = images[0].shape[1]
    ret_img = np.zeros((height * row, width * col, images[0].shape[2]), dtype=np.uint8) + 255
    ranges = []
    for i in range(row):
        for j in range(col):
            if i*col + j >= len(images):
                break
            img = images[i * col + j]
            # resize the image size
            img = cv2.resize(img, (width, height))
            ret_img[height * i: height * (i+1), width * j: width * (j+1)] = img
            ranges.append((width*j, height*i, width*(j+1), height*(i+1)))
    if resize:
        min_height = 1000
        if ret_img.shape[0] > min_height:
            scale = min_height/ret_img.shape[0]
            ret_img = cv2.resize(ret_img, None, fx=scale, fy=scale)
    if ret_range:
        return ret_img, ranges
    return ret_img