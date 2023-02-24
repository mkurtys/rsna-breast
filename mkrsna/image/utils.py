import numpy as np
import cv2


def normalization_with_min_max(img, val_min, val_max):
    if val_min<val_max:
        return (img - val_min)/(val_max - val_min)
    else:
        return img

def normalization(img):
    val_min = np.amin(img)
    val_max = np.amax(img)
    return normalization_with_min_max(img, val_min, val_max), val_min, val_max
    

def truncation_normalization(img, val_min, val_max):
    truncated = np.clip(img,val_min, val_max)
    normalized = (truncated - val_min)/(val_max - val_min)
    return normalized


def image_resize(image, width = None, height = None, inter = cv2.INTER_LINEAR):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)

    return resized