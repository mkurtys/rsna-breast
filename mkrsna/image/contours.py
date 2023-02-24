import numpy as np
import cv2

def find_contours(img, 
                  threshold_level: int):
    blured = cv2.GaussianBlur(img, (7, 7), 0)
    
    thresholded = np.zeros_like(blured, dtype=np.uint8)
    thresholded[ blured >= threshold_level ] = 1
    
    cnts, _ = cv2.findContours(thresholded,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def largest_countour_and_coords_or_image_size(cnts, img):
    if cnts:
        cnt = max(cnts, key = cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        return cnt, (x,y,w,h)
    else:
        return None, (0, 0, img.shape[1], img.shape[0])