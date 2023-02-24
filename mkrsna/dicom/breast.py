import pydicom
import cv2
import numpy as np
from ..image.contours import find_contours, largest_countour_and_coords_or_image_size
from ..image.utils import normalization
from ..constants import U16MAX
from ..image.utils import image_resize


def data_to_monochrome2_if_needed(data, ds):     
    if ds.PhotometricInterpretation == "MONOCHROME1":
        data_max = np.amax(data)
        return data_max - data  
    else:
        return data

def img_crop(img, threshold_level):
    contours = find_contours(img, threshold_level=threshold_level)
    #cv.drawContours(	image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]	) -> 	image
    max_cnt, cnt_coords = largest_countour_and_coords_or_image_size(contours, img)
    max_cnt_mask=None
    max_cnt_mask = np.zeros_like(img, dtype=np.uint8)
    if max_cnt is not None:
        cv2.drawContours(max_cnt_mask, [max_cnt], -1 , 1, cv2.FILLED)
        croped_img = np.where(max_cnt_mask > 0 , img, max_cnt_mask)
        # img = cv2.bitwise_or(img, img, mask=max_cnt_mask)
        x,y,w,h = cnt_coords
        croped_img = croped_img[y:y+h, x:x+w]
    else:
        croped_img = img.copy()
        
    return croped_img, cnt_coords, max_cnt_mask 

def read_breast_dicom_data_and_roi(filename):
    ds = pydicom.dcmread(filename)
    data = ds.pixel_array
    voi_data = pydicom.pixel_data_handlers.apply_voi_lut(data, ds)
    # data = data_to_monochrome2_if_needed(data, ds)
    voi_data = data_to_monochrome2_if_needed(voi_data, ds)
    img_croped, crop_coords, _ = img_crop(voi_data, threshold_level=5)
    
    img_croped, _, _ = normalization(img_croped)
    img_croped = img_croped*U16MAX
    img_croped=img_croped.astype(np.uint16)
    return data, img_croped, crop_coords


def read_breast_dicom_roi(filename, resize_longer_axis_to=None):
    data, img_croped, crop_coords = read_breast_dicom_data_and_roi(filename)
    if resize_longer_axis_to:
        img_croped=image_resize(img_croped, height=resize_longer_axis_to)
    return img_croped