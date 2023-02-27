import pydicom
import dicomsdl
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


def load_image_dicomsdl(img_path, voi_lut=False):
    dataset = dicomsdl.open(img_path)
    img = dataset.pixelData()
    
    if voi_lut:
        # Load only the variables we need
        center = dataset["WindowCenter"]
        width = dataset["WindowWidth"]
        bits_stored = dataset["BitsStored"]
        voi_lut_function = dataset["VOILUTFunction"]

        # For sigmoid it's a list, otherwise a single value
        if isinstance(center, list):
            center = center[0]
        if isinstance(width, list):
            width = width[0]

        # Set y_min, max & range
        y_min = 0
        y_max = float(2**bits_stored - 1)
        y_range = y_max

        # Function with default LINEAR (so for Nan, it will use linear)
        if voi_lut_function == "SIGMOID":
            img = y_range / (1 + np.exp(-4 * (img - center) / width)) + y_min
        else:
            # Checks width for < 1 (in our case not necessary, always >= 750)
            center -= 0.5
            width -= 1

            below = img <= (center - width / 2)
            above = img > (center + width / 2)
            between = np.logical_and(~below, ~above)

            img[below] = y_min
            img[above] = y_max
            if between.any():
                img[between] = (
                    ((img[between] - center) / width + 0.5) * y_range + y_min
                )
    
    if dataset["PhotometricInterpretation"] == "MONOCHROME1":
        img = np.amax(img) - img

    return img


def read_breast_dicom_data_and_roi(filename, pre_resize_for_countours_aspect=None, use_dicomsdl=False):
    if use_dicomsdl:
        voi_data = load_image_dicomsdl(filename, voi_lut=True)
    else:
        ds = pydicom.dcmread(filename)
        data = ds.pixel_array
        voi_data = pydicom.pixel_data_handlers.apply_voi_lut(data, ds)
        # data = data_to_monochrome2_if_needed(data, ds)
        voi_data = data_to_monochrome2_if_needed(voi_data, ds)
    if pre_resize_for_countours_aspect:
        (h, w) = voi_data.shape[:2]
        voi_data_resized = cv2.resize(voi_data, ( int(pre_resize_for_countours_aspect*w), int(pre_resize_for_countours_aspect*h)))
        _ , crop_coords, _ = img_crop(voi_data_resized, threshold_level=5)
        crop_coords_rescaled = [int(x*1/pre_resize_for_countours_aspect) for x in crop_coords]
        x,y,w,h = crop_coords_rescaled
        img_croped  = voi_data[y:y+h, x:x+w]
    else:
        img_croped, crop_coords, _ = img_crop(voi_data, threshold_level=5)
    
    img_croped, _, _ = normalization(img_croped)
    img_croped = img_croped*U16MAX
    img_croped=img_croped.astype(np.uint16)
    return voi_data, img_croped, crop_coords


def read_breast_dicom_roi(filename, resize_longer_axis_to=None, pre_resize_for_countours_aspect=None):
    data, img_croped, crop_coords = read_breast_dicom_data_and_roi(filename, pre_resize_for_countours_aspect=pre_resize_for_countours_aspect, use_dicomsdl=True)
    if resize_longer_axis_to:
        img_croped=image_resize(img_croped, height=resize_longer_axis_to)
    return img_croped