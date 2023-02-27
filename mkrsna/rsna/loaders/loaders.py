import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mkrsna.dicom.breast import read_breast_dicom_roi

def resolve_img_path(img_folder, patient_id, image_id, extension="png", has_patient_folder_sturcture=True):
    if has_patient_folder_sturcture:
        return os.path.join(img_folder, str(patient_id), f"{image_id}.{extension}")
    else:
        return os.path.join(img_folder, f"{image_id}.{extension}")

def load_img(img_path, resize_dim=None, resize_aspect_ratio=None, transform=None):    
    # need anydepth to load 16bit grayscale png
    # don't know if pngs are bgr or rgb
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"No image {img_path} found")
    # cast to np float, as torch can't into uint16
    if resize_dim:
        img = cv2.resize(img, resize_dim)
    if resize_aspect_ratio and resize_aspect_ratio!=1.0:
        img = cv2.resize(img, tuple(int(d*resize_aspect_ratio) for d in img.shape[::-1]))
                
    # convert to RGB for pretrained networks
    # maybe we will move it to the different place
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[0] == 1):
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pass
    else:
        img = cv2.cvtColor(img, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    
    # should we really normalize here? Why not in transforms?
    # again there are problems (there are?) with 16bits PNGS
    img = img.astype(np.float32)
    img_max = np.amax(img)
    img_min = np.amin(img)
    img_range = img_max-img_min
    if img_range > 0:
        img = (img-img_min)/img_range
    
    if transform:
        img = transform(image=img)['image']      
    
    # shouldn't be done in transforms?
    img = torch.tensor(img, dtype=torch.float)
    return img


def load_dicom(dicom_path, resize_longer_axis_to=None, pre_resize_for_countours_aspect=None, transform=None):
    img = read_breast_dicom_roi(dicom_path, resize_longer_axis_to=resize_longer_axis_to, pre_resize_for_countours_aspect=pre_resize_for_countours_aspect)
    img = img.astype(np.float32)
    img_max = np.amax(img)
    img_min = np.amin(img)
    img_range = img_max-img_min
    if img_range > 0:
        img = (img-img_min)/img_range
    if transform:
        img = transform(image=img)['image']
    img = torch.tensor(img, dtype=torch.float)
    return img

# supports both 3-channels PNGs, and single-channel 16 bit PNG
class RSNAData(Dataset):
    def __init__(self, df,
                 img_folder,
                 resize_dim=None,
                 resize_aspect_ratio=None,
                 resize_longer_axis_to=None,
                 pre_resize_for_countours_aspect=None,
                 transform=None,
                 is_test=False,
                 has_patient_folder_sturcture=False,
                 extension="png",
                 return_filepath=False):
        
        assert not (resize_dim and resize_aspect_ratio)
        self.df = df
        self.is_test = is_test
        self.transform = transform
        self.img_folder = img_folder
        self.resize_dim = resize_dim
        self.resize_aspect_ratio=resize_aspect_ratio
        self.extension = extension
        self.has_patient_folder_sturcture=has_patient_folder_sturcture
        self.return_filepath=return_filepath
        self.resize_longer_axis_to = resize_longer_axis_to
        self.pre_resize_for_countours_aspect = pre_resize_for_countours_aspect
        
    def __getitem__(self, idx):
        row = self.df.loc[idx, :]

        img_path = resolve_img_path(self.img_folder, row["patient_id"], row["image_id"], self.extension, self.has_patient_folder_sturcture)
        if self.extension == "dcm" or self.extension == "dicom":
            img = load_dicom(img_path, self.resize_longer_axis_to, self.pre_resize_for_countours_aspect, self.transform)
        else:
            img = load_img(img_path, self.resize_dim, self.resize_aspect_ratio, self.transform)
        
        if not self.is_test:
            target = self.df['cancer'][idx]
            target = torch.tensor(target, dtype=torch.float32)
            if self.return_filepath:
                return img, target, img_path, idx
            return (img, target, idx)
        if self.return_filepath:
            return img, img_path, idx
        return (img, idx)
    
    def __len__(self):
        return len(self.df)