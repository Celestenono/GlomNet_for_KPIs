import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np

from monai.data.dataset import Dataset
from openslide import OpenSlide

MAGNIFICATIONS = {"56Nx": 80, "DN": 80, "NEP25": 40, "normal": 80}
class ImageDataset(Dataset):
    def __init__(self, path_data, data, transform = None):
        super().__init__(data=data, transform=transform)
        self.path_data = path_data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def _transform(self, index: int):
        image_name = self.data[index]
        if os.path.exists(self.path_data + "/56Nx/" + image_name):
            image_microscope = Image.open(self.path_data + "/56Nx/" + image_name)
            image_microscope = image_microscope.reduce(4)
        elif os.path.exists(self.path_data + "/DN/" + image_name):
            image_microscope = Image.open(self.path_data + "/DN/" + image_name)
            image_microscope = image_microscope.reduce(4)
        elif os.path.exists(self.path_data + "/NEP25/" + image_name):
            image_microscope = Image.open(self.path_data + "/NEP25/" + image_name)
            image_microscope = image_microscope.reduce(2)
        elif os.path.exists(self.path_data + "/normal/" + image_name):
            image_microscope = Image.open(self.path_data + "/normal/" + image_name)
            image_microscope = image_microscope.reduce(4)
        array_microscope = np.array(image_microscope)
        array_microscope_alpha_channel = np.expand_dims(array_microscope[:, :, 2], axis=2)
        array_microscope = np.concatenate((array_microscope, array_microscope_alpha_channel), axis=2)
        array_microscope[:, :, 3][array_microscope[:, :, 2] > 0] = 255
        item = {"image": array_microscope, "image_name": image_name}
        item_trans = self.transform(item)
        return item_trans