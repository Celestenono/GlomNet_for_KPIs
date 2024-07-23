import os

import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np

from monai.data.dataset import Dataset
from openslide import OpenSlide
import math

MAGNIFICATIONS = {"56Nx": 80, "DN": 80, "NEP25": 40, "normal": 80}

def read_image_openslide(path_image, reduce_factor, patch_size = (10000,10000)):
    image_openslide = OpenSlide(path_image)
    image_dimension = image_openslide.dimensions
    array_microscope = np.full((math.ceil(image_dimension[1]/reduce_factor), math.ceil(image_dimension[0]/reduce_factor), 4), 0, dtype='uint8')
    for x in range(0, image_dimension[0], patch_size[0]):
        patch_size_x = min(image_dimension[0] - x, patch_size[0])
        for y in range(0, image_dimension[1], patch_size[1]):
            patch_size_y = min(image_dimension[1]-y, patch_size[1])
            patch_image_pil = image_openslide.read_region((x, y), 0, (patch_size_x, patch_size_y))
            patch_image_pil = patch_image_pil.reduce(reduce_factor)
            patch_image_array = np.array(patch_image_pil)
            patch_image_size = patch_image_array.shape
            new_x = x//reduce_factor
            new_y = y//reduce_factor
            array_microscope[new_y:new_y+patch_image_size[0], new_x:new_x+patch_image_size[1], :] = patch_image_array
    return array_microscope
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
            # image_microscope = Image.open(self.path_data + "/56Nx/" + image_name)
            # image_microscope = image_microscope.reduce(4)
            array_microscope = read_image_openslide(self.path_data + "/56Nx/" + image_name, reduce_factor=4)
        elif os.path.exists(self.path_data + "/DN/" + image_name):
            # image_microscope = Image.open(self.path_data + "/DN/" + image_name)
            # image_microscope = image_microscope.reduce(4)
            array_microscope = read_image_openslide(self.path_data + "/DN/" + image_name, reduce_factor=4)
        elif os.path.exists(self.path_data + "/NEP25/" + image_name):
            # image_microscope = Image.open(self.path_data + "/NEP25/" + image_name)
            # image_microscope = image_microscope.reduce(2)
            array_microscope = read_image_openslide(self.path_data + "/NEP25/" + image_name, reduce_factor=2)
        elif os.path.exists(self.path_data + "/normal/" + image_name):
            # image_microscope = Image.open(self.path_data + "/normal/" + image_name)
            # image_microscope = image_microscope.reduce(4)
            array_microscope = read_image_openslide(self.path_data + "/normal/" + image_name, reduce_factor=4)
        else:
            raise Exception("No file with this name:"+image_name)
        # array_microscope = np.array(image_microscope)
        # array_microscope_alpha_channel = np.expand_dims(array_microscope[:, :, 2], axis=2)
        # array_microscope = np.concatenate((array_microscope, array_microscope_alpha_channel), axis=2)
        # array_microscope[:, :, 3][array_microscope[:, :, 2] > 0] = 255
        item = {"image": array_microscope, "image_name": image_name}
        item_trans = self.transform(item)
        return item_trans


if __name__ == "__main__":
    path_image = "/Users/nmoreau/Documents/KPIs_challenge/Validation_bis/56NX/img/12-174_wsi.tiff"
    read_image_openslide(path_image, reduce_factor=4)