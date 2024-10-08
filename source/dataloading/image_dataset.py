import os

import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np

from monai.data.dataset import Dataset
# from openslide import OpenSlide
import math
import tifffile
import scipy.ndimage as ndi

MAGNIFICATIONS = {"56Nx": 80, "DN": 80, "NEP25": 40, "normal": 80}

# def read_image_openslide(path_image, reduce_factor, patch_size = (10000,10000)):
#     image_openslide = OpenSlide(path_image)
#     image_dimension = image_openslide.dimensions
#     array_microscope = np.full((math.ceil(image_dimension[1]/reduce_factor), math.ceil(image_dimension[0]/reduce_factor), 4), 0, dtype='uint8')
#     for x in range(0, image_dimension[0], patch_size[0]):
#         patch_size_x = min(image_dimension[0] - x, patch_size[0])
#         for y in range(0, image_dimension[1], patch_size[1]):
#             patch_size_y = min(image_dimension[1]-y, patch_size[1])
#             patch_image_pil = image_openslide.read_region((x, y), 0, (patch_size_x, patch_size_y))
#             patch_image_pil = patch_image_pil.reduce(reduce_factor)
#             patch_image_array = np.array(patch_image_pil)
#             patch_image_size = patch_image_array.shape
#             new_x = x//reduce_factor
#             new_y = y//reduce_factor
#             array_microscope[new_y:new_y+patch_image_size[0], new_x:new_x+patch_image_size[1], :] = patch_image_array
#     return array_microscope
class ImageDataset(Dataset):
    def __init__(self, list_data, transform = None):
        super().__init__(data=list_data, transform=transform)
        self.list_data=list_data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def _transform(self, index: int):
        image_path = self.list_data[index]
        if 'NEP25' in image_path:
            lv = 1
        else:
            lv = 2
        if os.path.exists(image_path):
            original_tiff = tifffile.imread(image_path, key=0)
            tiff_x20 = ndi.zoom(original_tiff, (1 / lv, 1 / lv, 1), order=1)
            tiff_x20_shape = tiff_x20.shape
        else:
            raise Exception("No file: "+image_path)
        array_microscope = ndi.zoom(tiff_x20, (0.5, 0.5, 1), order=1)
        # array_microscope = np.array(image_microscope)
        array_microscope_alpha_channel = np.expand_dims(array_microscope[:, :, 2], axis=2)
        array_microscope = np.concatenate((array_microscope, array_microscope_alpha_channel), axis=2)
        array_microscope[:, :, 3][array_microscope[:, :, 2] > 0] = 255
        item = {"image": array_microscope, "image_path": image_path, "image_x20_shape": tiff_x20_shape}
        item_trans = self.transform(item)
        return item_trans


# if __name__ == "__main__":
#     path_image = "/Users/nmoreau/Documents/KPIs_challenge/Validation_bis/56NX/img/12-174_wsi.tiff"
#     read_image_openslide(path_image, reduce_factor=4)