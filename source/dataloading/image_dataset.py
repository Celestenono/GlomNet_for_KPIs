import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np

from monai.data.dataset import Dataset

class ImageDataset(Dataset):
    def __init__(self, path_data, data, transform = None):
        super().__init__(data=data, transform=transform)
        self.path_data = path_data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def _transform(self, index: int):
        image_name = self.data[index]
        if os.path.exists(self.path_data + "/images/" + image_name):
            image_microscope = Image.open(self.path_data + "/images/" + image_name)
            array_microscope = np.array(image_microscope)
        elif os.path.exists(self.path_data + "/images/" + image_name):
            image_microscope = Image.open(self.path_data + "/images/" + image_name)
            array_microscope = np.array(image_microscope)
        item = {"image": array_microscope, "image_name": image_name}
        item_trans = self.transform(item)
        return item_trans