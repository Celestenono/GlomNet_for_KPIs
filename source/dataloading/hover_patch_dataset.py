import matplotlib.pyplot as plt
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import random
from typing import Callable, Optional

from monai.data.dataset import Dataset

from source.utilities.compute_hovermaps import compute_hovermaps


class HoverPatchDataset(Dataset):
    def __init__(self, path_data: str, data: dict, hovermaps: bool = True,
                 transform: Optional[Callable] = None) -> None:
        super().__init__(data=data, transform=transform)
        self.path_data = path_data
        self.patch_size = data["infos"]["patch_size"]
        self.transform = transform
        self.hovermaps = hovermaps
        random.seed(0)

    def __len__(self):
        return len(self.data["all_shuffle"])
        # return 14

    def _transform(self, index: int):
        patch_id = int(index)
        patch_info = self.data["all_shuffle"][patch_id]
        image_name = patch_info["image_name"]
        x = patch_info["x"]
        y = patch_info["y"]
        flip_h = random.randint(0, 1)
        flip_v = random.randint(0, 1)
        if os.path.exists(self.path_data + "/masks/" + image_name) and os.path.exists(self.path_data + "/images/"
                                                                                      + image_name):
            image_microscope = Image.open(self.path_data + "/images/" + image_name)
            array_microscope = np.array(image_microscope)
            image_microscope.close()
            image_mask = Image.open(self.path_data + "/masks/" + image_name)
            array_mask = np.array(image_mask)
            image_mask.close()
            if self.hovermaps:
                array_hoverMaps = compute_hovermaps(array_mask)
                patch_array_hoverMaps = array_hoverMaps[:, x:x + self.patch_size[0], y:y + self.patch_size[1]]
            array_mask[array_mask > 0] = 1
            patch_array_microscope = array_microscope[x:x + self.patch_size[0], y:y + self.patch_size[1]]
            patch_array_mask = array_mask[x:x + self.patch_size[0], y:y + self.patch_size[1]]
            if flip_h:
                patch_array_microscope = np.fliplr(patch_array_microscope)
                patch_array_mask = np.fliplr(patch_array_mask)
                if self.hovermaps:
                    patch_array_hoverMaps[0, ...] = np.fliplr(patch_array_hoverMaps[0, ...])
                    patch_array_hoverMaps[1, ...] = np.fliplr(patch_array_hoverMaps[1, ...])
                    patch_array_hoverMaps[0, ...] = -patch_array_hoverMaps[0, ...]
                    patch_array_hoverMaps[patch_array_hoverMaps == 0] = 0
            if flip_v:
                patch_array_microscope = np.flipud(patch_array_microscope)
                patch_array_mask = np.flipud(patch_array_mask)
                if self.hovermaps:
                    patch_array_hoverMaps[0, ...] = np.flipud(patch_array_hoverMaps[0, ...])
                    patch_array_hoverMaps[1, ...] = np.flipud(patch_array_hoverMaps[1, ...])
                    patch_array_hoverMaps[1, ...] = -patch_array_hoverMaps[1, ...]
                    patch_array_hoverMaps[patch_array_hoverMaps == 0] = 0
        patch_array_microscope = patch_array_microscope.astype(int)
        patch_array_mask = patch_array_mask.astype(int)
        if self.hovermaps:
            patch_array_hoverMaps = patch_array_hoverMaps.astype("float32")
            item = {"image": patch_array_microscope, "label_bin": np.array([patch_array_mask]),
                    "label_hover": patch_array_hoverMaps, "image_name": image_name}
        else:
            item = {"image": patch_array_microscope, "label_bin": np.array([patch_array_mask]),
                    "image_name": image_name}
        item_trans = self.transform(item)
        return item_trans
