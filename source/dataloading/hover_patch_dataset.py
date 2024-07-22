import matplotlib.pyplot as plt
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import random
from typing import Callable, Optional

from monai.data.dataset import Dataset

from source.utilities.compute_hovermaps import compute_hovermaps
from source.dataloading.cross_validation_split import cross_validation_split
import json
from scipy.ndimage import label
from openslide import OpenSlide



class HoverPatchDataset(Dataset):
    def __init__(self, path_data: str, data: dict, hovermaps: bool = True,
                 transform: Optional[Callable] = None) -> None:
        super().__init__(data=data, transform=transform)
        self.path_data = path_data
        self.patch_size = (1024, 1024)
        # self.patch_size = data["infos"]["patch_size"]
        self.transform = transform
        self.hovermaps = hovermaps
        random.seed(0)

    def __len__(self):
        return len(self.data["all_shuffle"])
        # return 14

    def _transform(self, index: int):
        patch_id = int(index)
        patch_info = self.data["all_shuffle"][patch_id]
        image_filename = patch_info["image_name"] + "wsi.tiff"
        mask_filename = patch_info["image_name"] + "mask.tiff"
        dir = patch_info["dir"]
        magnification = patch_info["magnification"]
        patch_size_patch = patch_info["patch_size"]
        x = patch_info["x"]
        y = patch_info["y"]
        pad_size_x = patch_size_patch[0]
        pad_size_y = patch_size_patch[1]
        flip_h = random.randint(0, 1)
        flip_v = random.randint(0, 1)
        # if True:
        if os.path.exists(self.path_data + "/" + dir + "/" + image_filename) and os.path.exists(self.path_data + "/" + dir + "/" + mask_filename):
            image_openslide = OpenSlide(self.path_data + "/" + dir + "/" + image_filename)
            mask_openslide = OpenSlide(self.path_data + "/" + dir + "/" + mask_filename)

            x_big = max(0, x-pad_size_x)
            y_big = max(0, y-pad_size_y)
            pad_size_x = x-x_big
            pad_size_y = y - y_big
            big_patch_size = [patch_size_patch[0]+(2*pad_size_x), patch_size_patch[0]+(2*pad_size_y)]
            big_patch_image = image_openslide.read_region((x_big, y_big), 0, big_patch_size)
            big_patch_mask = mask_openslide.read_region((x_big, y_big), 0, big_patch_size)

            if magnification == 80:
                big_patch_image = big_patch_image.reduce(2)
                big_patch_mask = big_patch_mask.reduce(2)
                pad_size_x = pad_size_x // 2
                pad_size_y = pad_size_y // 2
            big_patch_image_array = np.array(big_patch_image)
            big_patch_image.close()
            big_patch_mask_array = np.array(big_patch_mask)[:, :, 0]
            big_patch_mask_array, _ = label(big_patch_mask_array)
            big_patch_mask.close()

            if self.hovermaps:
                array_hoverMaps = compute_hovermaps(big_patch_mask_array)
                patch_array_hoverMaps = array_hoverMaps[:, pad_size_x:pad_size_x + self.patch_size[0], pad_size_y:pad_size_y + self.patch_size[1]]
            big_patch_mask_array[big_patch_mask_array > 0] = 1
            patch_array_microscope = big_patch_image_array[pad_size_x:pad_size_x + self.patch_size[0], pad_size_y:pad_size_y + self.patch_size[1]]
            patch_array_mask = big_patch_mask_array[pad_size_x:pad_size_x + self.patch_size[0], pad_size_y:pad_size_y + self.patch_size[1]]

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
                    "label_hover": patch_array_hoverMaps, "image_name": image_filename}
        else:
            item = {"image": patch_array_microscope, "label_bin": np.array([patch_array_mask]),
                    "image_name": image_filename}
        try:
            item_trans = self.transform(item)
        except:
            print(image_filename, x, y)
        return item_trans

from monai.transforms import (
    EnsureChannelFirstd,
    ScaleIntensityd,
    Compose,
    RandGaussianNoised,
    OneOf,
    MedianSmoothd,
    RandGaussianSmoothd,
    Resized
)
from torch.utils.data import DataLoader


if __name__ == "__main__":
    raw_training_directory = "/Users/nmoreau/Documents/KPIs_challenge/KPIs24 Training Data/Task2_WSI_level/"
    with open(raw_training_directory + "/data_train.json", "r") as json_file:
        data_train = json.load(json_file)
    with open(raw_training_directory + "/data_val.json", "r") as json_file:
        data_val = json.load(json_file)
    train_transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image"], channel_dim=2),
            ScaleIntensityd(keys=["image"]),
            # OneOf(
            #     transforms=[
            #         RandGaussianSmoothd(keys=["image"], sigma_x=(0.1, 1.1), sigma_y=(0.1, 1.1), prob=1.0),
            #         MedianSmoothd(keys=["image"], radius=1),
            #         RandGaussianNoised(keys=["image"], prob=1.0, std=0.05),
            #     ]
            # ),
            Resized(keys=["image", "label_bin", "label_hover"], spatial_size=(512, 512))

        ]
    )
    validation_transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image"], channel_dim=2),
            ScaleIntensityd(keys=["image"]),
        ]
    )
    train_ds = HoverPatchDataset(raw_training_directory, data_train, hovermaps=True, transform=train_transforms)
    train_loader = DataLoader(train_ds, shuffle=False, batch_size=1, num_workers=1)
    train_loader_iterator = iter(train_loader)
    first_sample = next(train_loader_iterator)
    raw_image = (first_sample["image"][0, ...].numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    mask_array = first_sample["label_bin"][0, 0, ...].numpy()
    hover_array = first_sample["label_hover"][0, 0, ...].numpy()
    plt.imshow(raw_image)
    plt.show()
    plt.imshow(mask_array, cmap="prism", vmax=100, alpha=0.7 * (mask_array > 0))
    plt.show()
    plt.imshow(first_sample["label_hover"][0, 0, ...], vmin=-1, vmax=1)
    plt.show()
    plt.imshow(first_sample["label_hover"][0, 1, ...], vmin=-1, vmax=1)
    plt.show()
    first_sample = next(train_loader_iterator)
    raw_image = (first_sample["image"][0, ...].numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    mask_array = first_sample["label_bin"][0, 0, ...].numpy()
    hover_array = first_sample["label_hover"][0, 0, ...].numpy()
    plt.imshow(raw_image)
    plt.show()
    plt.imshow(mask_array, cmap="prism", vmax=100, alpha=0.7 * (mask_array > 0))
    plt.show()
    plt.imshow(first_sample["label_hover"][0, 0, ...], vmin=-1, vmax=1)
    plt.show()
    plt.imshow(first_sample["label_hover"][0, 1, ...], vmin=-1, vmax=1)
    plt.show()
    first_sample = next(train_loader_iterator)
    raw_image = (first_sample["image"][0, ...].numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    mask_array = first_sample["label_bin"][0, 0, ...].numpy()
    hover_array = first_sample["label_hover"][0, 0, ...].numpy()
    plt.imshow(raw_image)
    plt.show()
    plt.imshow(mask_array, cmap="prism", vmax=100, alpha=0.7 * (mask_array > 0))
    plt.show()
    plt.imshow(first_sample["label_hover"][0, 0, ...], vmin=-1, vmax=1)
    plt.show()
    plt.imshow(first_sample["label_hover"][0, 1, ...], vmin=-1, vmax=1)
    plt.show()
