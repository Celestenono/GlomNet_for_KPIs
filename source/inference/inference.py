import logging
import os
from datetime import datetime
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
from typing import Optional

import torch
from torch.utils.data import DataLoader

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AsDiscrete
)
from monai.data import decollate_batch

from source.dataloading.cross_validation_split import cross_validation_split
from source.dataloading.image_dataset import ImageDataset
from source.models.glomNet import HoVerNet
from source.utilities.postprocessing_hovermaps import postprocessing_hovermaps


class Inference():
    def __init__(self, cfg: dict, transforms: Optional[dict] = None) -> None:
        date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        self.configs: dict = cfg
        self.transforms = transforms

        self.path_data = cfg["paths"]["path_data"]
        self.path_model = cfg["paths"]["path_model"]

        self.fold_num = cfg["hyperparameters"]["fold_num"]
        self.dataset_type = cfg["hyperparameters"]["dataset_type"]
        self.num_workers = cfg["hyperparameters"]["num_workers"]
        self.cross_validation_fold = cross_validation_split(fold_num=self.fold_num)

        self.in_channels = cfg["hyperparameters"]["in_channels"]
        self.out_channels = cfg["hyperparameters"]["out_channels"]
        self.hovermaps = cfg["hyperparameters"]["hovermaps"]
        self.batch_size = cfg["hyperparameters"]["batch_size"]
        self.patch_size = (cfg["hyperparameters"]["patch_size"], cfg["hyperparameters"]["patch_size"])

        self.configs["inference_info"] = {}
        self.configs["inference_info"]["cross_validation"] = self.cross_validation_fold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configs["inference_info"]["device"] = self.device

        if self.hovermaps:
            self.path_output = cfg["paths"]["path_output"] + "/inf_" + str(date) + "_fold_" + str(
                self.fold_num) + "_" + str(self.dataset_type) + "_hover/"
        else:
            self.path_output = cfg["paths"]["path_output"] + "/inf_" + str(date) + "_fold_" + str(
                self.fold_num) + "_" + str(self.dataset_type) + "_bin/"

        if not os.path.exists(self.path_output):
            os.makedirs(self.path_output)
        logging.basicConfig(filename=self.path_output + "/std.log",
                            format='%(asctime)s %(message)s',
                            filemode='w',
                            force=True)
        self.logger = logging.getLogger()
        self.logger.info(self.configs)

        self.val_ds = None
        self.test_ds = None
        self.val_loader = None
        self.test_loader = None

        self.model = None
        self.model_checkpoint = None

    def prepare_data(self):
        self.val_ds = ImageDataset(self.path_data, self.cross_validation_fold["val"], transform=self.transforms["val"])
        self.test_ds = ImageDataset(self.path_data, self.cross_validation_fold["test"], transform=self.transforms["val"])
        self.val_loader = DataLoader(self.val_ds, batch_size=1, num_workers=self.num_workers)
        self.test_loader = DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers)

    def load_model(self):
        self.model = HoVerNet(in_channels=4, out_channels=2, backbone="efficientnet-b7", hovermaps=self.hovermaps)
        self.model.to(self.device)
        self.model_checkpoint = torch.load(self.path_model, map_location=torch.device(self.device))
        self.model.load_state_dict(self.model_checkpoint["model_state_dict"])
        self.model.eval()

    def __sliding_window_inference(self, inf_image):
        post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])
        inf_image = inf_image.float()
        inf_outputs = sliding_window_inference(inf_image, self.patch_size, self.batch_size, self.model, sw_device=self.device)

        inf_outputs_bin = inf_outputs["pred_bin"]
        inf_outputs_bin = [post_trans(i) for i in decollate_batch(inf_outputs_bin)][0]
        inf_results_bin = inf_outputs_bin.clone().detach().to('cpu').numpy()
        inf_results_bin = inf_results_bin[0]

        if self.hovermaps:
            inf_outputs_hover = inf_outputs["pred_hover"]
            inf_results_hover = inf_outputs_hover.clone().detach().to('cpu').numpy()
            inf_results_hover = inf_results_hover[0]
            inf_results_ho = inf_results_hover[0]
            inf_results_ver = inf_results_hover[1]

            inf_results = {
                "inf_results_bin": inf_results_bin,
                "inf_results_ho": inf_results_ho,
                "inf_results_ver": inf_results_ver
            }
        else:
            inf_results = {
                "inf_results_bin": inf_results_bin,
            }
        return inf_results

    def infer(self):
        if self.dataset_type == "test":
            loader = self.test_loader
        elif self.dataset_type == "val":
            loader = self.val_loader
        for test_data in loader:
            test_image, test_image_name = test_data["image"], test_data["image_name"]
            test_image_name = test_image_name[0][:-4]
            print(test_image_name)
            with torch.no_grad():
                test_results = self.__sliding_window_inference(test_image)

                test_results_bin = test_results["inf_results_bin"]
                np.save(self.path_output + "/" + test_image_name + "_bin.npy", test_results_bin)
                test_results_bin_image = Image.fromarray(test_results_bin.astype(np.uint8))
                test_results_bin_image.save(self.path_output + "/" + test_image_name + "_bin.png")
                if self.hovermaps:
                    test_results_ho = test_results["inf_results_ho"]
                    test_results_ver = test_results["inf_results_ver"]
                    np.save(self.path_output + "/" + test_image_name + "_ho.npy", test_results_ho)
                    np.save(self.path_output + "/" + test_image_name + "_ver.npy", test_results_ver)

                    test_results_inst = postprocessing_hovermaps(test_results)
                    test_results_inst_image = Image.fromarray(test_results_inst.astype(np.uint8))
                    test_results_inst_image.save(self.path_output + "/" + test_image_name + "_inst.png")


import yaml

from monai.transforms import (
    EnsureChannelFirstd,
    ScaleIntensityd,
    Compose,
)


if __name__ == "__main__":
    with open("inf_config_hover.yml", "r") as ymlfile:
    # with open("test_inf_config_hover.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    validation_transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image"], channel_dim=2),
            ScaleIntensityd(keys=["image"]),
        ]
    )

    transforms = {}
    transforms["train"] = None
    transforms["val"] = validation_transforms

    infer = Inference(cfg, transforms=transforms)
    infer.prepare_data()
    infer.load_model()
    infer.infer()