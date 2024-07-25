import json
import logging
import os
from datetime import datetime
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from monai.visualize import plot_2d_or_3d_image

from source.dataloading.cross_validation_split import cross_validation_split
from source.dataloading.patch_extraction_kpis import patch_extraction
from source.dataloading.hover_patch_dataset import HoverPatchDataset
from source.models.glomNet import HoVerNet
from source.losses.custum_loss import HoVerNetLoss

from torch.optim.lr_scheduler import _LRScheduler

class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        print(new_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class Trainer_hover():
    def __init__(self, cfg: dict, transforms: Optional[dict] = None) -> None:
        date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        self.configs: dict = cfg
        self.transforms = transforms

        self.path_data = cfg["paths"]["path_data"]

        self.fold_num = cfg["hyperparameters"]["fold_num"]
        self.use_same_data_as = cfg["hyperparameters"]["use_same_data_as"]
        self.num_workers = cfg["hyperparameters"]["num_workers"]
        self.cross_validation_fold = cross_validation_split(fold_num=self.fold_num)

        self.in_channels = cfg["hyperparameters"]["in_channels"]
        self.out_channels = cfg["hyperparameters"]["out_channels"]
        self.hovermaps = cfg["hyperparameters"]["hovermaps"]
        self.freeze_encoder = cfg["hyperparameters"]["freeze_encoder"]
        self.freeze_decoder_bin = cfg["hyperparameters"]["freeze_decoder_bin"]
        self.continue_training = cfg["hyperparameters"]["continue_training"]
        self.initial_learning_rate = cfg["hyperparameters"]["initial_learning_rate"]
        self.weight_decay = cfg["hyperparameters"]["weight_decay"]
        self.nb_batch_per_epochs = cfg["hyperparameters"]["nb_batch_per_epochs"]
        self.nb_epochs = cfg["hyperparameters"]["nb_epochs"]
        self.batch_size = cfg["hyperparameters"]["batch_size"]
        # self.patch_size = (cfg["hyperparameters"]["patch_size"], cfg["hyperparameters"]["patch_size"])

        self.configs["training_info"] = {}
        self.configs["training_info"]["cross_validation"] = self.cross_validation_fold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configs["training_info"]["device"] = self.device

        if self.hovermaps:
            self.path_output = cfg["paths"]["path_output"] + "/runs_" + str(date) + "_fold_" + str(
                self.fold_num) + "_hover/"
        else:
            self.path_output = cfg["paths"]["path_output"] + "/runs_" + str(date) + "_fold_" + str(
                self.fold_num) + "_bin/"

        if not os.path.exists(self.path_output):
            os.makedirs(self.path_output)
        logging.basicConfig(filename=self.path_output + "/std.log",
                            format='%(asctime)s %(message)s',
                            filemode='w',
                            force=True)
        self.logger = logging.getLogger()
        self.logger.info(self.configs)

        self.data_train = None
        self.data_val = None
        self.train_ds = None
        self.val_ds = None
        self.train_loader = None
        self.val_loader = None

        self.model = None
        self.loss_function = None
        self.first_epoch = None
        self.writer = None

    def prepare_data(self):
        if self.use_same_data_as == "None":
            self.data_train = patch_extraction(self.path_data, data_list=self.cross_validation_fold["train"])
            self.data_val = patch_extraction(self.path_data, data_list=self.cross_validation_fold["val"])
            with open(self.path_output + "/data_train.json", "w") as json_file:
                json_file.write(json.dumps(self.data_train))
            with open(self.path_output + "/data_val.json", "w") as json_file:
                json_file.write(json.dumps(self.data_val))
        else:
            with open(self.use_same_data_as + "/data_train.json", "r") as json_file:
                self.data_train = json.load(json_file)
            with open(self.use_same_data_as + "/data_val.json", "r") as json_file:
                self.data_val = json.load(json_file)
        self.train_ds = HoverPatchDataset(self.path_data, self.data_train, hovermaps=self.hovermaps,
                                          transform=self.transforms["train"])
        self.val_ds = HoverPatchDataset(self.path_data, self.data_val, hovermaps=self.hovermaps,
                                        transform=self.transforms["val"])
        self.train_loader = DataLoader(self.train_ds, shuffle=False, batch_size=self.batch_size,
                                       num_workers=self.num_workers)
        self.val_loader = DataLoader(self.val_ds, shuffle=False, batch_size=self.batch_size,
                                     num_workers=self.num_workers)

    def prepare_training(self):
        self.model = HoVerNet(in_channels=4, out_channels=2, backbone="efficientnet-b7", pretrained=True,
                              hovermaps=self.hovermaps, freeze_encoder=self.freeze_encoder, freeze_decoder_bin=self.freeze_decoder_bin)
        self.model.to(self.device)
        self.loss_function = HoVerNetLoss(lambda_ce=0.8, lambda_dice=0.2, hovermaps=self.hovermaps)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                           lr=self.initial_learning_rate,
                                           weight_decay=self.weight_decay)
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        self.lr_scheduler = PolyLRScheduler(self.optimizer, self.initial_lr, self.num_epochs)
        if self.continue_training != "None":
            checkpoint = torch.load(self.continue_training)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.first_epoch = checkpoint['epoch'] + 1

        else:
            self.first_epoch = 0
        self.writer = SummaryWriter(self.path_output + "/tensorbord/")

    def train(self):
        best_epoch_loss_val = 1000
        best_epoch_loss_val_epoch = 0
        epoch_loss_values_train = list()
        epoch_loss_values_val = list()
        iter_train_loader = iter(self.train_loader)
        iter_val_loader = iter(self.val_loader)
        nb_iterator_train = 1
        nb_iterator_val = 1
        iterator_loss_train = 0
        iterator_loss_val = 0
        iterator_step_train = 0
        iterator_step_val = 0
        for epoch in range(self.first_epoch, self.nb_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{self.nb_epochs}")
            print("training")
            self.logger.info("-" * 10)
            self.logger.info(f"epoch {epoch + 1}/{self.nb_epochs}")
            self.logger.info("training")
            ##### Training part ########
            self.model.train()
            epoch_loss_train = 0
            step_train = 0
            for b in range(self.nb_batch_per_epochs):
                try:
                    train_data = next(iter_train_loader)
                except StopIteration:
                    iterator_loss_train /= iterator_step_train
                    print(f"iteration {nb_iterator_train} average training loss: {iterator_loss_train:.4f}")
                    self.logger.info(f"iteration {nb_iterator_train} average training loss: {iterator_loss_train:.4f}")
                    self.writer.add_scalar("iterator_train_loss", iterator_loss_train, nb_iterator_train)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                    }, self.path_output + "/iterator_train_model_segmentation2d_dict_epoch_" + str(
                        nb_iterator_train) + ".pth")
                    iterator_step_train = 0
                    iterator_loss_train = 0
                    print("new iterator train")
                    self.logger.info("new iterator train")
                    nb_iterator_train += 1
                    iter_train_loader = iter(self.train_loader)
                    train_data = next(iter_train_loader)
                step_train += 1
                iterator_step_train += 1
                if self.hovermaps:
                    train_images, train_label_bin, train_label_hover = train_data["image"].to(self.device), train_data[
                        "label_bin"].to(self.device), train_data["label_hover"].to(self.device)
                    train_labels = {
                        "label_bin": train_label_bin,
                        "label_hover": train_label_hover
                    }
                else:
                    train_images, train_label_bin = train_data["image"].to(self.device), train_data[
                        "label_bin"].to(self.device)
                    train_labels = {
                        "label_bin": train_label_bin
                    }
                train_images = train_images.float()
                self.optimizer.zero_grad()
                train_outputs = self.model(train_images)
                loss_train = self.loss_function(train_outputs, train_labels)
                loss_train.backward()
                self.optimizer.step()
                epoch_loss_train += loss_train.item()
                iterator_loss_train += loss_train.item()
                epoch_len_train = self.nb_batch_per_epochs
                print(f"{step_train}/{epoch_len_train}, train_loss: {loss_train.item():.4f}")
                self.logger.info(f"{step_train}/{epoch_len_train}, train_loss: {loss_train.item():.4f}")
            self.lr_scheduler.step(epoch)
            epoch_loss_train /= step_train
            epoch_loss_values_train.append(epoch_loss_train)
            print(f"epoch {epoch + 1} average training loss: {epoch_loss_train:.4f}")
            self.logger.info(f"epoch {epoch + 1} average training loss: {epoch_loss_train:.4f}")
            self.writer.add_scalar("train_loss", epoch_loss_train, epoch + 1)

            ######### Validation part ###########
            print("validation")
            self.logger.info("validation")
            epoch_loss_val = 0
            step_val = 0
            self.model.eval()
            with torch.no_grad():
                for b in range(int(self.nb_batch_per_epochs / 2)):
                    try:
                        val_data = next(iter_val_loader)
                    except StopIteration:
                        iterator_loss_val /= iterator_step_val
                        print(f"iteration {nb_iterator_val} average validation loss: {iterator_loss_val:.4f}")
                        self.logger.info(
                            f"iteration {nb_iterator_val} average validation loss: {iterator_loss_val:.4f}")
                        self.writer.add_scalar("iterator_val_loss", iterator_loss_val, nb_iterator_val)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()
                        }, self.path_output + "/iterator_val_model_segmentation2d_dict_epoch_" + str(
                            nb_iterator_val) + ".pth")
                        iterator_step_val = 0
                        iterator_loss_val = 0
                        print("new iterator val")
                        self.logger.info("new iterator val")
                        nb_iterator_val += 1
                        iter_val_loader = iter(self.val_loader)
                        val_data = next(iter_val_loader)
                    step_val += 1
                    iterator_step_val += 1
                    if self.hovermaps:
                        val_images, val_label_bin, val_label_hover = val_data["image"].to(self.device), val_data[
                            "label_bin"].to(self.device), val_data["label_hover"].to(self.device)
                        val_labels = {
                            "label_bin": val_label_bin,
                            "label_hover": val_label_hover
                        }
                    else:
                        val_images, val_label_bin, = val_data["image"].to(self.device), val_data[
                            "label_bin"].to(self.device)
                        val_labels = {
                            "label_bin": val_label_bin
                        }
                    val_images = val_images.float()
                    val_outputs = self.model(val_images)
                    loss_val = self.loss_function(val_outputs, val_labels)
                    epoch_loss_val += loss_val.item()
                    iterator_loss_val += loss_val.item()
                    epoch_len_val = int(self.nb_batch_per_epochs / 2)
                    print(f"{step_val}/{epoch_len_val}, validation loss: {loss_val.item():.4f}")
                    self.logger.info(f"{step_val}/{epoch_len_val}, validation loss: {loss_val.item():.4f}")
                epoch_loss_val /= step_val
                epoch_loss_values_val.append(epoch_loss_val)
                print(f"epoch {epoch + 1} average validation loss: {epoch_loss_val:.4f}")
                self.logger.info(f"epoch {epoch + 1} average validation loss: {epoch_loss_val:.4f}")
                if epoch_loss_val < best_epoch_loss_val:
                    best_epoch_loss_val = epoch_loss_val
                    best_epoch_loss_val_epoch = epoch + 1
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                    }, self.path_output + "/best_metric_model_segmentation2d_dict_epoch_" + str(
                        best_epoch_loss_val_epoch) + ".pth")
                    print("saved new best metric model")
                    self.logger.info("saved new best metric model")
                if epoch+1 == 50 or epoch+1 == 100:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                    }, self.path_output + "/model_segmentation2d_dict_epoch_" + str(
                        epoch+1 ) + ".pth")

                print(
                    "current epoch: {} current loss val: {:.4f} best loss val: {:.4f} at epoch {}".format(
                        epoch + 1, epoch_loss_val, best_epoch_loss_val, best_epoch_loss_val_epoch
                    )
                )
                self.logger.info(
                    "current epoch: {} current loss val: {:.4f} best loss val: {:.4f} at epoch {}".format(
                        epoch + 1, epoch_loss_val, best_epoch_loss_val, best_epoch_loss_val_epoch
                    )
                )
                self.writer.add_scalar("val_mean_dice", epoch_loss_val, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, self.writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels["label_bin"], epoch + 1, self.writer, index=0, tag="label_bin")
                plot_2d_or_3d_image(val_outputs["pred_bin"][:, 1, :, :], epoch + 1, self.writer, index=0,
                                    tag="pred_bin")
            #
        print(f"train completed, best validation loss: {best_epoch_loss_val:.4f} at epoch: {best_epoch_loss_val_epoch}")
        self.logger.info(
            f"train completed, best validation loss: {best_epoch_loss_val:.4f} at epoch: {best_epoch_loss_val_epoch}")
        print(f"number of iteration on the all dataset (train): {nb_iterator_train}")
        print(f"number of iteration on the all dataset (val): {nb_iterator_val}")
        self.logger.info(f"number of iteration on the all dataset (train): {nb_iterator_train}")
        self.logger.info(f"number of iteration on the all dataset (val): {nb_iterator_val}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, self.path_output + "/best_metric_model_segmentation2d_dict_epoch_" + str(epoch + 1) + ".pth")
        logs = self.logger.handlers[:]
        for log in logs:
            self.logger.removeHandler(log)
            log.close()

        self.writer.close()
