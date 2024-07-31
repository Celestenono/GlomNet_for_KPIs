# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import cv2
import logging
import os
import sys
from glob import glob

import torch
from PIL import Image

from monai import config
from monai.data import decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
from matplotlib import cm
import matplotlib.pyplot as plt
import tifffile
import scipy.ndimage as ndi
import numpy as np
import pandas as pd

from source.dataloading.image_dataset import ImageDataset

from monai.transforms import (
    EnsureChannelFirstd,
    ScaleIntensityd,
    Compose,
)

from source.models.glomNet import HoVerNet

from pathlib import Path


def calculate_contour_iou(contour1, contour2, image_shape):
    x1_min = contour1[:,:,0].min()
    x1_max = contour1[:,:,0].max()
    y1_min = contour1[:,:,1].min()
    y1_max = contour1[:,:,1].max()

    x2_min = contour2[:,:,0].min()
    x2_max = contour2[:,:,0].max()
    y2_min = contour2[:,:,1].min()
    y2_max = contour2[:,:,1].max()

    if x1_max < x2_min or x2_max < x1_min:
        return 0
    if y1_max < y2_min or y2_max < y1_min:
        return 0

    'crop'
    x_min = np.min([x1_min, x2_min]) - 10
    y_min = np.min([y1_min, y2_min]) - 10
    x_max = np.max([x1_max, x2_max]) + 10
    y_max = np.max([y1_max, y2_max]) + 10

    contour1[:,:,0] = contour1[:,:,0] - x_min
    contour1[:,:,1] = contour1[:,:,1] - y_min

    contour2[:,:,0] = contour2[:,:,0] - x_min
    contour2[:,:,1] = contour2[:,:,1] - y_min
    image_shape = (y_max - y_min, x_max - x_min)

    mask1 = np.zeros(image_shape, dtype=np.uint8)
    mask2 = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask1, [contour1], -1, 1, -1)
    cv2.drawContours(mask2, [contour2], -1, 1, -1)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0


def save_validate(val_images, val_labels, val_outputs, output_dir, images, cnt):
    for i in range(val_images.shape[0]):
        folder_list = os.path.dirname(images[cnt+i]).split('/')
        save_folder = os.path.join(output_dir, folder_list[-3], folder_list[-2])

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        now_image = val_images[i].permute([2,1,0]).detach().cpu().numpy()
        now_label = val_labels[i][0].permute([1,0]).detach().cpu().numpy()
        now_pred = val_outputs[i][0].permute([1,0]).detach().cpu().numpy()
        name = os.path.basename(images[cnt+i])
        plt.imsave(os.path.join(save_folder, 'val_%s_img.png' % (name)), now_image)
        plt.imsave(os.path.join(save_folder, 'val_%s_lbl.png' % (name)), now_label, cmap = cm.gray)
        plt.imsave(os.path.join(save_folder, 'val_%s_pred.png' % (name)), now_pred, cmap = cm.gray)

    cnt += val_images.shape[0]
    return cnt

def calculate_f1(precision, recall):
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # Convert NaNs to zero if precision and recall are both zero
    return f1_scores


def dice_coefficient(mask1, mask2):
    # Convert masks to boolean arrays
    mask1 = np.asarray(mask1).astype(np.int8)
    mask2 = np.asarray(mask2).astype(np.int8)

    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2)
    mask1_sum = np.sum(mask1)
    mask2_sum = np.sum(mask2)
    intersection_sum = np.sum(intersection)

    # Compute Dice coefficient
    if mask1_sum + mask2_sum == 0:  # To handle division by zero if both masks are empty
        return 1.0
    else:
        return 2 * intersection_sum / (mask1_sum + mask2_sum)


def calculate_metrics_ap50(pred_contours_list, gt_contours_list, image_shape, iou_thresholds=[0.5]):
    # Initialize lists to hold precision and recall values for each threshold
    precision_scores = []
    recall_scores = []

    for threshold in iou_thresholds:
        tp = 0
        fp = 0
        fn = 0

        # Calculate matches for predictions
        for pred_contours in pred_contours_list:
            match_found = False
            for gt_contours in gt_contours_list:
                if calculate_contour_iou(pred_contours, gt_contours, image_shape) >= threshold:
                    tp += 1
                    match_found = True
                    break
            if not match_found:
                fp += 1

        # Calculate false negatives
        for gt_contours in gt_contours_list:
            if not any(calculate_contour_iou(pred_contours, gt_contours, image_shape) >= threshold for pred_contours in pred_contours_list):
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Compute F1 scores
    f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision_scores, recall_scores)]
    return precision_scores, recall_scores, f1_scores


def sodelete(wsi, min_size):
    """
    Remove objects smaller than min_size from binary segmentation image.

    Args:
    img (numpy.ndarray): Binary image where objects are 255 and background is 0.
    min_size (int): Minimum size of the object to keep.

    Returns:
    numpy.ndarray: Image with small objects removed.
    """
    # Find all connected components (using 8-connectivity, as default)
    _, binary = cv2.threshold(wsi* 255, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8, cv2.CV_32S)

    # Create an output image that will store the filtered objects
    # output = np.zeros_like(wsi, dtype=np.uint8)
    output = np.zeros_like(wsi)

    # Loop through all found components
    for i in range(1, num_labels):  # start from 1 to ignore the background
        size = stats[i, cv2.CC_STAT_AREA]

        # If the size of the component is larger than the threshold, copy it to output
        if size >= min_size:
            output[labels == i] = 1.

    return output


def calculate_ap50(precisions, recalls):
    # Ensure that the arrays are sorted by recall
    sort_order = np.argsort(recalls)
    precisions = np.array(precisions)[sort_order]
    recalls = np.array(recalls)[sort_order]

    # Pad precisions array to include the precision at recall zero
    precisions = np.concatenate(([0], precisions))
    recalls = np.concatenate(([0], recalls))

    # Calculate the differences in recall to use as weights in weighted average
    recall_diff = np.diff(recalls)

    # Compute the average precision
    ap50 = np.sum(precisions[:-1] * recall_diff)
    return ap50

def inference(inf_image, patch_size, batch_size, model, device, post_trans, hovermaps=True):
    inf_image = inf_image.float()
    inf_outputs = sliding_window_inference(inf_image, patch_size, batch_size, model, sw_device=device)

    inf_outputs_bin = inf_outputs["pred_bin"]
    inf_outputs_bin = [post_trans(i) for i in decollate_batch(inf_outputs_bin)][0]
    inf_results_bin = inf_outputs_bin.clone().detach().to('cpu').numpy()
    inf_results_bin = inf_results_bin[0]

    if hovermaps:
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


def main(inputdir, path_model, output_dir, df):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    image = []
    seg = []
    types = glob(os.path.join(inputdir, '*'))
    for type in types:
        now_imgs = glob(os.path.join(type, 'img', '*.tiff'))
        image.extend(now_imgs)
        now_lbls = glob(os.path.join(type, 'mask', '*mask.tiff'))
        seg.extend(now_lbls)

    images = sorted(image)
    segs = sorted(seg)

    print('total image: %d' % (len(images)))

    imtrans = Compose(
        [
            EnsureChannelFirstd(keys=["image"], channel_dim=2),
            ScaleIntensityd(keys=["image"]),
        ]
    )
    test_ds = ImageDataset(images, transform=imtrans)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())
    post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HoVerNet(in_channels=4, out_channels=2, backbone="efficientnet-b7", hovermaps=True)

    model.to(device)
    model_checkpoint = torch.load(path_model, map_location=torch.device(device))
    model.load_state_dict(model_checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        for test_data in test_loader:
            test_image, test_image_path, test_image_shape = test_data["image"], test_data["image_path"], test_data["image_x20_shape"]
            print(test_image_path)
            test_results = inference(test_image, patch_size=(512, 512), batch_size=1, model=model, device=device, post_trans=post_trans)

            wsi_prediction = test_results["inf_results_bin"]
            wsi_prediction = Image.fromarray(wsi_prediction.astype(np.uint8))
            wsi_prediction = wsi_prediction.resize((test_image_shape[1], test_image_shape[0]))
            wsi_prediction = np.array(wsi_prediction)
            wsi_prediction[wsi_prediction < 1] = 0
            wsi_prediction[wsi_prediction != 0] = 1
            sm = 10000
            wsi_prediction_sm = sodelete(wsi_prediction, sm)


            preds_root = test_image_path[0].replace(inputdir, output_dir).replace("_wsi.tiff", "_mask.tiff").replace(
                "/img/", "/")
            p = Path(preds_root)
            if not os.path.exists(p.parent):
                os.makedirs(p.parent)
            wsi_prediction_sm = Image.fromarray(wsi_prediction_sm.astype(np.uint8))
            wsi_prediction_sm.save(preds_root)
            # plt.imsave(preds_root, wsi_prediction_sm, cmap=cm.gray)


    wsi_F1_50 = []
    wsi_AP50 = []
    wsi_dice = []
    for img, seg in zip(images, segs):

        case_name = os.path.basename(img)
        print(case_name)

        pred = img.replace(inputdir, output_dir).replace("_wsi.tiff", "_mask.tiff").replace("/img/", "/")

        if 'NEP25' in img:
            lv = 1
        else:
            lv = 2

        # img_tiff = tifffile.imread(img, key=0)
        # img_tiff_X20 = ndi.zoom(img_tiff, (1 / lv, 1 / lv, 1), order=1)
        # tiff_X20_shape = img_tiff_X20.shape

        mask_tiff = tifffile.imread(seg, key=0)
        mask_tiff_X20 = ndi.zoom(mask_tiff, (1 / lv, 1 / lv), order=1)
        mask_tiff_X20[mask_tiff_X20 < 1] = 0
        mask_tiff_X20[mask_tiff_X20 != 0] = 1
        tiff_X20_shape = mask_tiff_X20.shape

        wsi_prediction = tifffile.imread(pred, key=0)

        'f1'
        ret, binary = cv2.threshold(wsi_prediction, 0, 255, cv2.THRESH_BINARY_INV)
        preds_contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ret, binary = cv2.threshold(mask_tiff_X20, 0, 255, cv2.THRESH_BINARY_INV)
        masks_contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        precision_scores, recall_scores, f1_scores_50 = calculate_metrics_ap50(preds_contours[1:], masks_contours[1:],
                                                                               (tiff_X20_shape[0], tiff_X20_shape[1]))
        ap50 = precision_scores

        wsi_F1_50.append((f1_scores_50[0]))
        wsi_AP50.append((ap50[0]))
        wsi_dice.append((dice_coefficient(wsi_prediction, mask_tiff_X20)))

        print((f1_scores_50[0]))
        print((ap50[0]))
        print((dice_coefficient(wsi_prediction, mask_tiff_X20)))

        row = len(df)
        df.loc[row] = [case_name, dice_coefficient(wsi_prediction, mask_tiff_X20), f1_scores_50[0], ap50[0]]
        df.to_csv(os.path.join(output_dir, 'testing_wsi_results_all.csv'), index=False)

    print("slide level F1 metric:", np.mean(wsi_F1_50))
    print("slide level AP(50) metric:", np.mean(wsi_AP50))
    print("slide level Dice metric:", np.mean(wsi_dice))

if __name__ == "__main__":
    # with tempfile.TemporaryDirectory() as tempdir:
    input_dir = '/scratch/nmoreau/KPIs_challenge/Validation_bis/'
    output_dir = '/scratch/nmoreau/KPIs_challenge/output_dir/'

    # input_dir = '/Users/nmoreau/Documents/KPIs_challenge/Validation_bis/'
    # output_dir = '/Users/nmoreau/Documents/KPIs_challenge/output_dir/'

    model_dir = '/scratch/nmoreau/KPIs_challenge/training_runs/runs_30_07_2024_13_56_25_fold_0_hover/best_metric_model_segmentation2d_dict_epoch_605.pth'

    df = pd.DataFrame(columns=['case name', 'wsi_dice', 'wsi_F1_50', 'wsi_AP50'])
    main(input_dir, model_dir, output_dir, df)