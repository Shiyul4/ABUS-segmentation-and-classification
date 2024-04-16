import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import cv2
import os
from scipy.spatial.distance import directed_hausdorff

def calculate_metrics(pred_mask, gt_mask):
    # Flatten arrays for simplicity
    pred_mask = pred_mask.flatten()
    gt_mask = gt_mask.flatten()


    tp = np.sum((abs(pred_mask-255) < 1) & (abs(gt_mask-255) < 1))
    fp = np.sum((abs(pred_mask-255) < 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (abs(gt_mask-255) < 1))
    tn = np.sum((pred_mask == 0) & (gt_mask == 0))

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    dice_loss = 1 - dice
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

    pred_mask_2d = pred_mask.reshape((int(np.sqrt(len(pred_mask))), -1))
    gt_mask_2d = gt_mask.reshape((int(np.sqrt(len(gt_mask))), -1))

    pred_points = np.argwhere(pred_mask_2d == 255)
    gt_points = np.argwhere(gt_mask_2d == 255)

    hd_forward = directed_hausdorff(pred_points, gt_points)[0]
    hd_backward = directed_hausdorff(gt_points, pred_points)[0]
    hd95 = np.percentile([hd_forward, hd_backward], 95)

    return iou, recall, precision, dice, dice_loss, accuracy, hd95



