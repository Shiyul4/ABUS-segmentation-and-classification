import glob
import numpy as np
import torch
import os
import cv2

from model.unet_model import UNet

# Some basic or classic models used to test the performance, temporarily not provided on the GitHub
'''
from model.Unet3Plus.UNet_3Plus import UNet_3Plus
from model.ResUnet.resunet_d6_causal_mtskcolor_ddist import *
from model.ResUnet.resunet_d7_causal_mtskcolor_ddist import *
from mxnet import nd
from model.ResUnet2.res_model import *
from model.ResUnet2.res_model2 import *
from model.Alexnet.Alexnet import *
from model.Netcollection import Netcollection
from model.Segnet import Segnet
from model.FCN3.FCN import *
from model.UnetPlusPlus.UnetPlusPlus import UnetPlusPlus
from model.ResUnetPlus.ResUnetPlus import 
'''

from model.ResUnetPlus.ResUnetPlusAtt import ResUnetPlusAtt


from utils.utils_metrics import calculate_metrics
from PIL import Image
from torchvision import transforms
import random

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(0.330, 0.204)
    transforms.Normalize(0.5, 0.5)
])


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Net1
    # net = UNet(n_channels=1, n_classes=2)

    #Net2
    # net = UNet_3Plus(in_channels=1, n_classes=2, feature_scale=4)

    #Net3
    #net = ResNet(block = BasicBlock, blocks_num = [3, 4, 6, 3], num_classes=2, include_top = True)
    
    #Net4
    net = ResUnetPlusAtt()

    # copy net to device
    net.to(device=device)
    # provide model parameters
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    # test modes
    net.eval()
    # read image path
    tests_path = glob.glob('xxx') # tests_path

    def sort_numeric_paths(path):
        basename = os.path.basename(path)  
        numeric_part = os.path.splitext(basename)[0]  
        return int(numeric_part)

    # Sort the paths and rename them in numeric order
    tests_path = sorted(tests_path, key=sort_numeric_paths)

    count = 0
    save_path = "xxx" # save predict results
    pred_dir = save_path
    gt_dir = "xxx" # refer to the ground truth
    name_classes = ["background", "tumors"] # 2 num_classes, former for background, black, later for segmentation, white
    num_classes = 2
    ious, recalls, precisions = [], [], []

    for test_path in tests_path:
        count += 1
        save_res_path = save_path + f'{count}.png'

        gt_path = os.path.join(gt_dir, f'{count}.png')
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        # gt_mask = cv2.resize(gt_mask, (224, 224), interpolation=cv2.INTER_NEAREST)

    
        img = cv2.imread(test_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (512, 512))
        # img = cv2.resize(img, (224, 224))
        img = img.reshape(img.shape[0], img.shape[1])
        img_tensor = transform(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        #print(img_tensor.shape)

        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # predict
        pred = net(img_tensor)
        pred = torch.sigmoid(pred)  
        pred = pred[0, 1, :, :]  
        pred = (pred > 0.5).cpu().numpy().astype(np.uint8) * 255  
        pred = cv2.resize(pred, (512, 512), interpolation=cv2.INTER_NEAREST)
        # pred = cv2.resize(pred, (224, 224), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_res_path, pred)

        iou, recall, precision, dice, dice_loss, accuracy, hd95 = calculate_metrics(pred, gt_mask)
        ious.append(iou)
        recalls.append(recall)
        precisions.append(precision)
    
    # measurement
    mean_iou = np.mean(ious)
    mean_recall = np.mean(recalls)
    mean_precision = np.mean(precisions)
    mean_dice = np.mean(dice) 
    mean_dice_loss = np.mean(dice_loss)
    mean_accuracy = np.mean(accuracy) 
    mean_hd95 = np.mean(hd95)
    std_iou = np.std(ious)
    std_recall = np.std(recalls)
    std_precision = np.std(precisions)
    std_dice = np.std(dice) 
    std_dice_loss = np.std(dice_loss)
    std_accuracy = np.std(accuracy) 
    std_hd95 = np.std(hd95)
    
    print("Get miou.")
    print(f"Mean IoU: {mean_iou:.4f} , Std IoU: {std_iou:.4f}")
    print(f"Mean Recall: {mean_recall:.4f} , Std Recall: {std_recall:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}, Std Precision: {mean_precision:.4f}")
    print(f"Mean_Dice: {mean_dice:.4f} , Std_Dice: {std_dice:.4f}")
    print(f"Mean_Dice_Loss: {mean_dice_loss:.4f} ,  Std_Dice_Loss: {std_dice_loss:.4f}")
    print(f"Mean_Accuracy: {mean_accuracy:.4f} ,  Std_Accuracy: {std_accuracy:.4f}")
    print(f"Mean_Hd95: {mean_hd95:.4f} ,  Std_Hd95: {std_hd95:.4f}")

