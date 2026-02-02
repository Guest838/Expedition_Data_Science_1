import rasterio
import rasterio.mask
import geopandas as gpd
from shapely.geometry import box
import numpy as np
from affine import Affine
from rasterio.windows import Window
from rasterio.features import rasterize
import os
import matplotlib.pyplot as plt
import torch
from torchmetrics import JaccardIndex, Precision, Recall, F1Score
from typing import Any, Dict, Iterable, List, Optional, Tuple
import cv2

#version of preprocessing for ae photos
def apply_clahe_ae(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    
    if image.shape[0] == 3:
        image = np.transpose(image,(1,2,0)).astype(np.uint8)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_channel = clahe.apply(l_channel)
        
        lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        enhanced =np.transpose(enhanced,(2,0,1))
    else:
        enhanced = image

    return enhanced

#version of preprocessing for lidar photos
def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
   
    if image.shape[0] == 3:
        image = np.transpose(image,(1,2,0))
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_channel = clahe.apply(l_channel)
        
        lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        enhanced =np.transpose(enhanced,(2,0,1))
    else:
        enhanced = image
    return enhanced


# get window of multichannel input
def merge_multiband_windowss(image_paths,x,y,tile_size):

    new_transform =None
    arrays = []
    for images in image_paths:
        with rasterio.open(images[0]) as src:
            count = src.count
            height = tile_size
            width = tile_size
            img_width, img_height = src.width, src.height
            read_x2 = min(img_width, x + tile_size)
            read_y2 = min(img_height, y + tile_size)
            read_width = read_x2 - x
            read_height = read_y2 -y 
            out_shape = (3, height, width)
            out_array = np.full(out_shape, 0, dtype=float)
            data = None
            if read_width > 0 and read_height > 0:
                read_window = Window(x, y, read_width, read_height)
                if count==3:
                    data = src.read(window=read_window)
                elif count>3:
                    data = src.read([1, 2, 3],window=read_window)
                elif count==1:
                    band = src.read(1, window=read_window) 
                    data = np.stack([band, band, band], axis=0) 
            
            if data is not None:
                data = apply_clahe(data)
                out_array[
                    :,
                    0 : 0 + read_height,
                    0 : 0 + read_width,

                ] = data/255
            new_transform = src.transform * rasterio.Affine.translation(x, y)
            arrays.append(out_array)
    merged_array = np.concatenate(arrays)
    return merged_array

# get window of single channel image    
def get_windowss(image_path,x,y,tile_size):
    new_transform =None
    with rasterio.open(image_path) as src:
        count = src.count
        height = tile_size
        width = tile_size
        img_width, img_height = src.width, src.height
        read_x2 = min(img_width, x + tile_size)
        read_y2 = min(img_height, y + tile_size)
        read_width = read_x2 - x
        read_height = read_y2 -y 
        out_shape = (3, height, width)
        out_array = np.full(out_shape, 0, dtype=float)
        data = None
        if read_width > 0 and read_height > 0:
            read_window = Window(x, y, read_width, read_height)
            if count==3:
                data = src.read(window=read_window)
            elif count>3:
                data = src.read([1, 2, 3],window=read_window)
            elif count==1:
                band = src.read(1, window=read_window) 
                data = np.stack([band, band, band], axis=0) 
        if data is not None:
            data = apply_clahe_ae(data)
            out_array[
                :,
                0 : 0 + read_height,
                0 : 0 + read_width,

            ] = data/255
        new_transform = src.transform * rasterio.Affine.translation(x, y)

    return out_array
    
#compute metrics during validation
def metrics_test(pred, target, batch_img, threshold, device):
     jaccard = JaccardIndex(task='binary', threshold=threshold).to(device)
     precision = Precision(task='binary').to(device)
     recall = Recall(task='binary', threshold=threshold).to(device)
     f1 = F1Score(task="binary", threshold=threshold).to(device)
     ja = jaccard(pred, target)
     prec = precision(pred, target)
     rec =recall(pred, target)
     ff1 =f1(pred, target)
     pred = (pred>threshold).float()
     
     for i in range(len(target)):
            with torch.no_grad():
                pred = pred.cpu().squeeze(0)
                target = target.cpu().squeeze(0)
                batch_img =batch_img.cpu().squeeze(0)
                plt.imshow(target[i])
                plt.colorbar()
                plt.show()
                plt.imshow(pred[i])
                plt.colorbar()
                plt.show()
                plt.imshow(batch_img[i])
                plt.colorbar()
                plt.show()
     return ja, prec, rec, ff1

#compute metrics during training
def metrics(pred, target, batch_img, threshold, device):
     jaccard = JaccardIndex(task='binary', threshold=threshold).to(device)
     precision = Precision(task='binary').to(device)
     recall = Recall(task='binary', threshold=threshold).to(device)
     f1 = F1Score(task="binary", threshold=threshold).to(device)
     ja = jaccard(pred, target)
     prec = precision(pred, target)
     rec =recall(pred, target)
     ff1 =f1(pred, target)
     pred = (pred>threshold).float()
     
     for i in range(len(target)):
            with torch.no_grad():
                pred = pred.cpu().squeeze(0)
                target = target.cpu().squeeze(0)
                batch_img =batch_img.cpu().squeeze(0)
                plt.imshow(target[i][0])
                plt.colorbar()
                plt.show()
                plt.imshow(pred[i][0])
                plt.colorbar()
                plt.show()
                plt.imshow(batch_img[i][0])
                plt.colorbar()
                plt.show()
     return ja, prec, rec, ff1

# metrics for classification neural network
def metrics_det(pred, target, batch_img, device):
     precision = Precision(task='multiclass', num_classes=11).to(device)
     recall = Recall(task='multiclass', num_classes=11).to(device)
     f1 = F1Score(task='multiclass', num_classes=11).to(device)
     prec = precision(pred, target)
     rec =recall(pred, target)
     ff1 =f1(pred, target)
     
     for i in range(len(target)):
            with torch.no_grad():
                plt.imshow(batch_img[i][0].cpu())
                plt.colorbar()
                plt.show()
                print(pred[i].cpu())
                print(target[i].cpu())
     return  prec, rec, ff1

#metrics adapts for single-channel images
def metrics_ae(pred, target, batch_img, threshold, device):
    jaccard = JaccardIndex(task='binary', threshold=threshold).to(device)
    precision = Precision(task='binary').to(device)
    recall = Recall(task='binary', threshold=threshold).to(device)
    f1 = F1Score(task="binary", threshold=threshold).to(device)
    ja = jaccard(pred, target)
    prec = precision(pred, target)
    rec =recall(pred, target)
    ff1 =f1(pred, target)
    pred = (pred>threshold).float()
    
    for i in range(len(target)):
            with torch.no_grad():
                pred = pred.cpu().squeeze(0)
                target = target.cpu().squeeze(0)
                batch_img =batch_img.cpu().squeeze(0)
               
                plt.imshow(target[i][0])
                plt.colorbar()
                plt.show()
                plt.imshow(pred[i][0])
                plt.colorbar()
                plt.show()
                plt.imshow(batch_img[i][0])
                plt.colorbar()
                plt.show()

    return ja, prec, rec, ff1

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def plot_loss(Loss_train):
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(Loss_train)), Loss_train, color='orange', label='train', linestyle='--')
    plt.legend()
    plt.show()