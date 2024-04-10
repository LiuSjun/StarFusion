import os
import sys
import logging
import time
import argparse
import numpy as np
from collections import OrderedDict
import data
import options.options as option
import utils.util as util
import cv2
from tqdm import tqdm
import torch
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

if __name__ == '__main__':

    # if you use local edition to train this model, just input json location  into follow variable
    jsonpath = r'D:\SuperResolution\SPSR-master\code\options\test\test_spsr.json'
    opt = option.parse(jsonpath, is_train=False)
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
    opt = option.dict_to_nonedict(opt)
    path_img = r'D:\SuperResolution\SPSR-master\code\data\Test_Remote'
    test_set_name = 'S20201222sub.png'
    path = os.path.join(path_img, test_set_name)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    img = img[:, :, [2, 1, 0]]
    img = np.transpose(img, (2, 0, 1))#.astype(np.uint8)

    # Create model
    model = create_model(opt)
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    # save images
    suffix = opt['suffix']
    if suffix:
        save_img_path = os.path.join(dataset_dir, test_set_name + suffix)
    else:
        save_img_path = os.path.join(dataset_dir, test_set_name)
        save_img_g_path = os.path.join(dataset_dir,  'grad_' + test_set_name)

    mask = None
    interval = 50
    start = 50#16
    size = 100
    scale = 4
    b, xlen, ylen = img.shape
    half_size = int(size / 2)
    x_center = np.arange(start, xlen, interval, dtype=np.int)
    y_center = np.arange(start, ylen, interval, dtype=np.int)
    x_center, y_center = np.meshgrid(x_center, y_center)

    img_count = np.zeros((b,xlen*scale, ylen*scale), dtype=np.int16)
    img_seg = np.zeros((xlen*scale, ylen*scale,b), dtype=np.float32)
    img_gur = np.zeros((xlen * scale, ylen * scale), dtype=np.float32)

    xlen_chip, ylen_chip = x_center.shape
    with torch.autograd.set_detect_anomaly(True) and torch.no_grad():
        for i in tqdm(range(xlen_chip)):
            for j in range(ylen_chip):
                xloc0, xloc1 = max((x_center[i, j] - half_size, 0)), min((x_center[i, j] + half_size, xlen))
                yloc0, yloc1 = max((y_center[i, j] - half_size, 0)), min((y_center[i, j] + half_size, ylen))
                subset_img = np.zeros((b, size, size), dtype=np.float32)
                xsize, ysize = xloc1 - xloc0, yloc1 - yloc0
                subset_img[:, :xsize, :ysize] = img[:, xloc0:xloc1, yloc0:yloc1]
                subset_img = np.expand_dims(subset_img, 0)
                subset_img = torch.from_numpy(np.ascontiguousarray(subset_img)).float()
                subset_img_torch = subset_img.cuda()
                fake_H_branch, fake_H = model.testforrs(subset_img_torch)  # test
                out_seg = util.tensor2img(fake_H)
                fake_H_branch = util.tensor2img(fake_H_branch)
                if np.max(out_seg) == 0:
                    continue
                no_value_loc = np.where(np.mean(out_seg, axis=2) == 0, 0, 1)
                no_value_loc = no_value_loc.astype(np.uint8)
                out_seg[:,:,0] *= no_value_loc
                out_seg[:,:,1] *= no_value_loc
                out_seg[:,:,2] *= no_value_loc

                fake_H_branch[:,:,0] *= no_value_loc

                if mask is not None:
                    img_count[xloc0:xloc1, yloc0:yloc1] += mask[:xsize, :ysize]
                    out_seg *= mask
                else:
                    img_count[:,xloc0*scale:xloc1*scale, yloc0*scale:yloc1*scale] += 1

                img_seg[xloc0*scale:xloc1*scale, yloc0*scale:yloc1*scale,:] += out_seg[:xsize*scale, :ysize*scale]
                img_gur[xloc0 * scale:xloc1 * scale, yloc0 * scale:yloc1 * scale] += fake_H_branch[:xsize * scale,
                                                                                        :ysize * scale,0]

    epsilon = 1e-7
    img_count_grad = img_count[0]
    img_count = img_count.transpose(1, 2, 0)
    img_seg = img_seg / (img_count + epsilon)
    img_gur = img_gur / (img_count_grad + epsilon)

    sr_img = img_seg
    img_gur = img_gur


    util.save_img(sr_img, save_img_path)
    util.save_img(img_gur, save_img_g_path)
