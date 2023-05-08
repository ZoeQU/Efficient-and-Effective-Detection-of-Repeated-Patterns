#!/usr/bin/python
#coding:utf-8
import os
import pickle
import time
import math
import csv
import torch
import torch.nn as nn
from torchvision import models, transforms
from utils.models.AlexNetConvLayers import alexnet_conv_layers
import numpy as np
import matplotlib.pyplot as plt
import gc
import sys
from PIL import Image, ImageFilter, ImageDraw
from skimage.feature import peak_local_max
from utils.functions.custom_plot import custom_plot
from utils.functions.result_funs import compute_result, save_res_csv, grid_draw
from utils.functions.functions2019CNN import (select_filters, select_peaks, displacement_com, get_dstar)
from utils.functions.visualization import visualizeV
from utils.functions.hsm import hsm
# torch.cuda.set_device(0)


preprocess_transform = transforms.Compose([transforms.ToTensor()])
dev = torch.device("cuda")
model = alexnet_conv_layers()
model.to(dev)

Spath = 'output/2019_CNN_grid/'
if not os.path.exists(Spath):
    os.mkdir(Spath)

V_path = 'temps/2019_V_pkls/'
if not os.path.exists(V_path):
    os.mkdir(V_path)


def load_image(img_path):
    dev = torch.device("cuda")
    image = Image.open(img_path).convert('RGB')
    image = image.resize((227, 227))
    return preprocess_transform(image).unsqueeze(0).to(dev)


def run(folder, refine, regular, visualize):
    INPUT_FOLDER = 'input/textureimagesallbyfolder/' + str(folder) + '/'
    save_path = 'output/2019_CNN_grid/' + str(folder) + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    res_name = save_path + folder + '_results.csv'

    datas = []
    for files in os.listdir(INPUT_FOLDER):
        try:
            time0 = time.time()
            img_path = INPUT_FOLDER + files
            img_name = files[:-4]
            image = load_image(img_path)
            image_size = image.squeeze().shape
            image_size = tuple([image_size[1], image_size[2], image_size[0]])  # h,w

            # # 1. conv features computation
            conv_feats = model(image)

            # # 2. filter selection
            F_l = select_filters(conv_feats)
            time1 = time.time()
            time_cost1 = round(time1 - time0, 2)

            # # 3. peaks extraction
            peaks = select_peaks(F_l, image_size)
            time2 = time.time()
            time_cost2 = round(time2 - time1, 2)

            # # 4. compute displacement set and voting space
            pickefile = 'temps/2019_V_pkls/' + "V_" + os.path.basename(img_path) + ".pkl"

            if os.path.exists(pickefile):
                with open(pickefile, 'rb') as f:
                    V = pickle.load(f)

            else:
                V, dstars = displacement_com(peaks, image_size)
                with open(pickefile, 'wb') as handle:
                    pickle.dump(V, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if visualize == True:
                savename = save_path + img_name + '_2019CNN_V.png'
                visualizeV(V, savename)

            dstars = get_dstar(V, image_size)   # h,w

            if len(dstars) > 0:
                dd = []
                for dstar in dstars:
                    dstar = dstar.tolist()
                    dstar = [int(dstar[1] * (image_size[1]/227)), int(dstar[0] * (image_size[0]/227))]  # dstar w,h
                    iou_bf, prop_bf = compute_result(img_name, image_size[1], image_size[0], dstar)
                    d = [dstar, iou_bf, prop_bf]
                    dd.append(d)
                dd = sorted(dd, key=lambda x: x[1])
                dstar_best = dd[-1]

            else:
                dstar_ = ([image_size[1] / 20, image_size[0] / 20])
                iou_bf, prop_bf = compute_result(img_name, image_size[1], image_size[0], dstar_)
                dstar_best = [dstar_, iou_bf, prop_bf]

            grid_draw(image_path=img_path, dstar=dstar_best[0], save_name=save_path + img_name + '_2019CNN.png')

            time3 = time.time()
            time_cost3 = round(time3 - time2, 2)


            if refine:
                dstar_af = hsm(img_path, dstar_best[0], files[:-4], save_path, visualize)
                grid_draw(image_path=img_path, dstar=dstar_af, save_name=save_path + files[:-4] + '_2019CNNrefine.png')
                iou, prop = compute_result(image_name=files[:-4], img_x=image.size(3),
                                           img_y=image.size(2), dstar=dstar_af)
                time4 = time.time()
                time_cost4 = round(time4 - time3, 2)
                time5 = time.time()
                time_cost_all = round(time5 - time0, 2)
                data = [files[:-4], time_cost1, time_cost2, time_cost3, time_cost4, time_cost_all,
                        dstar_best[0], dstar_best[1], dstar_best[2], dstar_af, iou, prop]
                datas.append(data)
            else:
                time5 = time.time()
                time_cost_all = round(time5 - time0, 2)
                data = [files[:-4], time_cost1, time_cost2, time_cost3, time_cost_all,
                        dstar_best[0], dstar_best[1], dstar_best[2]]
                datas.append(data)

            gc.collect()

        except Exception as e:
            print("Erro: " + files)
            with open("logs/2019_CNN_erro.txt", "a") as f:
                f.write(str(files) + '\n')
                f.write(str(e) + '\n')
        continue

    save_res_csv(res_name, datas, regular)


# run(folder='texture1', refine=False, regular=False, visualize=True)