# -*- coding:utf-8 -*-
import pickle
import numpy as np
import os
import cv2
import time
import math
import heapq
import random as rand
import csv
from skimage import measure
from matplotlib import pyplot as plt
from matplotlib import colors
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from PIL import Image, ImageDraw, ImageFilter
import torch
from torchvision import models, transforms
from ..models.AlexNetConvLayers import alexnet_conv_layers
from ..models.VGGConvLayers import vgg16_conv_layers
from ..models.ResNetConvLayers import resnet50_conv_layers
from ..models.InceptionConvLayers import inceptionv3_conv_layers
from collections import Counter
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from scipy.stats import multivariate_normal
from skimage.feature import peak_local_max

from ..boundaryDetections.bdcn_run import BDCN_run as bdcn


preprocess_transform = transforms.Compose([transforms.ToTensor()])


def load_image(img_path):
    dev = torch.device("cuda")
    image = Image.open(img_path).convert('RGB')
    image = image.filter(ImageFilter.SMOOTH_MORE)
    # image = image.resize((227, 227))
    return preprocess_transform(image).unsqueeze(0).to(dev)


def get_boundary(image_name, image_path, boundarytype):
    if boundarytype == 'bdcn':
        BDCN_path = 'rpd/temps/boundarys/' + image_name + '_bdcn.png'
        fuse = 255 * bdcn(image_path)
        fuse_save = 255 - fuse  # opposite black, white and save
        cv2.imwrite(BDCN_path, fuse_save)
        return fuse

    if boundarytype == 'rcf':
        pass

    if boundarytype == 'canny':
        boundary_image = cv2.imread(image_path)
        fuse = cv2.Canny(boundary_image, 400, 400)
        plt.imshow(fuse, cmap='gray')
        # plt.show()
        plt.close()
        return fuse

    if boundarytype == 'hed':
        pass


def select_with_boundary(modeltype, conv_filters, boundary, image_size):
    F_l = []
    sigma_l = []
    fmap_b_max = []

    for li, l in enumerate(conv_filters):
        F_l.append([])
        fmap_b_max.append([])
        maps = l.squeeze().detach().cpu().numpy()

        if modeltype == 'alexnet':
            t = (image_size[0] / maps.shape[1]) / 2
            sigma_l.append(t)

        if modeltype == 'vgg':
            t = (image_size[0] / maps.shape[1]) / 1
            sigma_l.append(t)

        if modeltype == 'resnet':
            t = (image_size[0] / maps.shape[1]) / 1
            sigma_l.append(t)

        for fi, fmap in enumerate(maps[:]):
            boundary_fi = np.array(Image.fromarray(boundary).resize((fmap.shape[1], fmap.shape[0])))
            boundary_fi = np.where(boundary_fi > 0.01, 1, 0)
            fmap_b = boundary_fi * fmap
            fmap_b_max[li].append(np.sum(fmap_b))

        index_list = map(fmap_b_max[li].index, heapq.nlargest(1, fmap_b_max[li]))  # select 1 filter from each CNN layer
        fmapp = maps[index_list]

        for ii in range(len(fmapp)):
            F_l[li].append(fmapp[ii])
    return F_l, sigma_l


def filter_selection_boundary(votetype, image_name, image_path, save_path, modeltype, boundarytype, visualize):

    # # 1. load CNN model
    image = load_image(image_path)
    image_size = image.squeeze().shape
    image_size = tuple([image_size[1], image_size[2], image_size[0]])
    dev = torch.device("cuda")

    if modeltype == 'alexnet':
        model = alexnet_conv_layers()

    elif modeltype == 'vgg':
        model = vgg16_conv_layers()

    else:
        model = resnet50_conv_layers()

    model.to(dev)
    out_ = model(image)
    if votetype == 'gaussian' and modeltype == 'vgg':
        filters = out_[0]
    else:
        filters = out_[1]

    # # 2. add boundary
    boundary = get_boundary(image_name, image_path, boundarytype)

    # # 3. filter selection with boundary
    F_l, sigma_l = select_with_boundary(modeltype=modeltype, conv_filters=filters, boundary=boundary, image_size=image_size)

    if visualize:
        for ii in range(len(F_l)):
            plt.imshow(F_l[ii][0])
            plt.axis('off')
            savename = save_path + image_name + str(ii) + '_fmap.png'
            plt.savefig(savename, bbox_inches='tight', pad_inches=0)
            # plt.show()
            plt.close()
    return F_l, sigma_l, image_size