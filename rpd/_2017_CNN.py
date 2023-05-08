#!/usr/bin/python
#coding:utf-8
import os
import pickle
import gc
import time
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from utils.models.AlexNetConvLayers import alexnet_conv_layers
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from scipy.stats import multivariate_normal
from skimage.feature import peak_local_max
from utils.functions.custom_plot import custom_plot
from utils.functions.result_funs import compute_result, save_res_csv, grid_draw
from utils.functions.visualization import visualizeV
from utils.functions.hsm import hsm

preprocess_transform = transforms.Compose([transforms.ToTensor()]) # #归一化到(0,1)，简单直接除以255
dev = torch.device("cuda")


def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    return preprocess_transform(image).unsqueeze(0).to(dev)


model = alexnet_conv_layers()
model.to(dev)

Spath = 'output/2017_CNN_grid/'
if not os.path.exists(Spath):
    os.mkdir(Spath)

V_path = 'temps/2017_V_pkls/'
if not os.path.exists(V_path):
    os.mkdir(V_path)


def run(folder, refine, regular, visualize):
    # 0. parameters
    sigma_l = []
    alfa_l = [5, 7, 15, 15, 15]
    fi_prctile = 80
    delta = 0.65
    subsample_pairs = 10
    peaks_max = 10000

    INPUT_FOLDER = 'input/textureimagesallbyfolder/' + str(folder) + '/'
    save_path = 'output/2017_CNN_grid/' + str(folder) + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    res_name = save_path + folder + '_results.csv'

    datas = []
    for files in os.listdir(INPUT_FOLDER):
        try:
            time0 = time.time()
            img_path = INPUT_FOLDER + files
            image = load_image(img_path)
            image_name = files[:-4]

            # 1. conv features computation
            conv_feats = model(image)
            time1 = time.time()
            time_cost1 = round(time1 - time0, 2)

            # 2. peaks extraction
            peaks = []
            for li, l in enumerate(conv_feats):
                peaks.append([])
                maps = l.squeeze().detach().cpu().numpy()
                sigma_l.append((image.size(2) / maps.shape[1]) / 2)

                for fi, fmap in enumerate(maps):
                    fmap = np.array(Image.fromarray(fmap).resize((image.size(2), image.size(3))))
                    fmap = gaussian_filter(fmap, sigma=10)
                    tmp_max = maximum_filter(fmap, 1)  # origins 求二维阵列在一定值以上的局部极大值坐标
                    max_coords = peak_local_max(tmp_max, 5)

                    peaks[li].append(max_coords[np.random.permutation(max_coords.shape[0])[:peaks_max]])

            time2 = time.time()
            time_cost2 = round(time2 - time1, 2)

            # 3. compute displacement set and voting space
            pickefile = 'temps/2017_V_pkls/' + "V_" + os.path.basename(img_path) + ".pkl"
            if os.path.exists(pickefile):
                with open(pickefile, 'rb') as f:
                    V = pickle.load(f)
            else:
                quant_r, quant_c = np.mgrid[0:image.size(2):1, 0:image.size(3):1]
                V = np.zeros(quant_r.shape)
                quant_rc = np.empty(quant_r.shape + (2,), dtype=np.float32)
                quant_rc[:, :, 0] = quant_r
                quant_rc[:, :, 1] = quant_c
                disps = []
                for li, p in enumerate(peaks):
                    disps.append([])
                    for fi, p2 in enumerate(p):
                        pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])),
                                                dtype=np.uint8).T.reshape(-1, 2)
                        pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
                        if pairs_inds.shape[0] > 0:
                            tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
                        else:
                            tmp_disps = np.asarray([[]])
                        if tmp_disps.size == 0:
                            continue
                        tmp_disps = tmp_disps[np.random.permutation(tmp_disps.shape[0])[:subsample_pairs]]
                        for ij, dij in enumerate(tmp_disps):
                            tmp_Vfiij = multivariate_normal.pdf(quant_rc, mean=dij,
                                                                cov=np.asarray([[sigma_l[li], 0], [0, sigma_l[li]]],
                                                                               dtype=np.float32))
                            tmp_Vfiij /= tmp_disps.shape[0]
                            V += tmp_Vfiij

                with open(pickefile, 'wb') as handle:
                    pickle.dump(V, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if visualize:
                savename = save_path + files[:-4] + '_2017CNN_V.png'
                visualizeV(V, savename)

            # 4. find best step (repeat pattern size)
            starting_ind = 10
            dstar = np.asarray((V[starting_ind:, 0].argmax() + starting_ind
                                , V[0, starting_ind:].argmax() + starting_ind))
            dstar_bf = list(reversed(dstar))  # dstar[::-1]  #[dstar[1], dstar[0]]
            iou_bf, prop_bf = compute_result(image_name=files[:-4], img_x=image.size(3),
                                             img_y=image.size(2), dstar=dstar_bf)

            time3 = time.time()
            time_cost3 = round(time3 - time2, 2)
            grid_draw(image_path=img_path, dstar=dstar_bf, save_name=save_path + files[:-4] + '_2017CNN_bf.png')

            if refine:
                dstar = hsm(img_path, dstar_bf, image_name, save_path, visualize)
                grid_draw(image_path=img_path, dstar=dstar, save_name=save_path + files[:-4] + '_2017CNNrefine.png')
                iou, prop = compute_result(image_name=files[:-4], img_x=image.size(3),
                                           img_y=image.size(2), dstar=dstar)
                time4 = time.time()
                time_cost4 = round(time4 - time3, 2)
                time5 = time.time()
                time_cost_all = round(time5 - time0, 2)
                data = [image_name, time_cost1, time_cost2, time_cost3, time_cost4, time_cost_all,
                        dstar_bf, iou_bf, prop_bf, dstar, iou, prop]
                datas.append(data)

            else:
                time5 = time.time()
                time_cost_all = round(time5 - time0, 2)
                data = [image_name, time_cost1, time_cost2, time_cost3, time_cost_all,
                        dstar_bf, iou_bf, prop_bf]
                datas.append(data)

            if regular:
                grid_draw(image_path=img_path, dstar=dstar_bf, save_name=save_path + files[:-4] + '_2017CNN.png')

            else:
                # 6.1. compute consistent votes to compute fi
                fi_acc = []
                for li, p in enumerate(peaks):
                    for fi, p2 in enumerate(p):
                        pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])),
                                                dtype=np.uint8).T.reshape(-1, 2)
                        pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]

                        if pairs_inds.shape[0] > 0:
                            tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
                        else:
                            fi_acc.append(0)
                            continue
                        tmp_disps = tmp_disps[np.random.permutation(tmp_disps.shape[0])[:subsample_pairs]]
                        fi_acc.append(len([1 for dij in tmp_disps if (np.linalg.norm(dij - dstar)) < 3 * alfa_l[li]]))

                param_fi = np.percentile(fi_acc, fi_prctile)

                # 6.2. find weights for filters
                disps_star = []
                weights = []
                for li, p in enumerate(peaks):
                    disps_star.append([])
                    weights.append([])
                    for fi, p2 in enumerate(p):
                        # pairs_inds = np.asarray([(i, j) for i in range(p2.shape[0]) for j in range(p2.shape[0]) if i != j and j > i])
                        pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])),
                                                dtype=np.uint8).T.reshape(
                            -1, 2)
                        pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
                        if pairs_inds.shape[0] > 0:
                            tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
                        else:
                            tmp_disps = np.asarray([[]])
                        weights[li].append(0)
                        if tmp_disps.size == 0:
                            continue
                        tmp_disps = tmp_disps[np.random.permutation(tmp_disps.shape[0])[:subsample_pairs]]
                        # disps_star[li].append(tmp_disps)
                        # tmp_disps è Dfl

                        for ij, dij in enumerate(tmp_disps):
                            tmp_diff = np.linalg.norm(dij - dstar)
                            if tmp_diff < 3 * alfa_l[li]:
                                # φ è 80esimo percentile, bisogna sommare i pesi per calcolare per ogni filtro
                                wijfl = np.exp(-(tmp_diff ** 2)
                                               / (2 * (alfa_l[li] ** 2))) \
                                        / (tmp_disps.shape[0] + param_fi)
                                weights[li][-1] += wijfl

                # 6.3. find filters with weights higher than threshold
                selected_filters = []
                for li, w in enumerate(weights):
                    tmp_weight_thr = delta * max(w)
                    selected_filters.append([fi for fi, w2 in enumerate(w) if w2 > tmp_weight_thr])

                # 6.4. accumulate origin coordinates loss
                acc_origin = []
                acc_origin_weights = []
                for li, w in enumerate(weights):
                    for fi in selected_filters[li]:
                        p2 = peaks[li][fi]
                        pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])),
                                                dtype=np.uint8).T.reshape(
                            -1, 2)
                        pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
                        if pairs_inds.shape[0] > 0:
                            tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
                        else:
                            fi_acc.append(0)
                            continue
                        cons_disps = [dij for ij, dij in enumerate(tmp_disps)
                                      if (np.linalg.norm(dij - dstar)) < 3 * alfa_l[li]]
                        cons_disps_weights = [
                            np.exp(-(np.linalg.norm(dij - dstar) ** 2) / (2 * (alfa_l[li] ** 2))) / (tmp_disps.shape[0] + param_fi)
                            for dij in cons_disps]
                        acc_origin.extend(cons_disps)
                        acc_origin_weights.extend(cons_disps_weights)

                o_r = np.linspace(-dstar[0], dstar[0], 10)
                o_c = np.linspace(-dstar[1], dstar[1], 10)
                min_rc = (-1, -1)
                min_val = np.inf
                for r in o_r:
                    for c in o_c:
                        tmp_orig = np.asarray([r, c])
                        tmp_val = [np.linalg.norm(np.mod((dij - tmp_orig), dstar) - (dstar / 2)) * acc_origin_weights[ij]
                                        for ij, dij in enumerate(acc_origin)]
                        tmp_val = np.sum(tmp_val)
                        if tmp_val < min_val:
                            min_val = tmp_val
                            min_rc = (r, c)

                boxes = []
                tmp_img = np.array(Image.open(img_path))
                for ri in range(100):
                    min_r = min_rc[0] + (dstar[0] * ri) - (dstar[1] / 2)
                    if min_r > tmp_img.shape[0]:
                        break
                    for ci in range(100):
                        min_c = min_rc[1] + (dstar[1] * ci) - dstar[0] / 2
                        if min_c > tmp_img.shape[1]:
                            break
                        tmp_box = np.asarray([min_c, min_r, dstar[1], dstar[0]])
                        boxes.append(tmp_box)
                custom_plot(tmp_img, box=boxes, polygons=None, save_path=save_path + files[:-4] + '_2017CNN_irregular.png')

            gc.collect()

        except Exception as e:
            with open("logs/2017_CNN_erro_HSM.txt", "a") as f:
                f.write(str(files) + '\n')
                f.write(str(e) + '\n')
        continue


    save_res_csv(res_name, datas, refine)
    print ('===finish===')


run(folder='texture1', refine=False, regular=False, visualize=True)







