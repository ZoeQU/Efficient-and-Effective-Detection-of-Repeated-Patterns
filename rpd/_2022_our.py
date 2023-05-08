#!/usr/bin/python
#coding:utf-8
import pickle
import numpy as np
import os
import time
import gc
import sys
import cv2
from PIL import Image, ImageDraw, ImageFilter
from matplotlib import pyplot as plt

from utils.functions.custom_plot import custom_plot
from utils.functions.filter_selection_boundary import filter_selection_boundary
from utils.functions.classifier import classifier
from utils.functions.peaks_selection import peaks_selection
from utils.functions.vote_displacement import vote_displacement
from utils.functions.DisplacementVectorVoting_newproposed import displacement_voting
from utils.functions.visualization import visualizeV
from utils.functions.near_regular import near_regular
from utils.functions.hsm import hsm
from utils.functions.result_funs import compute_result, save_res_csv, grid_draw


Spath = 'rpd/output/2022_CNN_Lettrygrid_alexnet_canny_1215/'
if not os.path.exists(Spath):
    os.mkdir(Spath)

V_path = 'rpd/temps/2022_CNN_Lettrygrid_alexnet_canny_1215/'
if not os.path.exists(V_path):
    os.mkdir(V_path)


def run(folder, refine, regular, modeltype, boundarytype, votetype, visualize):
    INPUT_FOLDER = 'rpd/input/Lettry2017/' + str(folder) + '/'
    # INPUT_FOLDER = 'rpd/input/textureimagesallbyfolder/' + str(folder) + '/'
    # INPUT_FOLDER = 'rpd/input/cvpr2010/imagesallbyfolder/' + str(folder) + '/'
    # INPUT_FOLDER = 'rpd/input/PSU Near-Regular Texture Database/PSU_NRT_imagesbyfolder/' + str(folder) + '/'
    save_path = Spath + str(folder) + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    res_name = save_path + folder + '_results.csv'

    datas = []
    for files in os.listdir(INPUT_FOLDER):
            # files = 'c_FJ-ZTS15-SET1332-1415_COL_40_40.jpg'
            image_name = files[:-4]

        # try:
            image_path = INPUT_FOLDER + files
            print files

            time0 = time.time()

            # # 1. filter selection
            F_l, sigma_l, image_size = filter_selection_boundary(votetype, image_name, image_path, save_path, modeltype, boundarytype, visualize)
            time1 = time.time()
            time_cost1 = round(time1 - time0, 2)

            # # 2. peaks selection
            peaks, ori_peaks = peaks_selection(votetype, F_l, image_size, image_path, save_path, image_name, visualize)
            time2 = time.time()
            time_cost2 = round(time2 - time1, 2)

            peak_num = [len(k[0]) for k in peaks]
            print sum(peak_num)


            if votetype == 'gaussian':
                # # 3. voting and get dstar_bf
                pickefile = V_path + "V_" + os.path.basename(image_name) + ".pkl"
                if os.path.exists(pickefile):
                    with open(pickefile, 'rb') as f:
                        V = pickle.load(f)
                else:
                    disps, V = vote_displacement(peaks, image_size, sigma_l)
                    with open(pickefile, 'wb') as handle:
                        pickle.dump(V, handle, protocol=pickle.HIGHEST_PROTOCOL)

                dstar_bf = np.asarray((V[0, 10:].argmax() + 10, V[10:, 0].argmax() + 10))   # dstar[w,h]
                # iou_bf, prop_bf = compute_result(image_name, image_size[1], image_size[0], dstar_bf)
                iou_bf = 0
                prop_bf = 0

                if visualize:
                    savename = save_path + image_name + '_V.png'
                    visualizeV(V, savename)
                else:
                    pass

            else:
                # # # 3. voting and get dstar_bf (version 2)
                savename = save_path + image_name + '_V.png'
                dstar_bf = displacement_voting(peaks, image_size, visualize, savename)
                # iou_bf, prop_bf = compute_result(image_name, image_size[1], image_size[0], dstar_bf)
                iou_bf = 0
                prop_bf = 0

            if regular:
                grid_draw(image_path=image_path, dstar=dstar_bf, save_name=save_path + image_name + '_bf.png')
            else:
                dstar_bf_ = [dstar_bf[1], dstar_bf[0]]
                savename = save_path + image_name + '_' + str(dstar_bf) + '_near_regular_bf.png'
                near_regular(peaks, dstar_bf_, image_path, savename)

            time3 = time.time()
            time_cost3 = round(time3 - time2, 2)

            # # 4. refine dstar
            if refine:
                dstar_bf = list(dstar_bf)
                dstar = hsm(image_path, dstar_bf, image_name, save_path, visualize)
                # iou, prop = compute_result(image_name, image_size[1], image_size[0], dstar)
                iou = 0
                prop = 0

                if regular:
                    grid_draw(image_path=image_path, dstar=dstar, save_name=save_path + image_name + '_2022CNN.png')
                else:
                    dstar_ = [dstar[1], dstar[0]]
                    savename = save_path + image_name + '_' + str(dstar) + '_near_regular_af.png'
                    near_regular(peaks, dstar_, image_path, savename)

                time4 = time.time()
                time_cost4 = round(time.time() - time4, 2)
                time_cost_all = round(time.time() - time0, 2)

                data = [image_name, time_cost1, time_cost2, time_cost3, time_cost4, time_cost_all,
                        dstar_bf, iou_bf, prop_bf, dstar, iou, prop]
                datas.append(data)
                print dstar

            else:
                pass

            gc.collect()

        # except Exception as e:
        #     with open("rpd/logs/" + Spath.split('/')[-2] + ".txt", "a") as f:
        #         print (e)
        #         f.write(str(folder) + '/' + str(image_name) + '\n')
        #         f.write(str(e) + '\n')
        # continue

    # # 5. record
    save_res_csv(res_name, datas, refine)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    print ('===finish===')