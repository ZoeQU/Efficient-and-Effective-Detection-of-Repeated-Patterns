# -*- coding:utf-8 -*-
import numpy as np
import cv2
import os
import time
import sys
import gc
import pickle
import random
from skimage.feature import peak_local_max
from sklearn.cluster import AffinityPropagation

from utils.functions.custom_plot import custom_plot
from utils.functions.filter_selection_boundary import filter_selection_boundary
from utils.functions.classifier import classifier
from utils.functions.peaks_selection import peaks_selection
from utils.functions.vote_displacement import vote_displacement
from utils.functions.visualization import visualizeV
from utils.functions.near_regular import near_regular
from utils.functions.hsm import hsm
from utils.functions.result_funs import compute_result, save_res_csv, grid_draw


INPUT_FOLDER = 'rpd/input/textureimagesallbyfolder/'

Spath = 'rpd/output/2022_sift_cluster_0419/'
if not os.path.exists(Spath):
    os.mkdir(Spath)

V_path = 'rpd/temps/2022_sift_cluster_0419/'
if not os.path.exists(V_path):
    os.mkdir(V_path)


def show_peaks(peaks, img, save_name):
    point_size = 7
    point_color = (255, 0, 0)  # BGR
    thickness = -1
    point_list = peaks
    for point in point_list:
        point = tuple(point)
        cv2.circle(img, point, point_size, point_color, thickness)
    cv2.imwrite(save_name, img)


def sift_run(folder, visualize, peaks_cluster, regular):
    INPUT_FOLDER = 'rpd/input/textureimagesallbyfolder/' + str(folder) + '/'
    save_path = Spath + str(folder) + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    res_name = save_path + folder + '_results.csv'

    datas = []
    for files in os.listdir(INPUT_FOLDER):
        image_name = files[:-4]
        print image_name

        try:
            image_path = INPUT_FOLDER + files

            # # 1. sift keypoints
            time0 = time.time()
            img = cv2.imread(image_path)
            img_sift = img.copy()
            image_size = [img.shape[0], img.shape[1]]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kp = sift.detect(gray, None)

            img_sift = cv2.drawKeypoints(gray, kp, img_sift)
            cv2.imwrite(save_path + image_name + '_sift_keypoints.jpg', img_sift)

            kps = cv2.KeyPoint_convert(kp)
            kps2 = []
            for x in kps:
                i = int(x[1])
                j = int(x[0])
                kps2.append([i, j])

            time1 = time.time()
            time_cost1 = round(time1 - time0, 2)

        # # 2. peaks selection
            if peaks_cluster == '':
                map = np.zeros(tuple(image_size))
                for ij in kps2:
                    x = ij[0]
                    y = ij[1]
                    map[x, y] += 1
                area_para = 50
                peaks = peak_local_max(map, area_para)
                save_name = save_path + image_name + '_nms_peaks.jpg'
                show_peaks(peaks, img, save_name)

            else:
                clustering = AffinityPropagation().fit(kps2)
                cluster_centers_indices = clustering.cluster_centers_indices_
                labels = clustering.labels_
                n_clusters_ = len(cluster_centers_indices)
                peaks = [kps2[i] for i in cluster_centers_indices]
                save_name = save_path + image_name + '_cluster_peaks.jpg'
                show_peaks(peaks, img, save_name)

            time2 = time.time()
            time_cost2 = round(time2 - time1, 2)

            # # 3. voting and get dstar_bf
            pickefile = V_path + "V_" + os.path.basename(image_name) + ".pkl"
            if os.path.exists(pickefile):
                with open(pickefile, 'rb') as f:
                    V = pickle.load(f)
            else:
                sigma_l = [8, 8]
                peaks = np.array(peaks)
                peaks = peaks[np.newaxis, np.newaxis, :, :]
                disps, V = vote_displacement(peaks, image_size, sigma_l)
                with open(pickefile, 'wb') as handle:
                    pickle.dump(V, handle, protocol=pickle.HIGHEST_PROTOCOL)

            dstar_bf = np.asarray((V[0, 10:].argmax() + 10, V[10:, 0].argmax() + 10))  # dstar[w,h]
            iou_bf, prop_bf = compute_result(image_name, image_size[1], image_size[0], dstar_bf)
            grid_draw(image_path=image_path, dstar=dstar_bf, save_name=save_path + image_name + '_bf.png')

            if visualize:
                savename = save_path + image_name + '_V.png'
                visualizeV(V, savename)
            else:
                pass

            time3 = time.time()
            time_cost3 = round(time3 - time2, 2)

            # # 4. refine dstar
            dstar = hsm(image_path, list(dstar_bf), image_name, save_path, visualize)
            iou, prop = compute_result(image_name, image_size[1], image_size[0], dstar)

            if regular:
                grid_draw(image_path=image_path, dstar=dstar, save_name=save_path + image_name + '_2022CNN.png')
            else:
                near_regular(kp, peaks, dstar, image_path, save_path)

            time4 = time.time()
            time_cost4 = round(time.time() - time4, 2)
            time_cost_all = round(time.time() - time0, 2)

            data = [image_name, time_cost1, time_cost2, time_cost3, time_cost4, time_cost_all,
                    dstar_bf, iou_bf, prop_bf, dstar, iou, prop]
            datas.append(data)

            print dstar

            gc.collect()

        except Exception as e:
            with open("rpd/logs/" + Spath.split('/')[-2] + ".txt", "a") as f:
                f.write(str(folder) + '/' + str(image_name) + '\n')
                f.write(str(e) + '\n')
        continue
    save_res_csv(res_name, datas, refine=True)

if __name__ == "__main__":
    sift_run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    print ('===finish===')