# -*- coding:utf-8 -*-
import numpy as np
import cv2
import os
import time
from matplotlib import pyplot as plt
from matplotlib import colors
# from visualization import sim_curve
from utils.functions.result_funs import compute_result, save_res_csv, grid_draw


def crop_x(img_gray):
    template_x = img_gray[:, :10]
    return template_x


def crop_y(img_gray):
    template_y = img_gray[:10, :]
    return template_y


def predict(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    i_h = img.shape[0]
    i_w = img.shape[1]

    template_x = crop_x(img_gray)
    template_y = crop_y(img_gray)

    resx = cv2.matchTemplate(img_gray, template_x, cv2.TM_CCOEFF_NORMED)
    dx = np.argmin(resx, axis=1)   # 按每行求出最小值的索引

    resy = cv2.matchTemplate(img_gray, template_y, cv2.TM_CCOEFF_NORMED)
    dy = np.argmin(resy, axis=0)   # 按每行求出最小值的索引

    dstar = [dx, dy]
    return dstar


def _2017run():
    # # 1. creat save path
    INPUT_FOLDER = 'rpd/input/textureimagesallbyfolder/'

    # # 2. load images
    for folder in os.listdir(INPUT_FOLDER):
        if folder[0] == 't':
            save_path = 'rpd/output/2017_TM10/' + str(folder) + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            res_name = save_path + folder + '_results.csv'

            datas = []
            for files in os.listdir(INPUT_FOLDER + folder + '/'):
                try:
                    image_name = INPUT_FOLDER + folder + '/' + files

                    im = cv2.imread(image_name)
                    x_len = im.shape[1]
                    y_len = im.shape[0]
                    start_time = time.time()

                    dstar = predict(image_name)

                    save_name = save_path + files[:-4] + '_2017TM10.png'
                    grid_draw(image_name, dstar, save_name)

                    # # 6. compute time cost and save results
                    time_cost_all = round(time.time() - start_time, 4)
                    iou, prop = compute_result(image_name=files[:-4], img_x=x_len, img_y=y_len, dstar=dstar)
                    data = [files[:-4], '', '', '', time_cost_all, dstar, iou, prop]
                    datas.append(data)

                except Exception as e:
                    with open("rpd/logs/2017_TM10_erro.txt", "a") as f:
                        f.write(str(files) + '\n')
                        f.write(str(e) + '\n')
                continue

            save_res_csv(res_name, datas, refine=False)
            print ('===finish===')


# run(folder='texture1', refine=False, regular=False, visualize=False)