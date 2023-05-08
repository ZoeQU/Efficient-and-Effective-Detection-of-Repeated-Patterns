import os
import cv2
import time
import math
import gc
from rpd._2022_our import run
from rpd.utils.functions.hsm import hsm
from rpd.utils.functions.result_funs import compute_result, save_res_csv, grid_draw
from rpd._2022_sift import sift_run
from rpd._2017_TemplateMatching10pixel import _2017run


def only_hsm(pro):
    INPUT_FOLDER = 'rpd/input/textureimagesallbyfolder/'
    Spath = 'rpd/output/2022_tm_10_1103/'
    if not os.path.exists(Spath):
        os.mkdir(Spath)

    for folder in os.listdir(INPUT_FOLDER):
        folder_path = os.path.join(INPUT_FOLDER, folder)

        save_path = Spath + str(folder) + '/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        res_name = save_path + folder + '_results.csv'

        datas = []
        for imgPath in os.listdir(folder_path):
            start = time.time()
            time_cost1 = time_cost2 = time_cost3 = time_cost4 = float(0.00)
            image_name = imgPath[:-4]
            print image_name

            try:
                image_path = os.path.join(folder_path, imgPath)
                img = cv2.imread(image_path)
                image_size = [img.shape[0], img.shape[1]]
                # dstar_bf = [int(pro * image_size[1]), int(pro * image_size[0])]
                dstar_bf = [10, 10]
                iou_bf, prop_bf = compute_result(image_name, image_size[1], image_size[0], dstar_bf)

                dstar = hsm(image_path, dstar_bf, image_name, save_path, visualize=True)
                iou, prop = compute_result(image_name, image_size[1], image_size[0], dstar)
                grid_draw(image_path=image_path, dstar=dstar, save_name=save_path + image_name + '_2022_refine.png')

                time_cost_all = time.time() - start
                data = [image_name, time_cost1, time_cost2, time_cost3, time_cost4, time_cost_all,
                        dstar_bf, iou_bf, prop_bf, dstar, iou, prop]
                datas.append(data)
                gc.collect()

            except Exception as e:
                with open("rpd/logs/" + Spath.split('/')[-2] + ".txt", "a") as f:
                    f.write(str(folder) + '/' + str(image_name) + '\n')
                    f.write(str(e) + '\n')
            continue

        save_res_csv(res_name, datas, refine=True)
    print '===over==='


def Ourdata(input, method, pro):
    if method == 'CNN':
        INPUT_FOLDER = input

        for folder in os.listdir(INPUT_FOLDER):
            if folder[0] != 't':
                print(folder)
                # folder, refine, regular, modeltype, boundarytype, votetype, visualize
                # os.system('python rpd/_2022_our.py %s %s %s %s %s %s %s' % (folder, True, False, 'alexnet', 'canny', 'gaussian', True))
                run(folder, True, False, 'alexnet', 'canny', 'gaussian', True)
        print '===OVER==='

    if method == 'hsm':
        only_hsm(pro)

    if method == '2017':
        _2017run()

    if method == 'sift':
        INPUT_FOLDER = 'rpd/input/textureimagesallbyfolder/'
        for folder in os.listdir(INPUT_FOLDER):
            if folder[0] == 't':
                # num = int(folder[7:])
                print(folder)
                os.system('python rpd/_2022_sift.py %s %s %s %s' % (folder, True, True, True))
        print '===OVER==='


#########################################################################
INPUT_FOLDER = 'rpd/input/'
Ourdata(input=INPUT_FOLDER, method='CNN', pro=float(0.1))

