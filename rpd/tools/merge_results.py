# -*- coding:utf-8 -*-
import os
import csv
import pandas as pd


class MergeResults(object):
    def __init__(self, path, outname, num):
        self.path = path
        self.outname = outname
        self.num = num

    def merge2csv(self):
        for folder in os.listdir(self.path):
            folder_path = self.path + folder + '/'
            excels = [pd.read_csv(folder_path + fname) for fname in os.listdir(folder_path) if 'csv' in fname]
            df = pd.concat(excels)
            excel_name = self.path + str(folder) + '.csv'
            df.to_csv(excel_name)

    def getdatas(self):
        self.datas = []
        for curDir, dirs, files in os.walk(self.path):
            for dir in dirs:
                for fname in os.listdir(os.path.join(self.path, dir)):
                    if 'csv' in fname:
                        print fname
                        fname_path = self.path + dir + '/' + fname
                        with open(fname_path, "r") as csvfile:
                            reader = csv.reader(csvfile)
                            reader = list(reader)[1:]
                            for line in reader:
                                self.datas.append(line)
        return self.datas

    def mergemanycsv(self):
        self.datas = []
        # # 读取csv文件
        for fname in os.listdir(self.path):
            if 'csv' in fname:
                fname_path = self.path + fname
                with open(fname_path, "r") as csvfile:
                    reader = csv.reader(csvfile)
                    reader = list(reader)[1:]
                    for line in reader:
                        self.datas.append(line)
        return self.datas

    def writecsv(self):
        if self.num == 1:
            datas = self.getdatas()
        else:
            self.merge2csv()
            datas = self.mergemanycsv()

        excel_name = self.outname
        csv_head = [
            'image_name',
            'filter selection(time_cost1)',
            'peak selection(time_cost2)',
            'voting process(time_cost3)',
            'refine_time(time_cost4)',
            'time_cost_all',
            'dstar_bf',
            'iou_bf',
            'prop_bf',
            'dstar',
            'iou',
            'prop',
        ]
        with open(excel_name, 'w') as csvfile2:
            writer = csv.writer(csvfile2)
            writer.writerow(csv_head)
            writer.writerows(datas)
        print 'finish~'


# TODO(zoe) when use, check the path info

path = '../output/2022_CNN_grid_vgg_canny_0604/'
outname = '../output/2022_CNN_grid_vgg_canny_0604/2022_CNN_grid_vgg_canny_0604_summary.csv'
write2csv = MergeResults(path, outname, 1)
write2csv.writecsv()


