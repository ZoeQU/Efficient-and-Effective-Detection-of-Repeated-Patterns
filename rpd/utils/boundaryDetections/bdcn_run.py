# -*- coding:utf-8 -*-
import argparse
import os
import bdcn.bdcn as bdcn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser('test BDCN')
    parser.add_argument('-c', '--cuda', action='store_true',
                        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default=os.path.abspath('utils/boundaryDetections/bdcn/bdcn_pretrained_on_bsds500.pth'),
                        help='the model to test')
    # parser.add_argument('--res-dir', type=str, default='test_result',
    #     help='the dir to store result')
    # parser.add_argument('--data-root', type=str, default='test_images')
    parser.add_argument('--test-lst', type=str, default=None)
    return parser.parse_args()


def BDCN_run(nm):
    # args = parse_args()
    # # print(args)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # model = bdcn.BDCN()
    # model.load_state_dict(torch.load('%s' % (args.model)))
    # # save_dir = "../../BDCN-master/test_result"
    # if args.cuda:
    #     model.cuda()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = bdcn.BDCN()
    model_path = os.path.abspath('rpd/utils/boundaryDetections/bdcn/bdcn_pretrained_on_bsds500.pth')
    model.load_state_dict(torch.load('%s' % (model_path)))
    model.cuda()
    model.eval()
    mean_bgr = np.array([104.00699, 116.66877, 122.67892])
    data = cv2.imread(nm)
    data = np.array(data, np.float32)
    data -= mean_bgr
    data = data.transpose((2, 0, 1))
    data = torch.from_numpy(data).float().unsqueeze(0)
    # if args.cuda:
    data = data.cuda()
    data = Variable(data)
    out = model(data)
    # out = [F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]]
    out = F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
    return out

# test
