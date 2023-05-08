# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


def displacement_voting(peaks, image_size, visualize, savename):
    pp = []
    for a in peaks:
        for b in a[0]:
            pp.append(b)

    peaks = np.array(pp)

    quant_r, quant_c = np.mgrid[0:image_size[0]:1, 0:image_size[1]:1]
    quant_rc = np.empty(quant_r.shape + (2,), dtype=np.float32)
    quant_rc[:, :, 0] = quant_r
    quant_rc[:, :, 1] = quant_c

    pairs_inds = np.asarray(np.meshgrid(np.arange(peaks.shape[0]), np.arange(peaks.shape[0])),
                            dtype=np.uint8).T.reshape(-1, 2)
    pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]

    tmp_disps = np.abs(peaks[pairs_inds[:, 0]] - peaks[pairs_inds[:, 1]])

    x = [r[0] for r in tmp_disps]
    y = [r[1] for r in tmp_disps]
    V = plt.hist2d(x, y, bins=100, normed=colors.LogNorm())
    if visualize:
        plt.savefig(savename + '_voting.png')
        # plt.show()
        plt.close()

    counts = V[0]
    xedges = V[1]
    yedges = V[2]

    counts_sorted = sorted(counts.flatten())

    a = 0
    loc = np.where(counts == counts_sorted[-1 + a])
    while loc[0].any() * loc[1].any() == 0:
        a += 1
        loc = np.where(counts == counts_sorted[-1 - a])

    if len(loc[0]) > 1:
        d = []
        for b in range(len(loc[0])):
            if loc[0][b].any() * loc[1][b].any() != 0:
                d += [int(xedges[loc[0][b]]), int(yedges[loc[0][b]])]
    else:
        d = [int(xedges[loc[0]]), int(yedges[loc[1]])]
# else:
#     d = peaks[0]

    return d
