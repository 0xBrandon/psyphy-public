#############################################################################
# MIT License
#
# Copyright (c) 2018 Brandon RichardWebster
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#############################################################################

import os

import cv2
import numexpr as ne
import numpy as np
from skimage.util.shape import view_as_windows
from sklearn.metrics.pairwise import cosine_similarity

# Three-layer Face Network Feature Generation Code
#
# D.D. Cox and N. Pinto, "Beyond Simple Features: A Large-Scale Feature Search
# Approach to Unconstrained Face Recognition", IEEE FG, 2011.
#
# Implementation by Chuan-Yung Tsai (chuanyungtsai@fas.harvard.edu)
#
# PsyPhy-like Implementation by Brandon RichardWebster (brichar1@nd.edu)
def load_network(func):
    (FILT, ACTV, POOL, NORM)     =     range(4)
    (FSIZ, FNUM, FWGH)             =     range(3)
    (AMIN, AMAX)                 =     range(2)
    (PSIZ, PORD)                 =     range(2)
    (NSIZ, NCNT, NGAN, NTHR)     =     range(4)

    net = []

    layer = [[], [], [], []]
    layer[FILT][FSIZ:] = [0]
    layer[FILT][FNUM:] = [1]
    layer[ACTV][AMIN:] = [None]
    layer[ACTV][AMAX:] = [None]
    layer[POOL][PSIZ:] = [0]
    layer[POOL][PORD:] = [0]
    layer[NORM][NSIZ:] = [9]
    layer[NORM][NCNT:] = [0]
    layer[NORM][NGAN:] = [0.1]
    layer[NORM][NTHR:] = [1.0]
    net.append(layer)

    layer = [[], [], [], []]
    layer[FILT][FSIZ:] = [9]
    layer[FILT][FNUM:] = [128]
    layer[ACTV][AMIN:] = [0]
    layer[ACTV][AMAX:] = [None]
    layer[POOL][PSIZ:] = [9]
    layer[POOL][PORD:] = [2]
    layer[NORM][NSIZ:] = [5]
    layer[NORM][NCNT:] = [0]
    layer[NORM][NGAN:] = [0.1]
    layer[NORM][NTHR:] = [10.0]
    net.append(layer)

    layer = [[], [], [], []]
    layer[FILT][FSIZ:] = [3]
    layer[FILT][FNUM:] = [256]
    layer[ACTV][AMIN:] = [0]
    layer[ACTV][AMAX:] = [None]
    layer[POOL][PSIZ:] = [5]
    layer[POOL][PORD:] = [1]
    layer[NORM][NSIZ:] = [3]
    layer[NORM][NCNT:] = [0]
    layer[NORM][NGAN:] = [10.0]
    layer[NORM][NTHR:] = [1.0]
    net.append(layer)

    layer = [[], [], [], []]
    layer[FILT][FSIZ:] = [3]
    layer[FILT][FNUM:] = [512]
    layer[ACTV][AMIN:] = [0]
    layer[ACTV][AMAX:] = [None]
    layer[POOL][PSIZ:] = [9]
    layer[POOL][PORD:] = [10]
    layer[NORM][NSIZ:] = [5]
    layer[NORM][NCNT:] = [1]
    layer[NORM][NGAN:] = [0.1]
    layer[NORM][NTHR:] = [0.1]
    net.append(layer)

    np.random.seed(0)

    for i in xrange(len(net)):
        if (net[i][FILT][FSIZ] != 0):
            net[i][FILT][FWGH:] = [np.random.rand(net[i][FILT][FSIZ], net[i][FILT][FSIZ], net[i-1][FILT][FNUM], net[i][FILT][FNUM]).astype(np.float32)]
            for j in xrange(net[i][FILT][FNUM]):
                net[i][FILT][FWGH][:,:,:,j] -= np.mean(net[i][FILT][FWGH][:,:,:,j])
                net[i][FILT][FWGH][:,:,:,j] /= np.linalg.norm(net[i][FILT][FWGH][:,:,:,j])
            net[i][FILT][FWGH] = np.squeeze(net[i][FILT][FWGH])

    np.random.seed()

    def nepow(X, O):
        return ne.evaluate('X ** O') if (O != 1) else X

    def nediv(X, Y):
        if (np.ndim(X) == 2):
            return ne.evaluate('X / Y')
        else:
            Y = Y[:,:,None]
            return ne.evaluate('X / Y')

    def nemin(X, MIN):
        return ne.evaluate('where(X < MIN, MIN, X)')
    def nemax(X, MAX):
        return ne.evaluate('where(X > MAX, MAX, X)')

    def mcconv3(X, W):
        # print X.shape, W.shape
        X_VAW = view_as_windows(X, W.shape[0:-1])
        Y_FPS = X_VAW.shape[0:2]
        X_VAW = X_VAW.reshape(Y_FPS[0] * Y_FPS[1], -1)
        W = W.reshape(-1, W.shape[-1])
        Y = np.dot(X_VAW, W)
        Y = Y.reshape(Y_FPS[0], Y_FPS[1], -1)

        return Y

    def bxfilt2(X, F_SIZ, F_STRD):
        for i in reversed(xrange(2)):
            W_SIZ = np.ones(np.ndim(X))
            S_SIZ = np.ones(2, dtype=np.uint16)
            W_SIZ[i], S_SIZ[i] = F_SIZ, F_STRD
            X = np.squeeze(view_as_windows(X, tuple(W_SIZ)))[::S_SIZ[0], ::S_SIZ[1]] # subsampling before summation
            X = np.sum(X, -1)
        return X

    def compute(X):
        for i in xrange(len(net)):
            if (net[i][FILT][FSIZ] != 0):         X = mcconv3(X, net[i][FILT][FWGH])

            if (net[i][ACTV][AMIN] != None):     X = nemin(X, net[i][ACTV][AMIN])
            if (net[i][ACTV][AMAX] != None):     X = nemax(X, net[i][ACTV][AMAX])

            if (net[i][POOL][PSIZ] != 0):
                X = nepow    (X, net[i][POOL][PORD])
                X = bxfilt2    (X, net[i][POOL][PSIZ], 2)
                X = nepow    (X, (1.0 / net[i][POOL][PORD]))

            if (net[i][NORM][NSIZ] != 0):
                B = (net[i][NORM][NSIZ] - 1) / 2
                X_SQS = bxfilt2(nepow(X, 2) if (np.ndim(X) == 2) else np.sum(nepow(X, 2), -1), net[i][NORM][NSIZ], 1)

                if (net[i][NORM][NCNT] == 1):
                    X_SUM  = bxfilt2(X if (np.ndim(X) == 2) else np.sum(X, -1), net[i][NORM][NSIZ], 1)
                    X_MEAN = X_SUM / ((net[i][NORM][NSIZ] ** 2) * net[i][FILT][FNUM])

                    X = X[B:X.shape[0]-B, B:X.shape[1]-B] - X_MEAN[:,:,None]
                    X_NORM = X_SQS - ((X_SUM ** 2) / ((net[i][NORM][NSIZ] ** 2) * net[i][FILT][FNUM]))
                    X_NORM = X_NORM ** (1.0/2)
                else:
                    X = X[B:X.shape[0]-B, B:X.shape[1]-B]
                    X_NORM = X_SQS ** (1.0/2)

                np.putmask(X_NORM, X_NORM < (net[i][NORM][NTHR] / net[i][NORM][NGAN]), (1 / net[i][NORM][NGAN]))
                X = nediv(X, X_NORM) # numexpr for large matrix division

        return X

    def compute_similarity(n, m):
        nfeats = [0]*len(n)
        mfeats = [0]*len(m)
        scores = np.zeros((len(n), len(m)))

        for i,v in enumerate(n):
            img = cv2.imread(v, 0)
            img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
            feat = compute(img)
            nfeats[i] = feat

        for i,v in enumerate(m):
            img = cv2.imread(v, 0)
            img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
            feat = compute(img)
            mfeats[i] = feat

        for i, fn in enumerate(nfeats):
            for j, fm in enumerate(mfeats):
                scores[i,j] = cosine_similarity(fn.reshape(1, -1), fm.reshape(1, -1))[0,0]

        min_ = np.min(scores, axis=1)
        max_ = np.max(scores, axis=1)
        scores = (scores - min_.reshape(-1, 1)) / (max_ - min_).reshape(-1, 1)

        return scores

    return compute_similarity

@load_network
def decision(A,B):
    pass

if __name__ == '__main__':
    root = 'images-align'
    imgs = os.listdir(root)
    imgs = map(lambda e: os.path.join(root, e), imgs)

    output = decision(imgs, imgs)
    print output
