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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import openface


def load_network(func):
    model_path = './openface/openface'
    net = openface.TorchNeuralNet(os.path.join(model_path, 'nn4.small2.v1.t7'), 96)

    def compute_similarity(n, m):
        nfeats = [0]*len(n)
        mfeats = [0]*len(m)
        scores = np.zeros((len(n), len(m)))

        for i,v in enumerate(n):
            img = cv2.imread(v)
            feat = net.forward(img).copy()
            nfeats[i] = feat

        for i,v in enumerate(m):
            img = cv2.imread(v)
            feat = net.forward(img).copy()
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
