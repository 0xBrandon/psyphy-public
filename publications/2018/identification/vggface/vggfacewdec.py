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

import caffe
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PERCENT = 10

def load_network(func):
    layer = 'fc7'
    base = './vggface'
    deploy = os.path.join(base,'VGG_FACE_deploy.prototxt')
    model = os.path.join(base,'VGG_FACE.caffemodel')
    mean = os.path.join(base,'vgg_mean.npy')

    caffe.set_mode_gpu() # use GPU instead of cpu
    caffe.set_device(0) # use GPU 0

    net = caffe.Net(deploy, model, caffe.TEST)

    def modify(layer):
        # std = np.std(net.params[layer][0].data)
        # mean = np.mean(net.params[layer][0].data)
        # print np.min(net.params[layer][0].data), np.max(net.params[layer][0].data), mean, std
        shape = net.params[layer][0].data.shape
        nweights = np.random.normal(size=shape)
        no = np.random.random_integers(0, 99, size=shape)
        no = no >= PERCENT
        nweights[no] = net.params[layer][0].data[no]
        net.params[layer][0].data[...] = nweights


    for k in net.params:
        modify(k)
    # for i in xrange(len(net.params['fc6'])):
    #     print np.min(net.params['fc6'][i].data), np.max(net.params['fc6'][i].data)
    # help(net.params['fc6'])

    if layer not in net.blobs:
        raise TypeError('Invalid layer name: ' + layer)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(mean))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255.0)

    def compute_similarity(n, m):
        nfeats = [0]*len(n)
        mfeats = [0]*len(m)
        scores = np.zeros((len(n), len(m)))

        for i,v in enumerate(n):
            img = caffe.io.load_image(v)
            net.blobs['data'].data[...] = transformer.preprocess('data', img)
            out = net.forward()
            feat = net.blobs[layer].data.copy()
            nfeats[i] = feat

        for i,v in enumerate(m):
            img = caffe.io.load_image(v)
            net.blobs['data'].data[...] = transformer.preprocess('data', img)
            out = net.forward()
            feat = net.blobs[layer].data.copy()
            mfeats[i] = feat

        for i, fn in enumerate(nfeats):
            for j, fm in enumerate(mfeats):
                scores[i,j] = cosine_similarity(fn, fm)[0,0]

        min_ = np.min(scores, axis=1)
        max_ = np.max(scores, axis=1)
        scores = (scores - min_.reshape(-1, 1)) / (max_ - min_).reshape(-1, 1)

        return scores

    return compute_similarity

@load_network
def decision(A,B):
    pass

if __name__ == '__main__':
    root = 'images-no-align'
    imgs = os.listdir(root)
    imgs = map(lambda e: os.path.join(root, e), imgs)

    output = decision(imgs, imgs)
    print output
