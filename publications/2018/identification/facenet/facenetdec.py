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

import os, sys
root = './facenet'

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

import facenet

def load_network(func):
    base = './facenet'

    def compute_similarity(n, m):
        nfeats = [0]*len(n)
        mfeats = [0]*len(m)
        scores = np.zeros((len(n), len(m)))

        nimgs = np.zeros((len(n), 160, 160, 3), dtype=np.uint8)
        for i,v in enumerate(n):
            img = cv2.imread(v)
            nimgs[i] = cv2.resize(img, (160, 160), interpolation=cv2.INTER_LINEAR)

        mimgs = np.zeros((len(m), 160, 160, 3), dtype=np.uint8)
        for i,v in enumerate(m):
            img = cv2.imread(v)
            mimgs[i] = cv2.resize(img, (160, 160), interpolation=cv2.INTER_LINEAR)

        with tf.Graph().as_default():

            with tf.Session() as sess:

                # Load the model
                facenet.load_model(os.path.join('20170512-110547.pb'))

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                # Run forward pass to calculate embeddings
                nfeed_dict = { images_placeholder: nimgs, phase_train_placeholder:False }
                nfeats = sess.run(embeddings, feed_dict=nfeed_dict)
                mfeed_dict = { images_placeholder: mimgs, phase_train_placeholder:False }
                mfeats = sess.run(embeddings, feed_dict=mfeed_dict)

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
