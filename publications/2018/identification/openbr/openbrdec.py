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
import shutil
import subprocess
import uuid

import numpy as np
from sklearn import preprocessing

# Function assumes N-to-N meaning it forms an eigen matrix
def _openbr(func):
    base = ['br', '-algorithm', 'FaceRecognition', '-enrollAll', '-enroll']
    def copier(imgs, root):
        os.mkdir(root)

        totmp = {} # to temp path
        toimg = {} # to original path

        for img in imgs:
            tmp = root + '/' + str(uuid.uuid4()) + '.' + img.split('.')[-1]
            shutil.copy(img, tmp)
            totmp[img] = tmp
            toimg[tmp] = img

        return totmp, toimg

    def output(n,m):
        nname = '/tmp/' + str(uuid.uuid4())
        mname = '/tmp/' + str(uuid.uuid4())

        ntotmp, ntoimg = copier(n, nname)
        mtotmp, mtoimg = copier(m, mname)

        ngal = nname + '.gal'
        mgal = mname + '.gal'

        with open(os.devnull, 'w') as FNULL:
            subprocess.call(' '.join(base + [nname, ngal]), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
            subprocess.call(' '.join(base + [mname, mgal]), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

        shutil.rmtree(nname)
        shutil.rmtree(mname)

        csv = '/tmp/' + str(uuid.uuid4()) + '.csv'
        if os.path.exists(ngal) and os.path.exists(mgal):
            with open(os.devnull, 'w') as FNULL:
                subprocess.call(' '.join(['br', '-algorithm', 'FaceRecognition', '-compare', mgal, ngal, csv]), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

            with open(csv, 'r') as f:
                data = f.read().strip().split('\n')
                data = map(lambda e: e.split(','), data)

            hdata = {}
            for ri in xrange(1, len(data)):
                data[ri][0] = ntoimg[data[ri][0]]
                hdata[data[ri][0]] = {}
                for ci in xrange(1, len(data[0])):
                    hdata[data[ri][0]][mtoimg[data[0][ci]]] = float(data[ri][ci])

            nids = sorted(hdata.keys())
            mids = sorted(hdata[hdata.keys()[0]].keys())

            scores = np.zeros((len(n), len(m)))
            for i, nid in enumerate(n):
                if nid in hdata:
                    min_ = min(hdata[nid].values())
                    max_ = max(hdata[nid].values())
                    for j, mid in enumerate(m):
                        if mid in hdata[nid]:
                            scores[i][j] = hdata[nid][mid]
                        else:
                            # This marks all the missing templates wrong
                            if i != j:
                                scores[i,j] = max_ + 0.000001
                            else:
                                scores[i,j] = min_ - 0.000001
                else:
                    # This marks all the missing templates wrong
                    scores[i] = np.ones(len(m))
                    scores[i][i] = 0

            min_ = np.min(scores, axis=1)
            max_ = np.max(scores, axis=1)
            scores = (scores - min_.reshape(-1, 1)) / (max_ - min_).reshape(-1, 1)

            os.remove(mgal)
            os.remove(ngal)
            os.remove(csv)
        else:
            scores = 1 - np.eye(len(n), dtype=np.float32)
            if os.path.exists(mgal): os.remove(mgal)
            if os.path.exists(ngal): os.remove(ngal)

        return scores
    return output

@_openbr
def decision(A,B):
    pass


if __name__ == '__main__':
    root = 'images-no-align'
    imgs = os.listdir(root)
    imgs = map(lambda e: os.path.join(root, e), imgs)

    output = decision(imgs, imgs)
    print output
