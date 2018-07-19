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

import cPickle as pkl

import psyphy.n2nmatch as n2n

import openbrdec as brd

if __name__ == '__main__':
    root = os.path.join('images-facegen', 'no-align', 'original')
    imgs = os.listdir(root)
    imgs = map(lambda e: os.path.join(root, e), imgs)

    thresh, rem_imgs = n2n.preferred_view(brd.decision, imgs)
    sobj = {    'thresh' : thresh,
                'imgs' : rem_imgs}
    with open(os.path.join('openbr-pv.pkl'), 'w') as f:
        pkl.dump(sobj, f)

    print thresh, len(rem_imgs)
