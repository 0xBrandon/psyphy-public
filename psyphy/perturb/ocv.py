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

import os.path
import cv2

import psyphy.utils

@psyphy.utils.static_vars(lower=[0.0,True],upper=[None,False])
def gaussian_blur(ipath, sigma):
    if sigma < gaussian_blur.lower[0]:
        raise psyphy.utils.OutOfBoundsError('Gaussian Blur sigma must be >= 0.')

    img = cv2.imread(ipath)
    if img is None:
        raise IOError('[Errno 2] No such file or directory ' + ipath)
    if sigma > 0:
        img = cv2.GaussianBlur(img, (0,0), sigma)
    rpath = psyphy.utils.upath(os.path.splitext(ipath)[1])
    cv2.imwrite(rpath, img)

    return rpath
