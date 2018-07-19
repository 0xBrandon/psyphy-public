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
import skimage.io
import skimage.util

import psyphy.utils

@psyphy.utils.static_vars(lower=[0.0,True],upper=[1.0,True])
def noise_sap(ipath, percent):
    if percent > noise_sap.upper[0] or percent < noise_sap.lower[0]:
        raise psyphy.utils.OutOfBoundsError('Noise S&P percent must be >= 0 and <= 1.')

    img = skimage.io.imread(ipath)
    img = skimage.util.random_noise(img, mode='s&p', amount=percent)
    rpath = psyphy.utils.upath(os.path.splitext(ipath)[1])
    skimage.io.imsave(rpath, img)

    return rpath

@psyphy.utils.static_vars(lower=[0.0,True],upper=[None,False])
def noise_gaussian(ipath, variance):
    if variance < noise_gaussian.lower[0]:
        raise psyphy.utils.OutOfBoundsError('Noise Gaussian variance must be >= 0')

    img = skimage.io.imread(ipath)
    if variance > 0:
        img = skimage.util.random_noise(img, mode='gaussian', var=variance)
    rpath = psyphy.utils.upath(os.path.splitext(ipath)[1])
    skimage.io.imsave(rpath, img)

    return rpath
