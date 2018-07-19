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
from PIL import Image
from PIL import ImageEnhance

import psyphy.utils

@psyphy.utils.static_vars(lower=[0.0,True],upper=[None,False])
def brightness(ipath, factor):
    if factor < brightness.lower[0]:
        raise psyphy.utils.OutOfBoundsError('Brightness factor must be >= 0.')

    img = Image.open(ipath)
    rpath = psyphy.utils.upath(os.path.splitext(ipath)[1])
    ImageEnhance.Brightness(img).enhance(factor).save(rpath)

    return rpath

@psyphy.utils.static_vars(lower=[0.0,True],upper=[None,False])
def color(ipath, factor):
    if factor < color.lower[0]:
        raise psyphy.utils.OutOfBoundsError('Color factor must be >= 0.')

    img = Image.open(ipath)
    rpath = psyphy.utils.upath(os.path.splitext(ipath)[1])
    ImageEnhance.Color(img).enhance(factor).save(rpath)

    return rpath

@psyphy.utils.static_vars(lower=[0.0,True],upper=[None,False])
def contrast(ipath, factor):
    if factor < contrast.lower[0]:
        raise psyphy.utils.OutOfBoundsError('Contrast factor must be >= 0.')

    img = Image.open(ipath)
    rpath = psyphy.utils.upath(os.path.splitext(ipath)[1])
    ImageEnhance.Contrast(img).enhance(factor).save(rpath)

    return rpath

@psyphy.utils.static_vars(lower=[0.0,True],upper=[2.0,True])
def sharpness(ipath, factor):
    if factor > sharpness.upper[0] or factor < sharpness.lower[0]:
        raise psyphy.utils.OutOfBoundsError('Sharpness factor must be >= 0 and <= 2.0.')

    img = Image.open(ipath)
    rpath = psyphy.utils.upath(os.path.splitext(ipath)[1])
    ImageEnhance.Sharpness(img).enhance(factor).save(rpath)

    return rpath
