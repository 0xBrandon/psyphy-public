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

import numpy as np

'''
    This is the MAFC item-response curve function described in
    Visual Psychophysics for Making Face Recognition Algorithms More Explainable
    RichardWebster et al.

    Description:
    http://www.bjrichardwebster.com/papers/menagerie/pdf
'''
def item_response_curve_mafc(dec, pert, imgs, thresh, num, lower, upper, step='log', keep=False):
    num = max(num, 2)
    if step == 'log':
        if upper <= 0 or lower <= 0:
            nup = upper + abs(min(upper,lower))+1
            nlow = lower + abs(min(upper,lower))+1
            perts = np.logspace(np.log10(nup), np.log10(nlow), num) - (abs(min(upper,lower))+1)
        else:
            perts = np.logspace(np.log10(upper), np.log10(lower), num)
    else:
        raise NotImplementedError()

    if keep:
        raise NotImplementedError('saving each individual image response')

    all_ = []
    match = []
    nonmatch = []
    mmask = np.eye(len(imgs), dtype=np.bool)
    nmmask = np.logical_not(mmask)
    for k, p in enumerate(perts):
        gimgs = map(lambda e: pert(e, p), imgs)
        scores = dec(imgs, gimgs)

        for img in gimgs: os.remove(img)
        threshed = (scores >= thresh)
        mcount = np.count_nonzero(np.logical_and(mmask, threshed))
        match.append(float(mcount) / np.count_nonzero(mmask))
        nmcount = np.count_nonzero(np.logical_and(nmmask, np.logical_not(threshed)))
        nonmatch.append(float(nmcount) / np.count_nonzero(nmmask))
        all_.append(float(mcount + nmcount) / len(imgs)**2)

    return np.array([perts, all_, match, nonmatch]).transpose()

'''
    This is the 20AFC item-response curve function described in
    Visual Psychophysics for Making Face Recognition Algorithms More Explainable
    RichardWebster et al.

    Description:
    http://www.bjrichardwebster.com/papers/menagerie/pdf
'''
def item_response_curve_20afc(dec, pert, imgs, thresh, num, lower, upper, step='log', keep=False):
    num = max(num, 2)
    if step == 'log':
        if upper <= 0 or lower <= 0:
            nup = upper + abs(min(upper,lower))+1
            nlow = lower + abs(min(upper,lower))+1
            perts = np.logspace(np.log10(nup), np.log10(nlow), num) - (abs(min(upper,lower))+1)
        else:
            perts = np.logspace(np.log10(upper), np.log10(lower), num)
    else:
        raise NotImplementedError()

    if keep:
        raise NotImplementedError('saving each individual image response')

    all_ = []
    match = []
    nonmatch = []
    mmask = np.eye(20, dtype=np.bool)
    nmmask = np.logical_not(mmask)
    imgs = np.array(imgs)
    for k, p in enumerate(perts):
        simgs = imgs[np.random.random_integers(0,len(imgs)-1,size=mmask.shape[0])]
        gimgs = map(lambda e: pert(e, p), simgs)
        scores = dec(simgs, gimgs)

        for img in gimgs: os.remove(img)
        threshed = (scores >= thresh)
        mcount = np.count_nonzero(np.logical_and(mmask, threshed))
        match.append(float(mcount) / np.count_nonzero(mmask))
        nmcount = np.count_nonzero(np.logical_and(nmmask, np.logical_not(threshed)))
        nonmatch.append(float(nmcount) / np.count_nonzero(nmmask))
        all_.append(float(mcount + nmcount) / len(simgs)**2)

    return np.array([perts, all_, match, nonmatch]).transpose()
