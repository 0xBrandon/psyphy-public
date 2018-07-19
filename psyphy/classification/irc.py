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
    This is the 2AFC (2D) item-response curve function described in
    PsyPhy: A Psychophysics Driven Evaluation Framework for Visual Recognition
    RichardWebster et al.

    Description:
    http://www.bjrichardwebster.com/papers/psyphy/pdf
'''
def item_response_curve_2afc(dec, pert, imgs, num, lower, upper, step='log', keep=False):
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

    accs = []
    iarr = np.arange(len(imgs))
    for k,p in enumerate(perts):
        nimgs = map(lambda e: pert(e, p), imgs)
        correct = 0
        for i in xrange(len(imgs)):
            sam = nimgs[i]
            pos = imgs[i]
            negi = np.random.choice(iarr[iarr!=i])
            neg = imgs[negi]
            d = dec(sam,pos,neg)
            if d >= 0:
                correct += 1
        accs.append(float(correct) / len(imgs))

        for img in nimgs: os.remove(img)

    return np.array([perts, accs]).transpose()


'''
    This is the MAFC (2D) item-response curve function described in
    PsyPhy: A Psychophysics Driven Evaluation Framework for Visual Recognition
    RichardWebster et al.

    Description:
    http://www.bjrichardwebster.com/papers/psyphy/pdf
'''
def item_response_curve_mafc(dec, pert, imgs, classes, num, lower, upper, step='log', keep=False):
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

    accs = []
    for k,p in enumerate(perts):
        nimgs = map(lambda e: pert(e, p), imgs)
        correct = 0
        for i in xrange(len(imgs)):
            sam = nimgs[i]
            d = dec(sam,classes[i])
            if d >= 0:
                correct += 1
        accs.append(float(correct) / len(imgs))

        for img in nimgs: os.remove(img)

    return np.array([perts, accs]).transpose()
