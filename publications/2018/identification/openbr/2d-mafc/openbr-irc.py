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

import cPickle as pkl

import numpy as np
import psyphy.perturb as perturb
import psyphy.n2nmatch as n2n

import openbrdec as brd

if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise Exception('missing ptype or range value')
    ptype = sys.argv[1]
    low = float(sys.argv[2])
    high = float(sys.argv[3])
    with open(os.path.join('openbr-pv.pkl'), 'rb') as f:
        sobj = pkl.load(f)

    pfunc = {
        'gausblur' : perturb.gaussian_blur,
        'linocc' : perturb.linear_occlude,
        'brightness-dec' : perturb.brightness,
        'brightness-inc' : perturb.brightness,
        'color-dec' : perturb.color,
        'color-inc' : perturb.color,
        'contrast-dec' : perturb.contrast,
        'contrast-inc' : perturb.contrast,
        'sharpness-dec' : perturb.sharpness,
        'sharpness-inc' : perturb.sharpness,
        'noisesap' : perturb.noise_sap,
        'noisegaus' : perturb.noise_gaussian,
        'noisepink' : perturb.noise_pink,
        'noisebrown' : perturb.noise_brown
    }[ptype]
    output = n2n.item_response_curve_mafc(brd.decision, pfunc, sobj['imgs'], sobj['thresh'], 200, low, high)
    with open(os.path.join('openbr-irc-' + ptype + '.pkl'), 'w') as f:
        pkl.dump(output, f)
