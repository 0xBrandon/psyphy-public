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

import copy
import math
import os.path
import random

import cv2
import numpy as np
import hyperopt

import psyphy.utils

@psyphy.utils.static_vars(lower=[0.0,True],upper=[1.0,True])
def linear_occlude(ipath, percent):
    if percent > linear_occlude.upper[0] or percent < linear_occlude.lower[0]:
        raise psyphy.utils.OutOfBoundsError('Linear Occlusion percent must be >= 0 and <= 1.')

    img = cv2.imread(ipath)
    if img is None:
        raise IOError('[Errno 2] No such file or directory ' + ipath)

    rpath = psyphy.utils.upath(os.path.splitext(ipath)[1])

    if percent == 0:
        pass
    elif percent == 1:
        img *= 0
    else:
        h,w,c = img.shape
        bb = [[0,0],[w-1,h-1]]
        occluder_color=(0,0,0)
        per = percent

        flip = random.choice([True,False])
        bb_original = copy.deepcopy(bb)
        if flip:
            y0 = (h-1)-bb[1][1]
            y1 = (h-1)-bb[0][1]
            bb[0][1] = y0
            bb[1][1] = y1

        pbox = (random.randint(bb[0][0],bb[1][0]-1),random.randint(bb[0][1],bb[1][1]-1))
        slope = 0
        while slope == 0: # prevents a zero slope
            slope = random.expovariate(1)

        def sort(points):
            cx = sum(zip(*points)[0])*1.0 / len(points)
            cy = sum(zip(*points)[1])*1.0 / len(points)
            return sorted(points, key=lambda o: math.atan2(o[1]-cy,o[0]-cx))

        def polygon(p,m,d):
            b = p[1]-m*p[0]
            right = [w*3, m*w*3+b]
            left = [-w*3, m*-w*3+b]
            theta = math.atan(-1.0/(-m))

            lcorners = [
                        [d*math.cos(theta)+left[0],d*math.sin(theta)+left[1]],
                        [d*-math.cos(theta)+left[0],d*-math.sin(theta)+left[1]]
                        ]
            rcorners = [
                        [d*math.cos(theta)+right[0],d*math.sin(theta)+right[1]],
                        [d*-math.cos(theta)+right[0],d*-math.sin(theta)+right[1]]
                        ]
            lm = (lcorners[1][1] - lcorners[0][1]) / (lcorners[1][0] - lcorners[0][0])
            rm = (rcorners[1][1] - rcorners[0][1]) / (rcorners[1][0] - rcorners[0][0])

            if rm > 0:
                tcorners = copy.deepcopy(rcorners)
                tcorners[0][1] = rcorners[1][1]
                tcorners[1][1] = rcorners[0][1]
                rcorners = copy.deepcopy(tcorners)
            if lm > 0:
                tcorners = copy.deepcopy(lcorners)
                tcorners[0][1] = lcorners[1][1]
                tcorners[1][1] = lcorners[0][1]
                lcorners = copy.deepcopy(tcorners)

            output = []
            def create_corner(p,q,m):
                # conditions for the left side
                # pq below sw corner (fail)
                # pq crosses sw corner
                # pq crosses west edge
                # pq crosses nw corner
                # pq crosses north edge
                # pq crosses ne corner
                # pq above ne corner (fail)
                # pq crosses nw-ne
                # pq crosses nw-sw
                # pq crosses pq crosses sw-nw-ne

                output = []

                if p[0] < q[0]: # put p above q
                    p,q = q,p
                pb = p[1]-m*p[0]
                qb = q[1]-m*q[0]
                px_y0 = -pb/m
                qx_y0 = -qb/m

                if pb >= h and qb >= h:
                    # pq below sw corner
                    pass
                elif pb >= 0 and pb < h and qb >= h:
                    # pq crosses sw corner
                    output.append([0,pb])
                    output.append([0,h-1])
                elif pb >= 0 and pb < h and qb >= 0 and qb < h:
                    # pq crosses west edge
                    output.append([0,pb])
                    output.append([0,qb])
                elif pb < 0 and qb >= 0 and qb < h and px_y0 >= 0 and px_y0 < w:
                    # pq crosses nw corner
                    output.append([px_y0,0])
                    output.append([0,0])
                    output.append([0,qb])
                elif pb < 0 and qb < 0:
                    if px_y0 >= 0 and px_y0 < w and qx_y0 >= 0 and qx_y0 < w:
                        # pq crosses north edge
                        output.append([px_y0,0])
                        output.append([qx_y0,0])
                    elif px_y0 >= w and qx_y0 >= 0 and qx_y0 < w:
                        # pq crosses ne corner
                        output.append([qx_y0,0])
                        output.append([w-1,0])
                    elif px_y0 >= w and qx_y0 >= w:
                        # pq above ne corner
                        pass
                elif pb < 0 and qb >= 0 and qb < h and px_y0 >= w:
                    # pq crosses nw-ne
                    output.append([0,qb])
                    output.append([0,0])
                    output.append([w-1,0])
                elif pb < 0 and qb >= h and px_y0 >= 0 and px_y0 < w:
                    # pq crosses nw-sw
                    output.append([px_y0,0])
                    output.append([0,0])
                    output.append([0,h-1])
                elif pb < 0 and qb >= h and px_y0 >= w:
                    # pq crosses sw-nw-ne
                    output.append([0,h-1])
                    output.append([0,0])
                    output.append([w-1,0])
                return output

            # these asserts require an image size of w=500,h=375 minimum
            # assert create_corner([-10,h+10],[-20,h+20],1) == [], 'pq below sw corner'
            # assert sort(create_corner([-10,h-20],[-20,h+5],1)) == sort([[0,h-10],[0,h-1]]), 'pq crosses sw corner'
            # assert sort(create_corner([-10,h-50],[-20,h-30],1)) == sort([[0,h-40],[0,h-10]]), 'pq crosses west edge'
            # assert sort(create_corner([0,-50],[-20,50],1)) == sort([[50,0],[0,0],[0,70]]), 'pq crosses nw corner'
            # assert sort(create_corner([0,-50],[20,-70],1)) == sort([[50,0],[90,0]]), 'pq crosses north edge'
            # assert sort(create_corner([w-20,-10],[w+20,-30],1)) == sort([[w-10,0],[w-1,0]]), 'pq crosses ne corner'
            # assert create_corner([w,-10],[w+10,-20],1) == [], 'pq above ne corner'
            # assert sort(create_corner([-50,0],[w,-w-50],1)) == sort([[0,50],[0,0],[w-1,0]]), 'pq crosses nw-ne'
            # assert sort(create_corner([0,-50],[-h-50,h],1)) == sort([[50,0],[0,0],[0,h-1]]), 'pq crosses nw-sw'
            # assert sort(create_corner([-2*w-50,h],[w,-2*h-50],1)) == sort([[w-1,0],[0,0],[0,h-1]]), 'pq crosses sw-nw-ne'

            new_lcorners = create_corner(lcorners[0],lcorners[1],m)
            new_rcorners = map(lambda e: [-e[0]+(w-1),-e[1]+(h-1)],rcorners)
            new_rcorners = create_corner(new_rcorners[0],new_rcorners[1],m)
            new_rcorners = map(lambda e: [-e[0]+(w-1),-e[1]+(h-1)],new_rcorners)

            for p in new_lcorners:
                output.append(p)
            for p in new_rcorners:
                output.append(p)
            output = map(lambda e0: map(lambda e1: int(round(e1)), e0),output)
            output = list(tuple(output))
            return np.array(sort(output))

        def draw_poly(poly):
            timg = img.copy()
            if flip:
                timg = cv2.flip(timg,0)
                cv2.fillConvexPoly(timg, poly, occluder_color)
                timg = cv2.flip(timg,0)
            else:
                cv2.fillConvexPoly(timg, poly, occluder_color)
            return timg

        pimg = img[bb[0][1]:bb[1][1]+1, bb[0][0]:bb[1][0]+1]
        total = pimg.shape[0]*pimg.shape[1]
        goal = int(round(per*total))
        del pimg

        def objective(d):
            poly = polygon(pbox,slope,d)
            timg = draw_poly(poly)
            b, g, r = cv2.split(timg[bb[0][1]:bb[1][1]+1, bb[0][0]:bb[1][0]+1])
            count = total - max(cv2.countNonZero(b), cv2.countNonZero(g), cv2.countNonZero(r))
            return abs(goal-count)

        bd = math.sqrt((bb[1][1]-bb[0][1])*(bb[1][1]-bb[0][1])+(bb[1][0]-bb[0][0])*(bb[1][0]-bb[0][0]))

        # hyperopt search
        space = hyperopt.hp.uniform('d',0.01,2*bd)
        best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=100)
        img = draw_poly(polygon(pbox,slope,best['d']))

    cv2.imwrite(rpath, img)

    return rpath

#     # beta is 0 => white
#     # beta is -1 => pink
#     # beta is -2 => red
def _random_noise(ipath, beta, percent):
    img = cv2.imread(ipath)
    if img is None:
        raise IOError('[Errno 2] No such file or directory ' + ipath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]

    u1 = np.arange(0, int(np.ceil(w / 2.)) + 1)
    u2 = -np.arange(int(np.floor(w / 2)) - 1, 0, -1)
    u = np.append(u1, u2) / float(w)
    u = np.expand_dims(u, 0)
    u = np.repeat(u, h, axis=0)

    v1 = np.arange(0, int(np.ceil(h / 2.)) + 1)
    v2 = -np.arange(int(np.floor(h / 2)) - 1, 0, -1)
    v = np.append(v1, v2) / float(h)
    v = np.expand_dims(v, 1)
    v = np.repeat(v, w, axis=1)

    S_f = (np.square(u) + np.square(v)) ** (beta / 2.)
    S_f[S_f == np.inf] = 0
    S_f[S_f == np.nan] = 0

    phi = np.random.uniform(0, 1, (h, w))

    real_part = np.cos(2 * np.pi * phi)
    imag_part = np.sin(2 * np.pi * phi)
    part = real_part + 1j * imag_part
    phi = (S_f ** 0.5 * part)

    x = np.fft.ifft2(phi)
    x = np.real(x)

    #[0, 1]
    x -= np.min(x)
    x /= np.max(x)

    x = np.expand_dims(x, 2)
    x = np.repeat(x, 3, axis=2)

    # add it to the image:
    img = np.float32(img)
    noised = x * percent * img + img
    noised[noised > 255] = 255
    noised[noised < 0] = 0
    noised = np.uint8(noised)
    noised = cv2.cvtColor(noised, cv2.COLOR_RGB2BGR)

    return noised

@psyphy.utils.static_vars(lower=[0.0,True],upper=[1.0,True])
def noise_pink(ipath, percent):
    if percent > noise_pink.upper[0] or percent < noise_pink.lower[0]:
        raise psyphy.utils.OutOfBoundsError('Pink Noise percent must be >= 0 and <= 1.')
    rpath = psyphy.utils.upath(os.path.splitext(ipath)[1])

    img = _random_noise(ipath, -1, percent)

    cv2.imwrite(rpath, img)
    return rpath

@psyphy.utils.static_vars(lower=[0.0,True],upper=[1.0,True])
def noise_brown(ipath, percent):
    if percent > noise_brown.upper[0] or percent < noise_brown.lower[0]:
        raise psyphy.utils.OutOfBoundsError('Brown Noise percent must be >= 0 and <= 1.')
    rpath = psyphy.utils.upath(os.path.splitext(ipath)[1])

    img = _random_noise(ipath, -2, percent)

    cv2.imwrite(rpath, img)
    return rpath
