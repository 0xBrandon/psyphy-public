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

import hyperopt
import numpy as np

'''
    This is the "shepherd" function described in
    Visual Psychophysics for Making Face Recognition Algorithms More Explainable
    RichardWebster et al.

    High level description:
    http://www.bjrichardwebster.com/papers/menagerie/pdf
    Complete and detailed description:
    http://www.bjrichardwebster.com/papers/menagerie/supp
'''
def preferred_view(dec, imgs, ithresh=None):
    scores = dec(imgs, imgs)
    scores = (scores + np.transpose(scores)) / 2

    def subset(thresh):
        tscores = scores >= thresh
        misses = np.logical_xor(np.identity(tscores.shape[0]) == 1, tscores)
        list_ = [{'conn': 0, 'neigh' : {j:False for j in range(len(imgs))}, 'kept': True} for i in range(len(imgs))]
        for i,r in enumerate(misses):
            for j,v in enumerate(r):
                if v:
                    list_[i]['conn'] += 1
                    list_[i]['neigh'][j] = True
        def remove(index):
            list_[index]['kept'] = False
            for conn in list_[index]['neigh']:
                if list_[index]['neigh'][conn]:
                    list_[index]['neigh'][conn] = False
                    list_[conn]['neigh'][index] = False
                    list_[conn]['conn'] -= 1
            list_[index]['conn'] = 0
        # order the points with highest number of connections
        rank = []
        for i,v in enumerate(list_):
            rank.append(list_[i]['conn'])
        rank = sorted(range(len(rank)), key=lambda k: rank[k], reverse=True)
        for v in rank:
            if list_[v]['conn'] < 1:
                continue
            remove(v)

        # create a list of the remaining imgs
        remaining = []
        for index, v in enumerate(list_):
            if list_[index]['kept']:
                remaining.append(imgs[index])

        return remaining

    def obj(thresh):
        loss = len(imgs)-len(subset(thresh))
        loss += (1 - thresh/0.99999) # favor fewer false match at test time
        return {'loss': loss, 'status': hyperopt.STATUS_OK}

    if ithresh is None:
        space = hyperopt.hp.uniform('thresh', 0., 1.0)

        best = hyperopt.fmin(obj, space, algo=hyperopt.tpe.suggest, max_evals=250)
        rimgs = subset(best['thresh'])
        return best['thresh'], rimgs
    else:
        rimgs = subset(ithresh)
        return ithresh, rimgs
