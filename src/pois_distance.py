#!/usr/bin/env python3
"""Calculate de distance between pair of pois
"""

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from myutils import info, create_readme, graph

##########################################################
def plot_count_pois(g, outdir):
    """Plot the count of each tagvalue"""
    info(inspect.stack()[0][3] + '()')
    tagvalues = np.array(g.vs['tagValue'])
    aux1, aux2 = np.unique(tagvalues, return_counts=True)
    delidx = np.where(aux1 == 'none')[0]
    aux1 = np.delete(aux1, delidx)
    aux2 = np.delete(aux2, delidx)

    inds = np.argsort(aux2)
    plt.figure(figsize=(12, 12))
    for i in range(np.min(aux2), np.max(aux2)):
        if i % 5 == 0: plt.axhline(y=i, color='lightgray', linewidth=1, zorder=1)
    plt.bar(aux1[inds], aux2[inds], zorder=5)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(pjoin(outdir, 'counts.pdf'))

##########################################################
def calculate_distances(g, pois, weighted, outdir):
    """Calculate distance between pair of POIs"""
    info(inspect.stack()[0][3] + '()')
    tagvalues = np.array(g.vs['tagValue'])

    weights = 'length' if weighted else None
    poiinds = {}
    for k, v in pois.items():
        poiinds[k] = []
        for p in v:
            aux = np.where(tagvalues == p)
            poiinds[k].extend(aux[0])

    combs = itertools.combinations(list(pois.keys()), 2)

    wlens = {}
    for poi1, poi2 in combs:
        aux = g.shortest_paths(poiinds[poi1], poiinds[poi2], weights=weights)
        aux = np.array(aux).flatten()
        wlens[(poi1, poi2)] = aux
    return wlens

##########################################################
def plot_distance_distrib(lens, poicount, outdir):
    """Plot distance distribs"""
    info(inspect.stack()[0][3] + '()')

    W = 640; H = 480
    for k, v in lens.items():
        plt.close()
        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        ax.hist(v)
        ax.set_xlim(0, 35)
        ax.set_xlabel('Distance')
        ax.set_title('{}:{}, {}:{}'.format(k[0], poicount[k[0]],
                                           k[1], poicount[k[1]],))
        plt.savefig(pjoin(outdir, '{}_{}.png'.format(k[0], k[1])))

##########################################################
def count_pois(g, pois):
    """Count the pois"""
    info(inspect.stack()[0][3] + '()')
    tagvalues = np.array(g.vs['tagValue'])

    poicount = {}
    for k, v in pois.items():
        acc = 0
        for p in v:
            aux = np.where(tagvalues == p)
            acc += len(aux[0])
        poicount[k] = acc
    return poicount
##########################################################
def randomly_move_pois(g, poicount):
    """Randomly change the poi (@pois) locations in @g """
    info(inspect.stack()[0][3] + '()')

    m = np.sum(list(poicount.values()))

    from numpy.random import default_rng
    rng = default_rng()

    ids = rng.choice(g.vcount(), size=m, replace=False)

    tagvalues = np.array(['none'] * g.vcount(), dtype=object)

    idx = 0
    for k, v in poicount.items():
        tagvalues[ids[idx:idx+v]] = k
        idx += v

    g.vs['tagValue'] = tagvalues.tolist()
    return g

##########################################################
def discard_vertex_attribs(g, but=['tagValue']):
    """Discard all attributes but tagValue in @g"""
    info(inspect.stack()[0][3] + '()')
    for attr in g.vertex_attributes():
        if not attr in but: del g.vs[attr]
    return g

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graphml', help='Graphml path')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(pjoin(args.outdir, 'weighted'), exist_ok=True)
    # os.makedirs(pjoin(args.outdir, 'unweighted'), exist_ok=True)
    os.makedirs(pjoin(args.outdir, 'shuffled'), exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    nrealizations = 10

    pois = {
        'hospital': ['hospital'],
        'school': ['school'],
        'supermarket': ['supermarket'],
        'theatre': ['theatre', 'theater', 'cinema'],
    }

    g = graph.simplify_graphml(args.graphml) #Just connected component
    g = discard_vertex_attribs(g)

    plot_count_pois(g, args.outdir)
    poicount = count_pois(g, pois)

    # lens = calculate_distances(g, pois, False, args.outdir)
    # plot_distance_distrib(lens, pjoin(args.outdir, 'unweighted'))

    wlens = calculate_distances(g, pois, True, args.outdir)
    plot_distance_distrib(wlens, poicount, pjoin(args.outdir, 'weighted'))


    shufmeans = {}; shufall = {}
    combs = itertools.combinations(list(pois.keys()), 2)
    for c in combs: shufmeans[c] = []; shufall[c] = []

    for i in range(nrealizations):
        gnew = randomly_move_pois(g.copy(), poicount)
        wlens = calculate_distances(gnew, pois, True, args.outdir)
        for k, v in wlens.items():
            shufmeans[k].append(np.mean(v))
            shufall[k].extend(v)

    randpoicount = poicount # Random poi locations
    for k in addedpoicount: addedpoicount[k] *= nrealizations

    plot_distance_distrib(shufall, randpoicount, pjoin(args.outdir, 'shuffled'))

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
