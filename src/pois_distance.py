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
from myutils import info, create_readme, graph

##########################################################
def plot_counts(g, outdir):
    """Plot the count of each tagvalue"""
    info(inspect.stack()[0][3] + '()')
    tagkeys = np.array(g.vs['tagKey'])
    validids = np.array(g.vs['tagKey']) != 'none'
    tagvalues = np.array(g.vs['tagValue'])[validids]
    aux1, aux2 = np.unique(tagvalues, return_counts=True)
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
    tagkeys = np.array(g.vs['tagKey'])
    validids = np.array(g.vs['tagKey']) != 'none'
    tagvalues = np.array(g.vs['tagValue'])[validids]

    weights = 'length' if weighted else None
    poiinds = {}
    for k, v in pois.items():
        poiinds[k] = []
        for p in v:
            aux = np.where(tagvalues == p)
            poiinds[k].extend(aux[0])

    import itertools
    combs = itertools.combinations(list(pois), 2)

    wlens = {}
    for poi1, poi2 in combs:
        aux = g.shortest_paths(poiinds[poi1], poiinds[poi2], weights=weights)
        aux = np.array(aux).flatten()
        wlens[(poi1, poi2)] = aux
    return wlens

    plot_distance_distrib(lens, pjoin(args.outdir, 'unweighted'))
##########################################################
def plot_distance_distrib(lens, outdir):
    """Plot distance distribs"""
    info(inspect.stack()[0][3] + '()')

    W = 640; H = 480
    for k, v in lens.items():
        plt.close()
        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        ax.hist(v)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Distance')
        plt.savefig(pjoin(outdir, '{}_{}.png'.format(k[0], k[1])))

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
    os.makedirs(pjoin(args.outdir, 'unweighted'), exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    pois = {
        'hospital': ['hospital'],
        'school': ['school'],
        'supermarket': ['supermarket'],
        'theatre': ['theatre', 'theater', 'cinema'],
    }

    g = graph.simplify_graphml(args.graphml) #Just connected component

    plot_counts(g, args.outdir)
    lens = calculate_distances(g, pois, False, args.outdir)
    wlens = calculate_distances(g, pois, True, args.outdir)
    plot_distance_distrib(lens, pjoin(args.outdir, 'unweighted'))
    plot_distance_distrib(wlens, pjoin(args.outdir, 'weighted'))

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
