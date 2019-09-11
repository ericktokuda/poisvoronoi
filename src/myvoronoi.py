#!/usr/bin/env python
""" Voronoi from csv
"""

import argparse
import logging
from logging import debug
import numpy as np
import pandas as pd
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib as mpl
from matplotlib import patches
import smopy
import fiona
from shapely import geometry
from descartes import PolygonPatch
import copy
from scipy.spatial import KDTree
import itertools

##########################################################
def plot_voronoi_finite_polygons_2d(vor):
    newregions, newvertices = voronoi_finite_polygons_2d(vor, radius=1)
    for region in newregions:
        polygon = newvertices[region]
        plt.fill(*zip(*polygon), alpha=0.4)

    plt.plot(vor.points[:, 0], vor.points[:, 1], 'ko')
    plt.axis('equal')
    plt.xlim(vor.min_bound[0] - 1, vor.max_bound[0] + 1)
    plt.ylim(vor.min_bound[1] - 1, vor.max_bound[1] + 1)
    plt.show()

##########################################################
def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite Voronoi regions in a 2D diagram to finite regions.
    https://stackoverflow.com/a/20678647/1595060
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()
    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points,
                                  vor.ridge_vertices):
        all_ridges.setdefault(
            p1, []).append((p2, v1, v2))
        all_ridges.setdefault(
            p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0: # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an  infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]].                 mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # Sort region counterclockwise.
        vs = np.asarray([new_vertices[v]
                         for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(
            vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[
            np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)

##########################################################
def get_crossing_point_rectangle(v0, alpha, orient, encbox):
    mindist = 999999999

    for j, c in enumerate(encbox):
        i = j % 2
        d = (c - v0[i]) # xmin, ymin, xmax, ymax
        if alpha[i] == 0: d *= orient
        else: d = d / alpha[i] * orient
        if d < 0: continue
        if d < mindist: mindist = d

    p = v0 + orient * alpha * mindist
    return p

##########################################################
def get_bounded_polygons(vor, newvorvertices, newridgevertices, encbox):
    newvorregions = copy.deepcopy(vor.regions)
    newvorregions = np.array([ np.array(f) for f in newvorregions])

    # Update voronoi regions to include added vertices and corners
    for regidx, rr in enumerate(vor.regions):
        reg = np.array(rr)
        if not np.any(reg == -1): continue
        foo = np.where(vor.point_region==regidx)
        print(foo)
        seedidx = foo[0]
        print('seeidx')
        print(seedidx)

        newvorregions[regidx] = copy.deepcopy(rr)
        # Looking for ridges bounding my point
        for ridgeid, ridgepts in enumerate(vor.ridge_points):
            # print(ridgeid, ridgepts)
            # print('asfdasdfasdfasdf')
            if not np.any(ridgepts == seedidx): continue
            ridgevs = vor.ridge_vertices[ridgeid]
            if -1 not in ridgevs: continue # I want unbounded ridges
            myidx = 0 if ridgevs[0] == -1 else 1

            # ff.append(newridgevertices[ridgeid][myidx])
            # ff.remove(-1)
            # newvorregions[regidx] = ff
            newvorregions[regidx].append(newridgevertices[ridgeid][myidx])
        if -1 in newvorregions[regidx]:  newvorregions[regidx].remove(-1)

    print('#####################')
    print(vor.regions)
    print(newvorregions)
    # Include corners
    # from scipy.spatial import KDTree
# points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                   # [2, 0], [2, 1], [2, 2]])
    tree = KDTree(vor.points)
    corners = itertools.product((encbox[0], encbox[2]), (encbox[1], encbox[3]))
    ids = []
    # input(newvorregions)
    for c in corners:
        dist, idx = tree.query(c)
        # newvorvertices.append()
        k = len(newvorvertices)
        newvorvertices = np.row_stack((newvorvertices, c))
        # ids.append(idx)
        newvorregions[vor.point_region[idx]].append(k)

    convexpolys = []
    for reg in newvorregions:
        if len(reg) == 0: continue
        # TODO: remove it
        if len(reg) < 3: continue
        points = newvorvertices[reg]
        # print(type(points))
        hull = spatial.ConvexHull(points)
        # print(hull, points)
        # input(hull.simplices)
        pp = points[hull.vertices]
        # pp = points[hull.simplices]
        convexpolys.append(pp)
    # input(convexpolys)
    return convexpolys

##########################################################
def plot_finite_ridges(vor, ax):
    """Plot the finite ridges of voronoi

    Args:
    ridge_vertices(np.ndarray):
    """
    # Plot finite edges
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            ax.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-')

##########################################################
def create_bounded_ridges(vor, encbox, ax):
    """Create bounded voronoi vertices bounded by encbox

    Args:
    vor(spatial.Voronoi): voronoi structure

    Returns:
    ret
    """

    center = vor.points.mean(axis=0)
    newvorvertices = copy.deepcopy(vor.vertices)
    newridgevertices = copy.deepcopy(vor.ridge_vertices)

    for j in range(len(vor.ridge_vertices)):
        pointidx = vor.ridge_points[j]
        simplex = vor.ridge_vertices[j]
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0] # finite end Voronoi vertex
            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]]) # normal
            # input(n)
            midpoint = vor.points[pointidx].mean(axis=0)
            orient = np.sign(np.dot(midpoint - center, n))
            far_point_clipped = get_crossing_point_rectangle(vor.vertices[i],
                                                             n,
                                                             orient,
                                                             encbox)
            ii = np.where(simplex < 0)[0][0] # finite end Voronoi vertex
            kk = newvorvertices.shape[0]
            # print(newridgevertices)
            newridgevertices[j][ii] = kk
            # print(newridgevertices)
            # print('#############################')
            newvorvertices = np.row_stack((newvorvertices, far_point_clipped))
            plt.plot([vor.vertices[i,0], far_point_clipped[0]],
                     [vor.vertices[i,1], far_point_clipped[1]], 'k--')

            ax.plot(far_point_clipped[0], far_point_clipped[1], 'og')
    return newvorvertices, newridgevertices

##########################################################
def plot_hospitals_voronoi(regionpolygon):
    df = pd.read_csv('data/sample01.csv')
    # df = pd.read_csv('data/rphospitals.csv')
    points = df[['lat', 'lon']].to_numpy()
    vor = spatial.Voronoi(df[['lat', 'lon']])

    # Plot seeds (points) and voronoi vertices
    fig, ax = plt.subplots(1, 1)

    ax.plot(points[:, 0], points[:, 1], 'o')
    ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], '*')

    # ax.set_xlim(-21.55, -20.95);
    # ax.set_ylim(-47.95, -47.45)

    # encbox = [-21.5, -47.9, -21.0, -47.5]
    encbox = [-21.45, -47.9, -21.05, -47.5]

    # plot_voronoi_finite_polygons_2d(vor)

    plot_finite_ridges(vor, ax)
    # plt.show()
    # input()
    newvorvertices, newridgevertices = create_bounded_ridges(vor, encbox, ax)
    # print(vor.vertices, newvorvertices)
    # print(vor.ridge_vertices, newridgevertices)
    # print(vor.regions, newvorregions)
    # plt.show()
    # input('wait')

    # spatial.voronoi_plot_2d(vor)
    rect = patches.Rectangle((encbox[0], encbox[1]),
                             encbox[2]-encbox[0],
                             encbox[3]-encbox[1],
                             linewidth=1,edgecolor='r',facecolor='none')

    ax.add_patch(rect)
    print(newvorvertices)
    print(newridgevertices)
    polys = get_bounded_polygons(vor, newvorvertices, newridgevertices, encbox)
    print(polys)

    for p in polys:
        pgon = plt.Polygon(p, color=np.random.rand(3,), alpha=0.5)
        ax.add_patch(pgon)
    plt.show()
    return
#########################################################
    # plt.show()
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # box = (df.lat.min(), df.lon.min(),
           # df.lat.max(), df.lon.max())
    # m = smopy.Map(box, z=11)
    # cells = [m.to_pixels(vertices[region])
             # for region in regions]
    # input(regions)
    # input(len(vertices))
    # input((vertices[0]))
    # input(type(vertices))

    fig, ax = plt.subplots(1,1)
    polygons = []
    for r in regions:
        p = vertices[r]
        poly = geometry.Polygon(p)

        plt.plot(poly)
        # polygon_shape = patches.Polygon(points, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(poly)
        # ax.plot(*poly.exterior.xy);
        # ax.axis("auto")
        # break

    # plt.plot(polygons)
    plt.show()
    # return
    # ax = m.show_mpl(figsize=(12, 8))
    # fig, ax = plt.subplots(1, 1)
    ax.plot(*regionpolygon.exterior.xy);

    # ax.add_collection(
        # mpl.collections.PolyCollection(
            # cells, 
            # edgecolors='k', facecolor=(0, 0, 0, 0), linewidths=3))

    plt.show()

##########################################################
def load_map(shp):
    shape = fiona.open(shp)
    b = next(iter(shape))
    p = b['geometry']['coordinates'][0]
    x = [z[0] for z in p ]
    y = [z[1] for z in p ]
    poly = geometry.Polygon(p)
    return poly
    
##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shp', help='Map in shapefile')
    parser.add_argument('pois', help='POIs in csv')
    # parser.add_argument('--outdir', required=True, help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    regionpolygon = load_map(args.shp)
    plot_hospitals_voronoi(regionpolygon, args.csv)

if __name__ == "__main__":
    main()
