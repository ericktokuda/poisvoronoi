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
import smopy
import fiona
from shapely import geometry
from descartes import PolygonPatch


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

def line_intersection(line1, line2):
    # https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def get_crossing_point_rectangle(v0, alpha, orient, encbox):
    mindist = 999999999

    for c in encbox:
        i = 0 if c % 2 == 0  else 1 # xmin, ymin, xmax, ymax
        d = (v0[i] - c)
        d = d / alpha[i] * orient
        if d < 0: continue
        if d < mindist: mindist = d

    p = v0 + orient * alpha * mindist
    return p

def plot_hospitals_voronoi(regionpolygon):
    df = pd.read_csv('data/rphospitals.csv')
    points = df[['lat', 'lon']].to_numpy()
    vor = spatial.Voronoi(df[['lat', 'lon']])

#########################################################
    
    # Plot seeds (points) and voronoi vertices
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.plot(vor.vertices[:, 0], vor.vertices[:, 1], '*')
    plt.xlim(-21.50, -21.00);
    plt.ylim(-47.95, -47.65)

    encbox = [-21.22, -47.86, -21.10, -47.74]

    # Plot finite edges
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-')

    # Plot "infinite" ridges
    center = points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        # print(pointidx, simplex)
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0] # finite end Voronoi vertex
            t = points[pointidx[1]] - points[pointidx[0]]  # tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]]) # normal
            midpoint = points[pointidx].mean(axis=0)
            orient = np.sign(np.dot(midpoint - center, n))
            far_point = vor.vertices[i] +  orient* n * 1
            far_point_clipped = get_crossing_point_rectangle(vor.vertices[i],
                                                             n,
                                                             orient,
                                                             encbox)
            # print(far_point_clipped)
            plt.plot([vor.vertices[i,0], far_point_clipped[0]],
                     [vor.vertices[i,1], far_point_clipped[1]], 'k--')
            # plt.plot([vor.vertices[i,0], far_point[0]],
                     # [vor.vertices[i,1], far_point[1]], 'k--')
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
        # plt.plot(poly)
        # polygon_shape = patches.Polygon(points, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(poly)
        # ax.plot(*poly.exterior.xy);
        # ax.axis("auto")
        # break

    # plt.plot(polygons)
    # plt.show()
    # return
    # ax = m.show_mpl(figsize=(12, 8))
    # fig, ax = plt.subplots(1, 1)
    ax.plot(*regionpolygon.exterior.xy);

    # ax.add_collection(
        # mpl.collections.PolyCollection(
            # cells, 
            # edgecolors='k', facecolor=(0, 0, 0, 0), linewidths=3))

    plt.show()

def load_map():
    shape = fiona.open('data/rp_map.shp')
    b = shape.next()
    p = b['geometry']['coordinates'][0]
    # print(b['coordinates'])
    x = [z[0] for z in p ]
    y = [z[1] for z in p ]
    poly = geometry.Polygon(p)
    # plt.plot(*poly.exterior.xy);
    # plt.show()
    return poly
    

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.INFO)
    regionpolygon = load_map()
    plot_hospitals_voronoi(regionpolygon)

if __name__ == "__main__":
    main()
