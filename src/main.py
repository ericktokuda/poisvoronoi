#!/usr/bin/env python3
"""Full pipeline
"""

import argparse
import logging
from os.path import join as pjoin
from logging import debug, info
import poisvoronoi
import pandas as pd
import requests
import html
import json
import xml.etree.ElementTree as ET

def get_geojson_from_query(region, country, outdir):
    """Geocoding and 

    Args:
    keyword(str): string containing the keywods we are using for searching

    Returns:
    geojson: json structure contaning the map referred to the first result of the query
    """

    info('Geocoding ...:')
    rooturl = 'https://nominatim.openstreetmap.org/'
    params = 'search.php?polygon_geojson=1&format=geojson&limit=1&q='
    query = region.replace(', ', '+').replace(',', '+').replace(' ', '+') + '+'
    query +=  country.replace(', ', '+').replace(',', '+').replace(' ', '+')
    queryurl = rooturl + params + html.escape(query)
    try:
        response = requests.get(queryurl)
    except:
        raise Exception('Could not request {}'.format(queryurl))
    gjson = response.json()
    jsonpath = pjoin(outdir, 'region.geojson')
    with open(jsonpath, 'w') as fh:
        json.dump(gjson, fh)
    return gjson

def filter_small_regions(ingjson):
    """Filter out small regions

    Args:
    gjson(geojson)

    Returns:
    ret
    """

    info('Filtering small regions ...:')
    info('TODO')
    gjson = []
    return gjson

def get_shp_from_geojson(gjson):
    """Get shapefile containing the polygon from the geojson
    Please see https://en.wikipedia.org/wiki/GeoJSON

    Args:
    gjson(dict): json structure

    Returns:
    dict: geojson containg the 
    """
    info('Extracting shapefile ...:')
    info('TODO')
    shpfile = []
    return shpfile

def get_continent_from_country(countrystr, countriespath):
    """Get continent from country

    Args:
    country(str): name of the country

    Returns:
    str, str: continent name and continent abbreviation
    """
    return '', ''

##########################################################
def parse_ways(root, roitypes):
    """Get all ways in the xml struct. We are interested in two types of ways: streets
    and regions of interest

    Args:
    root(ET): root element
    roitypes(dict): dict of dicts, with k, v as keys, where v is a list

    Returns:
    dict of list: hash of wayid as key and list of nodes as values;
    dict of list: hash of nodeid as key and list of wayids as values;
    dict of dict of list: hash of regiontype as
    """

    t0 = time.time()
    ways = {}
    invways = {}
    regions = {r: {} for r in roitypes.keys()}

    for way in root:
        if way.tag != 'way': continue

        wayid = int(way.attrib['id'])
        isstreet = False
        regiontype = None

        nodes = []
        for child in way:
            if child.tag == 'nd':
                nodes.append(int(child.attrib['ref']))
            elif child.tag == 'tag':
                for k, r in ROI.items():
                    if child.attrib['k'] == r['k'] and child.attrib['v'] in r['v']:
                        regiontype = k
                        break

        if regiontype is None: continue

        if regiontype == 'road':
            ways[wayid] = nodes
            for node in nodes: # Create inverted index of ways
                if node in invways.keys(): invways[node].append(wayid)
                else: invways[node] = [wayid]
        else:
            regions[regiontype][wayid] = nodes  # Regions of interest

    debug('Found {} ways ({:.3f}s)'.format(len(ways.keys()), time.time() - t0))
    return ways, invways, regions

def get_pois_from_osm(roispath, country):
    """Get POIS from osm file

    Args:
    osmpath(str): path to the osm file

    Returns:
    ret
    """
    info('Extracting POIs from osm file ...:')

    pois = []
    df = pd.read_json(roispath)

    countryosmpath = pjoin('./data/', country)
    # info('countryosmpath:{}'.format(countryosmpath))
    info('TODO:{}'.format(TODO))
    return pois

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('region', help='String containing the region ')
    parser.add_argument('country', help='country of the region')
    parser.add_argument('--roispath', required=False,
                        default='data/rois.json',
                        help='Path to the regions of interest')
    parser.add_argument('--outdir', default='/tmp', help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.INFO)

    gjson = get_geojson_from_query(args.region, args.country, args.outdir) # Geocoding
    gjson = filter_small_regions(gjson) # Filter small areas
    myshp = get_shp_from_geojson(gjson) # Get shapefile from geojson

    # Get dataframe of POIs
    pois = get_pois_from_osm(args.roispath, args.country)
    print(pois)
    
    

if __name__ == "__main__":
    main()

