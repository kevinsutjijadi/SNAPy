### SNAPy (Spatial Network Analysis Python)
# geometry related processing core
# Kevin Sutjijadi @ 2023

__author__ = "Kevin Sutjijadi"
__copyright__ = "Copyright 2023, Kevin Sutjijadi"
__credits__ = ["Kevin Sutjijadi"]
__version__ = "0.1.0"

"""
Spatial Network Analysis (SNA) module
using networkx, shapely, and geopandas, for network analysis simulations 
with more control over spatial items
"""

import math as mt
from multiprocessing import Pool
from time import time

# importing dependent libraries
import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point, mapping, shape
from shapely.ops import nearest_points

# importing internal scripts


# functions
def geom_pointtoline(point:Point, lineset:list):
    """
    geom_pointtoline(point:tuple, lineset:list, sweep_rad:float=300, sweep_res:int=10)\n
    calculates nearest terms from a point to a set of lines\n
    returns nrstln, nrstpt, nrstdst\n
    nearest line index, intersection point, intersection distance\n
    """
    intrpt = []
    intrdst = []
    for ln in lineset:
        nearpt = nearest_points(ln, point)
        # neardist = shl.distance(nearpt, point)
        neardist = point.distance(nearpt[0])
        intrpt.append(nearpt[0])
        intrdst.append(neardist)
    if len(intrpt) == 0:
        return None, None, None
    nrstln = intrdst.index(min(intrdst))
    nrstpt = intrpt[nrstln]
    nrstdst = min(intrdst)
    return nrstln, nrstpt, nrstdst


def geom_linesplit(ln, point:Point):
    """
    geom_linesplit(line:shapely.MultiLineString/LineString, point:shapely.point)\n
    Splitting line at an intersecting point\n
    returns tuple of 2 MultiLineString, always the beginning and the end based on the line direction\n
    """
    ln = list(ln)[0]
    if ln.distance(point) > 1e-8:
        return None
    coortp = ln.coords
    j = None
    for i in range(len(coortp) - 1):
        if LineString(coortp[i:i+2]).distance(point) < 1e-8:
            j = i
            break
    assert j is not None
    if j == 0:
        lnspl = (None, ln)
    elif j == len(coortp)-2:
        lnspl = (ln, None)
    elif Point(coortp[j + 1]).equals(point):
        lnspl = (LineString(coortp[:j + 2]), LineString(coortp[j + 1:]))
    else:
        lnspl = (LineString(coortp[:j + 1] + [(point.x, point.y)]), LineString([(point.x, point.y)]+ coortp[j + 1:]))
    return lnspl


def geom_closestline(point:Point, lineset:gpd.GeoDataFrame, searchlim:float=300, AttrID:str='FID'):
    """
    geom_closestline(point:gpd.GeoDataFrame.geometry, lineset:gpd.GeoDataFrame, searchlim:float=200)\n
    Calculating closest line to a point\n
    returns lineid, point, and distance to entry point\n
    search limit is set to 200\n
    """
    # filter by box dimensions
    lnflt = []
    lnfid = []
    # plim = (point[0]-searchlim, point[0]+searchlim, point[1]-searchlim, point[1]+searchlim)
    plim = (point.x-searchlim, point.x+searchlim, point.y-searchlim, point.y+searchlim)
    for nn, ln in enumerate(lineset.geometry):
        st = 0
        lnt = tuple(ln.coords)
        for lsg in lnt:
            st = (
                (plim[0] < lsg[0] < plim[1]) + 
                (plim[2] < lsg[1] < plim[3])
                )
            if st > 0: break
        if st > 0: # if true, get line
            lnfid.append(lineset[AttrID][nn])
            lnflt.append(ln)
    if len(lnflt) == 0:
        return None, None, None
    nrLn, ixPt, ixDs = geom_pointtoline(point, lnflt)
    if nrLn is None:
        return None, None, None
    lnID = lnfid[nrLn]
    return lnID, ixPt, ixDs


def checkclosePt(pt, ptLt):
    for n, p in enumerate(ptLt):
        if mt.dist(pt, p) < 1e-3:
            return n
    return None