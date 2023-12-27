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
import numpy as np

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


def geom_linesplit(ln, point:Point, tol=1e-3):
    """
    geom_linesplit(line:shapely.MultiLineString/LineString, point:shapely.point)\n
    Splitting line at an intersecting point\n
    returns tuple of 2 MultiLineString, always the beginning and the end based on the line direction\n
    """
    ln = list(ln)[0]
    if ln.distance(point) > tol:
        return None
    coortp = ln.coords
    j = None
    for i in range(len(coortp) - 1):
        if LineString(coortp[i:i+2]).distance(point) < tol:
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


def geom_linesplits(ln, points, tol=1e-3):
    """
    geom_linesplit(line:shapely.MultiLineString/LineString, point:list of shapely.point)\n
    Splitting line at multiple intersecting point\n
    returns tuple of LineString, always the beginning and the end based on the line direction\n
    """
    # ln = list(ln)[0] # wtf

    ist = []
    for pt in points: # checks if any intersecting
        if ln.distance(pt) < tol:
            ist.append(pt)

    if len(ist) == 0:
        return None
    coorn = [ln,]

    for pt in ist:
        cutd = False
        coortp = coorn
        coorn = []
        for lnc in coortp:
            lnc = lnc.coords
            j = None
            for i in range(len(lnc) - 1):
                if LineString(lnc[i:i+2]).distance(pt) < tol:
                    j = i
                    break
            if j is not None:
                if Point(lnc[0]).distance(pt) < tol or Point(lnc[-1]).distance(pt) < tol:
                    lnspl = (ln,)
                elif Point(lnc[j + 1]).equals(pt):
                    lnspl = (LineString(lnc[:j + 2]), LineString(lnc[j + 1:]))
                else:
                    lnspl = (LineString(lnc[:j + 1] + [(pt.x, pt.y)]), LineString([(pt.x, pt.y)]+ lnc[j + 1:]))
                coorn += lnspl
            else:
                coorn.append(LineString(lnc))
    return coorn

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


def checkclosePt(pt, ptLt, tol=1e-3):
    for n, p in enumerate(ptLt):
        if mt.dist(pt, p) < tol:
            return n
    return None


def eucDist(ptA:np.ndarray, ptO:np.ndarray):
    '''
    eucDistArr(ptA:np.ndarray, ptO:np.ndarray)\n
    calculates distance of arrays of points using numpy based 0 loop for fast performance
    '''
    if len(ptA.shape) == 1: # for pointtopoint
        return np.linalg.norm(ptA - ptO)
    if len(ptO.shape) == 1: # if ptO is a single point 1D data
        return np.sqrt(np.sum((ptA - ptO)**2, axis=1))
    else:
        dsts = np.sqrt(np.maximum(0,
                        np.repeat([np.sum(ptA**2, axis=1)], ptO.shape[0], axis=0).T - 
                       2 * np.matmul(ptA, ptO.T) + 
                       np.repeat([np.sum(ptO**2, axis=1)], ptA.shape[0], axis=0)
                       ))
        return dsts

def bbox(pline:LineString, tl=1.0):
    """
    bbox(pline:LineString)
    make a np array of bbox coordinates of (Xmin, Ymin, Xmax, Ymax)
    """
    ln = np.array(pline.coords)
    return np.array((np.min(ln[:,0])-tl, np.min(ln[:,1])-tl, np.max(ln[:,0])+tl, np.max(ln[:,1])+tl))

def bbox_intersects(a, b):
    """
    bbox_intersects
    quick comparison, returns pattern of touching bboxes
    """
    if b[2] < a[0] or b[0] > a[2] or b[1] > a[3] or b[3] < a[1]:
        return  False
    return True

def IntersectLinetoPoints(d, f, tol=1e-3):
    if d.geometry.intersects(f):
        pt = d.geometry.intersection(f)
        if pt.type == "MultiPoint":
            pt = list(pt.geoms)
        elif not (isinstance(pt, list) or isinstance(pt, tuple)):
            pt = (pt,)
        return pt
    elif f.distance(Point(d.geometry.coords[0])) < tol:
        return (Point(d.geometry.coords[0]),)
    elif f.distance(Point(d.geometry.coords[-1])) < tol:
        return (Point(d.geometry.coords[-1]),)
    else:
        return None

def NetworkSegmentIntersections(df, dfi=None, EndPoints=True, tol=1e-3):
    """
    NetworkSegment(df:GeoDataFrame, Endpoints:bool)\n
    intersect lines from a single geodataframe. Returns segmented lines in geopandas, and end points geopandas that contains boolean attributes as intersections or end points.
    """
    df = df.copy()
    ndf = {}
    clt = []
    for c in df.columns:
        ndf[c] = []
        if c not in ('geometry'):
            clt.append(c)
    
    df['fid'] = df.index

    if dfi is None:
        dfi = df

    df['bbox'] = df.apply(lambda x: bbox(x.geometry), axis=1) # vectorized bbox
    for i, d in df.iterrows():
        ptlt = []
        dbx = d['bbox']
        dfx = dfi[dfi.apply(lambda x: bbox_intersects(dbx, x.bbox), axis=1)]
        dfx = dfx[dfx['fid'] != i]
        ptr = dfx.apply(lambda x: IntersectLinetoPoints(d, x.geometry, tol), axis=1)
        for p in ptr:
            if p is not None:
                ptlt += p

        try:
            lns = geom_linesplits(d.geometry, ptlt)
            if lns is not None:
                for l in lns:
                    ndf['geometry'].append(l)
                    for c in clt:
                        ndf[c].append(d[c])
                        
            else:
                print(f'\tline {i} has no intersections')
                ndf['geometry'].append(d.geometry)
                for c in clt:
                    ndf[c].append(d[c])
        except:
            print(f'\tline {i} bounds has no intersections')
            ndf['geometry'].append(d.geometry)
            for c in clt:
                ndf[c].append(d[c])
    ndf = gpd.GeoDataFrame(ndf, crs=df.crs)

    
    if EndPoints:
        ptlt = []
        for ln in ndf['geometry']:
            ptl = ln.coords
            ptlt += [ptl[0], ptl[-1]] # collecting endpoints
        ptar = np.asarray(ptlt)

        alar = np.sum(eucDist(ptar, ptar) < tol, axis=1) # using numpy array to solve near points
        snar = tuple(alar > 1)

        pts = [Point(p) for p in ptar]
        pts = gpd.GeoDataFrame(geometry=pts, crs=df.crs)
        pts['fid'] = pts.index
        pts['Intersection'] = snar

        return ndf, pts
    else:
        pts = []
        return ndf, pts