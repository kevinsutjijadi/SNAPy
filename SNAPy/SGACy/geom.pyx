# distutils: language = c++

### SGACy (Spatial Graph Algorithm Cython)
# geometry related utilities
# Kevin Sutjijadi @ 2024

cimport cython
from libcpp.queue cimport priority_queue
from libcpp.algorithm cimport sort, find
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.string cimport string
from libc.math cimport sqrt
import geopandas as gpd
import pandas as pd
from libc.stdlib cimport malloc, realloc, free
from typing import List
from libc.stdint cimport int32_t, uint32_t
from libc.string cimport memset
from shapely.geometry import LineString, MultiLineString, Point, mapping, shape
from shapely.ops import nearest_points, split, linemerge
import numpy as np
from collections import defaultdict

# Main graph class, 

cdef struct Point3d:
    float x
    float y
    float z

cdef struct bBox:
    float[4] bounds

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Point3d MakePoint3d(float& x, float& y, float z= 0.0):
    cdef Point3d pt
    pt.x = x
    pt.y = y
    pt.z = z
    return pt

@cython.boundscheck(False)
@cython.wraparound(False)
def bbox(pline:LineString, tl=1.0):
    """
    bbox(pline:LineString)
    make a np array of bbox coordinates of (Xmin, Ymin, Xmax, Ymax)
    """
    ln = np.array(pline.coords)
    return np.array((np.min(ln[:,0])-tl, np.min(ln[:,1])-tl, np.max(ln[:,0])+tl, np.max(ln[:,1])+tl))

@cython.boundscheck(False)
@cython.wraparound(False)
def bbox_intersects(a, b):
    """
    bbox_intersects
    quick comparison, returns pattern of touching bboxes
    """
    if b[2] < a[0] or b[0] > a[2] or b[1] > a[3] or b[3] < a[1]:
        return  False
    return True

def multilinestring_to_linestring(gdf):
    # Function to explode a MultiLineString into multiple LineStrings
    def explode_multilinestring(row):
        if row.geometry.geom_type == 'MultiLineString':
            return [LineString(line) for line in row.geometry.geoms]
        else:
            return [row.geometry]

    # Explode MultiLineStrings and reset the index
    exploded = gdf.explode(column='geometry', ignore_index=True)
    
    # Apply the explode function and stack the results
    exploded['geometry'] = exploded.apply(explode_multilinestring, axis=1)
    exploded = exploded.explode('geometry', ignore_index=True)
    
    # Reset the geometry column
    exploded = exploded.set_geometry('geometry')
    exploded = exploded.reset_index(drop=True)
    
    return exploded

def geom_pointtoline(point:Point, lineset:tuple):
    """
    calculates nearest terms from a point to a set of lines
    returnes nrstln, nrstpt, nrstdst
    """
    cdef int TempId
    cdef float TempDst = 1_000_000.0
    cdef float neardist
    cdef int lnid = 0

    for ln in lineset:
        nearpt = nearest_points(ln, point)
        neardist = point.distance(nearpt[0])
        if neardist < TempDst:
            TempDst = neardist
            TempPt = nearpt[0]
            TempId = lnid
        lnid += 1
    return TempId, TempPt, TempDst

def geom_linesplit(ln:LineString, point:Point, tol=1e-3):
    """
    geom_linesplit(line:shapely.MultiLineString/LineString, point:shapely.point)\n
    Splitting line at an intersecting point\n
    returns tuple of 2 MultiLineString, always the beginning and the end based on the line direction\n
    """
    cdef int j = -1
    cdef int i
    if ln.distance(point) > tol:
        return None
    coortp = ln.coords
    for i in range(len(coortp) - 1):
        if LineString(coortp[i:i+2]).distance(point) < tol:
            j = i
            break
    if j == 0:
        lnspl = (LineString([coortp[0]] + [(point.x, point.y)]), LineString([(point.x, point.y)] + coortp[1:]))
    elif j == len(coortp)-2:
        lnspl = (LineString(coortp[:-1] + [(point.x, point.y)]), LineString([(point.x, point.y)] + [coortp[-1]]))
    elif Point(coortp[j]).distance(point) < tol:
        lnspl = (LineString(coortp[:j + 1]), LineString(coortp[j:]))
    else:
        lnspl = (LineString(coortp[:j + 1] + [(point.x, point.y)]), LineString([(point.x, point.y)]+ coortp[j + 1:]))
    return lnspl     


def geom_linesplits(line:LineString, point:Point, tol:float=1e-3):
    """
    geom_linesplit(line:shapely.MultiLineString/LineString, point:list of shapely.point)\n
    Splitting line at multiple intersecting point\n
    returns tuple of LineString, always the beginning and the end based on the line direction\n
    """
    cdef int iter = 0
    cdef int j = -1
    cdef int i
    coorn = [line,]
    for pt in point:
        iter += 1
        coorntp = coorn
        coorn = []
        for lnc in coorntp:
            lnc = lnc.coords
            j = -1
            for i in range(len(lnc) - 1):
                if LineString(lnc[i:i+2]).distance < tol:
                    j = i
                    break
            if j != -1:
                if Point(lnc[0]).distance(pt) < tol or Point(lnc[-1]).distance(pt) < tol:
                    lnspl = (LineString(lnc),)
                elif Point(lnc[j+1]).distance(pt) < tol:
                    lnspl = (LineString(lnc[:j+2]), LineString(lnc[j+1:]))
                else:
                    lnspl = (LineString(lnc[:j+1] + [(pt.x, pt.y)]), LineString([(pt.x, pt.y)]+ lnc[j + 1:]))
                coorn += lnspl
            else:
                coorn.append(LineString(lnc))
    return coorn

def geom_closestline(point:Point, lineset:gpd.GeoDataFrame, searchlim:float=200, AttrIDx:int=0):
    """
    geom_closestline(point:gpd.GeoDataFrame.geometry, lineset:gpd.GeoDataFrame, searchlim:float=200)\n
    Calculating closest line to a point\n
    returns lineid, point, and distance to entry point\n
    search limit is set to 200\n
    """
    # filter by box dimensions

    # plim = (point[0]-searchlim, point[0]+searchlim, point[1]-searchlim, point[1]+searchlim)
    plim = np.array((point.x-searchlim, point.y-searchlim, point.x+searchlim, point.y+searchlim), dtype=float)
    dfx = lineset[lineset.apply(lambda x: bbox_intersects(plim, x.bbox), axis=1)]
    if len(dfx) == 0:
        return None, None, None
    nrLn, ixPt, ixDs = geom_pointtoline(point, tuple(dfx.geometry))
    if nrLn is None:
        return None, None, None
    lnID = dfx.iat[nrLn, AttrIDx]
    
    return lnID, ixPt, ixDs

def FlattenLineString(gdf):
    """
    FlattenLineString(gdf:GeodataFrame)
    converts geodataframe with linetringZ to linestring
    """
    odf = gdf.copy()
    odf["geometry"] = gdf.apply(lambda x: LineString(np.array(x.geometry.coords)[:,:2]), axis=1)
    return odf

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

def IntersectLinetoPoints(d, f, tol=1e-3):
    px = []
    if d.geometry.distance(Point(f.coords[0])) < tol:
        px.append(Point(f.coords[0]))
    if d.geometry.distance(Point(f.coords[-1])) < tol:
        px.append(Point(f.coords[-1]))
    if d.geometry.intersects(f):
        pt = d.geometry.intersection(f)
        if pt.type == "MultiPoint":
            pt = list(pt.geoms)
        elif not (isinstance(pt, list) or isinstance(pt, tuple)):
            pt = [pt,]
        px += pt
        return px
    else:
        return px

def NetworkSegmentDistance(df, dist=50):
    """
    NetworkSegmentDistance(df:GeoDataFrame of Network, dist=50)
    Segments network lines by an approximate distance of projection.
    """
    df = df.copy()
    if len(df.geometry[0].coords[0]) == 3:
        df = FlattenLineString(df)
        print('Dataframe has LineStringZ, flattening to LineString.')

    ndf = {}
    clt = []
    for c in df.columns:
        ndf[c] = []
        if c not in ('geometry',):
            clt.append(c)
        
    for i, d in df.iterrows():
        dgl = d.geometry.length
        if dgl > dist*1.5:
            NSg = round(dgl/dist, 0)
            LSg = dgl/NSg
            lnc = d.geometry.coords
            Sgmts = []
            tpts = []
            wlks = 0
            wlkd = 0
            for i in range(len(lnc)-1):
                tpts.append(lnc[i])
                sgd = eucDist(np.array(lnc[i]), np.array(lnc[i+1]))
                sga = (wlks + sgd) // LSg
                if sga > 0:
                    for n in range(int(sga)):
                        tdst = eucDist(np.array(tpts[-1]), np.array(lnc[i+1]))
                        wlkd += LSg - wlks
                        if (dgl - wlkd) < (LSg*1.1 - wlks) and n == (int(sga)-1):
                            break
                        else:
                            param = (LSg - wlks)/tdst
                            edpt = (((lnc[i+1][0] - tpts[-1][0])*param + tpts[-1][0]), 
                                    ((lnc[i+1][1] - tpts[-1][1])*param + tpts[-1][-1]))
                            tpts.append(edpt)
                            Sgmts.append(LineString(tpts))
                            tpts = [edpt]
                            if n != (int(sga)-1):
                                wlks = 0
                            else:
                                wlks = eucDist(np.array(tpts[-1]), np.array(lnc[i+1]))
                                wlkd += eucDist(np.array(tpts[-1]), np.array(lnc[i+1]))
                else:
                    wlks += sgd
                    wlkd += sgd
            if len(tpts) > 0:
                tpts.append(lnc[-1])
                Sgmts.append(LineString(tpts))
                
            for n in Sgmts:
                ndf['geometry'].append(n)
                for c in clt:
                    ndf[c].append(d[c])
        else:
            ndf['geometry'].append(d.geometry)
            for c in clt:
                ndf[c].append(d[c])
    return gpd.GeoDataFrame(ndf, crs=df.crs)

def NetworkCompileIntersections(df, tol=3):
    """
    Gathers endpoints as intersections and counts number of intersections
    """
    ar = np.concatenate([
        np.array([(ln.coords[0][0], ln.coords[0][1]) for ln in df.geometry]),
        np.array([(ln.coords[-1][0], ln.coords[-1][1]) for ln in df.geometry])
        ])
    ar = ar.round(tol)
    a, b = np.unique(ar, return_counts=True, axis=0)
    pts = np.array([Point(x[0], x[1]) for x in a])
    dfpt = gpd.GeoDataFrame(
        data={'JunctCnt':b},
        geometry=pts,
    )
    return dfpt


def NetworkSegmentIntersections(df, EndPoints=True, tol=3):
    """
    NetworkSegment(df:GeoDataFrame, Endpoints:bool)\n
    intersect lines from a single geodataframe. Returns segmented lines in geopandas, and end points geopandas that contains boolean attributes as intersections or end points.
    """
    df = df.copy()

    if 'MultiLineString' in tuple(df.geometry.type):
        print('DataFrame contains MultiLineString, exploding to LineString')
        df = multilinestring_to_linestring(df)

    dfNw = df['geometry']
    ar, ds = dfNw.geometry.sindex.nearest(dfNw.geometry, return_all=True, return_distance=True, exclusive=True)
    dfNw.geometry.type
    # for a, x, y in zip(ar[1], ar[0][0], ar[0][1]):
    #     print(a, x, y)
    ptr = ds < 1e-3
    arac = ar[0][ptr]
    arbc = ar[1][ptr]
    gp = defaultdict(list)
    for first, second in zip(arac, arbc):
        gp[first].append(second)

    for k, v in gp.items():
        vg = dfNw[dfNw.index.isin(v)]
        kg = dfNw.loc[k]
        # print('\t', k, v, kg.length)
        ixpt = kg.intersection(vg).explode(ignore_index=False)
        ixpj = np.array(kg.project(ixpt))
        ixpt = np.array(ixpt[(ixpj > 0.0)*(kg.length > ixpj)])
        # print(ixpt)
        if len(ixpt) == 0:
            continue
        for p in ixpt:
            kg = MultiLineString(split(kg, p).geoms)
        df.loc[k, 'geometry'] = kg

    nwdf = multilinestring_to_linestring(df)
    print(f'NetworkSegmentIntersection splits from {len(dfNw)} lines to {len(nwdf)} lines')
    if EndPoints:
        ixdf = NetworkCompileIntersections(nwdf, tol)
        return nwdf, ixdf
    return nwdf
    
    




def MapEntries(GphDf:gpd.GeoDataFrame, EntryDf:gpd.GeoDataFrame, EntryDist:float = 100.0, AttrNodeID:str='FID', AttrEdgeID:str='FID', EdgeCost:str|None=None):
    """
    Mapping Entries into graph
    """
    # cdef int EntriesN = len(EntryDf)
    # cdef EntryMap* Entryinfo = <EntryMap*>malloc(EntriesN * sizeof(EntryMap))
    EntryInfo = []
    NearestMatch = GphDf.geometry.sindex.nearest(EntryDf.geometry, return_all=False,  return_distance=True)
    EdgeMatch = NearestMatch[0][1]

    EntryCtr = EntryDf.geometry.centroid.values
    EdgeMatch_geom = GphDf.geometry.values[EdgeMatch]
    Match_distO = EdgeMatch_geom.project(EntryCtr)
    Match_distD = EdgeMatch_geom.length - Match_distO
    Match_ixPt = EdgeMatch_geom.interpolate(Match_distO)
    Match_Eid = GphDf.index.values[EdgeMatch]
    Match_distx = NearestMatch[1]

    cullp = Match_distx > EntryDist
    # cull by dist
    if np.sum(cullp) > 0.0:
        Match_Eid[cullp] = -1

    if EdgeCost is not None:
        EntryInfo = pd.DataFrame({
            'fid':EntryDf.index,
            'lnID':Match_Eid,
            'ixDs':Match_distx,
            'lnDist':np.vstack((Match_distO, Match_distD)).T.tolist(),
            'ixPt':np.vstack((Match_ixPt.x, Match_ixPt.y, np.zeros(Match_ixPt.shape, dtype=np.float32))).T.tolist(),
            'cost':GphDf[EdgeCost].values[EdgeMatch]
            })
    else:
        EntryInfo = pd.DataFrame({
            'fid':EntryDf.index,
            'lnID':Match_Eid,
            'ixDs':Match_distx,
            'lnDist':np.vstack((Match_distO, Match_distD)).T.tolist(),
            'ixPt':np.vstack((Match_ixPt.x, Match_ixPt.y, np.zeros(Match_ixPt.shape, dtype=np.float32))).T.tolist(),
            'cost':np.zeros(Match_ixPt.shape, dtype=np.float32)
            })
    
    # free(gphbounds)
    return EntryInfo