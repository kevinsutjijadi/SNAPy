### SNAPy (Spatial Network Analysis Python)
# graph processing, contains graph processing functions
# Kevin Sutjijadi @ 2023

__author__ = "Kevin Sutjijadi"
__copyright__ = "Copyright 2023, Kevin Sutjijadi"
__credits__ = ["Kevin Sutjijadi"]
__version__ = "0.2.0"

"""
Spatial Network Analysis (SNA) module
using networkx, shapely, and geopandas, for network analysis simulations 
with more control over spatial items
"""

import math as mt
import os
import pickle
from multiprocessing import Pool
from time import time

# importing dependent libraries
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point, mapping, shape
from shapely.ops import nearest_points

# importing internal stuff
# from .prcs_geom import *
from .SGACy.geom import *
from .SGACy.graph import GraphCy

# functions


def mappath_featid(df:gpd.GeoDataFrame, path:tuple, AttrID:str='FID'):
    """
    mappath(df:gpd.GeoDataFrame, path:tupleOfNodesID)\n
    mapping path of nodes into the featid of the related geodataframe\n
    returns tuple of edge featid\n
    """
    featIdLt = []
    for n, nd in enumerate(path[:-1]):
        fid = df[(df['EdgePtSt'] == nd) & (df['EdgePtEd'] == path[n+1])][AttrID]
        if fid.empty:
            fid = df[(df['EdgePtSt'] == path[n+1]) & (df['EdgePtEd'] == nd)][AttrID]
        featIdLt.append(fid.iloc[0])
    return featIdLt


def BuildGraph(
        dataframe:gpd.GeoDataFrame,
        sizeBuffer:float=1.1,
        A_Lnlength:str|None = None,
        A_LnlengthR:str|None = None,
        A_LnW:str|None = None,
        A_PtstW:str|None = None,
        A_PtstC:str|None = None,
        A_PtedW:str|None = None,
        A_PtedC:str|None = None,
        ):
    """
    buildgraph(dataframe:gpd.GeoDataFrame, defaultcost:float=1.0)\n
    build graph on networkx from gpd.GeoDataFrame format\n
    returns Graph:nx.Graph, pointID:tupleOfNodeCoordinates, dataframe:GeoDataFrame appended extra info\n
    default cost of 1.0 multiplied by length\n
    default weight of 1.0, else input by geodataframe attribute name\n
    linetype is an attribute of the geodataframe\n
    """
    
    SgGraph = GraphCy(int(dataframe.size*sizeBuffer), int(dataframe.size*sizeBuffer))
    SgGraph.from_pandas_edgelist(dataframe,
                                A_Lnlength,
                                A_LnlengthR,
                                A_LnW,
                                A_PtstW,
                                A_PtstC,
                                A_PtedW,
                                A_PtedC,
                                )
    NetworkSize = SgGraph.sizeInfo() # returns (nodesize, edgesize)
    SgGraph.reallocNodes(NetworkSize[0]*sizeBuffer)

    return SgGraph


def graph_addentries(GphDf:gpd.GeoDataFrame, EntryDf:gpd.GeoDataFrame, EntryDist:float=100.0, AttrNodeID:str='FID', AttrEdgeID:str='FID', EdgeCost:str|None=None):
    """
    graph_addentries(GphDf:gpd.GeoDataFrame, EntryDf:gpd.GeoDataFrame, EntryDist:float=100.0)\n
    Adding entry points into the graph\n
    returns tuple (OriginPtID, featID, distToLine, (distToNodes), (NodesID))
    """
    # assigning closest line fid to point
    # i will hate myself for this later but it wont mince the base graph
    # it will create a dataset of how the point connects with the geodataframe
    # for origin - distance on the same line will just calculate from the distance, not nodes
    
    lnNode_name = ('EdgePtSt', 'EdgePtEd', EdgeCost)
    ptLnEntry = [] # nested tuple (OriginPtID, featID, distToLine, PointIntersect, (distToNodes), (NodesID))
    EntryID = list(EntryDf[AttrNodeID])
    GphDf['bbox'] = GphDf.apply(lambda x: bbox(x.geometry), axis=1)
    
    AttrEdgeIDx = tuple(GphDf.columns).index(AttrEdgeID)
    for ptn, pt in enumerate(EntryDf.geometry):
        lnID, ixPt, ixDs = geom_closestline(pt, GphDf, EntryDist, AttrEdgeIDx)
        lnID = int(lnID)
        if lnID is not None:
            lnFeat = GphDf[(GphDf[AttrEdgeID]==lnID)] # the line
            lnSplit = geom_linesplit(lnFeat.geometry, ixPt)
            lnDist = []
            for sg in lnSplit:
                if sg is None: lnDist.append(0)
                else: lnDist.append(sg.length)
            lnDist = tuple(lnDist)
            if EdgeCost is None:
                cost = 0.0
            else:
                cost = lnFeat[lnNode_name[2]].iloc[0]

            if len(tuple(ixPt.coords)[0])==2:
                ixPt = tuple(ixPt.coords)[0] + (0.0,)
            else:
                ixPt = tuple(ixPt.coords)[0]
                
            ptLnEntry.append((
                EntryID[ptn], #  AttrID:str='FID' 0 - Entry Point ID
                lnID, # 1 - ID of connected edge
                ixDs, # 2 - Distance to intersection
                lnDist, # 3 - tuple of distance to the two nodes
                ixPt, # 4 - Point of intersection
                cost, # 5 - cost of edge
                lnSplit # 6 - line geometry, 2 items
                )) # Entry data formatted, idk if this is the most effective way
    ptLnEntry = tuple(ptLnEntry) # tupleized bcs it will be called a lot of times
    return ptLnEntry


def graphsim_dist(Gph:GraphCy, 
                  DtOri:tuple, 
                  DtDst:tuple, 
                  ScDist:float=1500.0,
                  DistMul:float=2.0, 
                  LimCycle:int=1_000_000,
                  IncFBDist:bool=False):
    """
    graphsim_dist() \n
    getting single shortest path distance \n
    returns distance
    """

    # checks if the origin and destination on the same line
    if not IncFBDist:
        rslt = Gph.PathDist_AStar_VirtuEntry(
            DtOri[1], DtOri[4], DtOri[3],
            DtDst[1], DtDst[4], DtDst[3],
            LimDist = ScDist,
            LimCycle = LimCycle,
            DistMul = DistMul,
        )
        return rslt
    else:
        rslt = Gph.PathDistComp_AStar_VirtuEntry(
            DtOri[1], DtOri[4], DtOri[3],
            DtDst[1], DtDst[4], DtDst[3],
            LimDist = ScDist,
            LimCycle = LimCycle,
            DistMul = DistMul,
        )
        return rslt


def graphsim_paths(Gph:GraphCy, 
                   DtOri:tuple, 
                   DtDst:tuple,  
                   ScDist:float=800.0, 
                   DetourR:float=1.0,  
                   DistMul:float=2.0, 
                   EdgeCmin:float=0.9,
                   PathLim:int=2_000,
                   LimCycle:int=1_000_000):
    """
    graphsim_paths() \n
    getting path from an origin to a destination \n
    returns nested tuple of paths with line ID, and tuple distance
    """

    # checks if the origin and destination on the same line
    sln_Dst = None
    
    if DtOri[1] == DtDst[1]:
        sln_Dst = abs(DtOri[3][0] - DtDst[3][0]) # delta distance between the points
        if sln_Dst < 0.1:
            sln_Dst = 0.1
        return ((sln_Dst,),((DtOri[1],),))
    rslt = Gph.PathFind_Multi_VirtuEntry(
        DtOri[1], DtOri[4], DtOri[3],
        DtDst[1], DtDst[4], DtDst[3],
        DistMulLim = DetourR,
        LimDist = ScDist,
        LimCycle = LimCycle,
        DistMul = DistMul,
        EdgeCmin = EdgeCmin,
        PathLim = PathLim,
    )

    return rslt


# package for multiprocessing
def gph_addentries_multi(inpt):
    '''
    packaged graph_addentries for multiprocessing
    graph_addentries(GphDf:gpd.GeoDataFrame, EntryDf:gpd.GeoDataFrame, EntryDist:float=100.0, AttrNodeID:str='FID', AttrEdgeID:str='FID')
    '''
    opt = graph_addentries(inpt[0], inpt[1], inpt[2], inpt[3], inpt[4], inpt[5])
    return opt
