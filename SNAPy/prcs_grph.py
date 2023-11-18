### SNAPy (Spatial Network Analysis Python)
# graph processing, contains graph processing functions
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
import os
import pickle
from multiprocessing import Pool
from time import time

# importing dependent libraries
import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point, mapping, shape
from shapely.ops import nearest_points

# importing internal stuff
from .prcs_geom import *

# functions
def pathlength(G:nx.Graph, nodes:tuple, cost:str="cost"):
    """
    pathlength(G:nx.graph, nodes:nodesIDTuple, cost:strAttr)\n
    Calculating path length on a set of nodes\n
    returns length:float\n
    cost default value is \'cost\'\n
    """
    w = sum((G[nodes[ind]][nd][cost] for ind, nd in enumerate(nodes[1:])))
    return w


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


def BuildGraph(dataframe:gpd.GeoDataFrame, defcost:float=1.0, weightAtt:str=None, linetype=None):
    """
    buildgraph(dataframe:gpd.GeoDataFrame, defaultcost:float=1.0)\n
    build graph on networkx from gpd.GeoDataFrame format\n
    returns Graph:nx.Graph, pointID:tupleOfNodeCoordinates, dataframe:GeoDataFrame appended extra info\n
    default cost of 1.0 multiplied by length\n
    default weight of 1.0, else input by geodataframe attribute name\n
    linetype is an attribute of the geodataframe\n
    """
    pointid = [] # dict for all the points to id, with key as location and and item as id
    pointid_cnt = 0

    prmSt = [] # parameters for building the finished gdf for networkx graph build
    prmEd = []
    prmCost = []
    if weightAtt is None:
        defaultcost = (defcost,)*len(dataframe.geometry)
    else:
        defaultcost = tuple(dataframe[weightAtt])

    for n, itm in enumerate(dataframe.geometry):
        lnSt = itm.coords[0]
        lnEd = itm.coords[-1]

        ckSt = checkclosePt(lnSt, pointid)
        if ckSt == None:
            pointid.append(lnSt)
            idSt = pointid_cnt
            pointid_cnt += 1
        else:
            idSt = ckSt

        ckEd = checkclosePt(lnEd, pointid)
        if ckEd == None:
            pointid.append(lnEd)
            idEd = pointid_cnt
            pointid_cnt += 1
        else:
            idEd = ckEd
        
        prmSt.append(idSt)
        prmEd.append(idEd)
        prmCost.append(itm.length*defaultcost[n]) # cost includes multiplier
    
    if linetype is not None: # if need to specify linetype
        prmTy = list(dataframe[linetype])
    else: # if linetype is none or not detected, it will be name of graph
        prmTy = [None] * dataframe.shape[0]
    
    dataframe["EdgeCost"] = prmCost
    dataframe["EdgePtSt"] = prmSt
    dataframe["EdgePtEd"] = prmEd

    PdGraph = pd.DataFrame(
        {
        "source" : prmSt,
        "target" : prmEd,
        "cost" : prmCost,
        "LineID" : tuple(dataframe.index),
        "LineType" : prmTy,
        }
    )
    NxGraph = nx.from_pandas_edgelist(PdGraph, edge_attr=True) # making the graph
    pointid = tuple(pointid) # converting pointid to tuple

    return NxGraph, pointid, dataframe


def graph_addentries(GphDf:gpd.GeoDataFrame, EntryDf:gpd.GeoDataFrame, EntryDist:float=100.0, AttrNodeID:str='FID', AttrEdgeID:str='FID', EdgeCost:str='EdgeWeight'):
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
    for ptn, pt in enumerate(EntryDf.geometry):
        lnID, ixPt, ixDs = geom_closestline(pt, GphDf, EntryDist, AttrEdgeID)    
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
                cost = 1.0
            else:
                cost = lnFeat[lnNode_name[2]].iloc[0]
            ptLnEntry.append((
                EntryID[ptn], #  AttrID:str='FID' 0 - Entry Point ID
                lnID, # 1 - ID of connected edge
                ixDs, # 2 - Distance to intersection
                lnDist, # 3 - tuple of distance to the two nodes
                ixPt, # 4 - Point of intersection
                (lnFeat[lnNode_name[0]].iloc[0], lnFeat[lnNode_name[1]].iloc[0]), # 5 - tuple of connected node ID
                cost, # 6 - weight of edge
                lnSplit
                )) # Entry data formatted, idk if this is the most effective way
    ptLnEntry = tuple(ptLnEntry) # tupleized bcs it will be called a lot of times
    return ptLnEntry


def graphsim_dist(Gph:nx.Graph, DtOri:tuple, DtDst:tuple, AttrWgt:str="cost", OriginAdd:bool=True):
    """
    graphsim_dist() \n
    getting single shortest path distance \n
    returns distance
    """

    # checks if the origin and destination on the same line
    sln_Dst = None
    EdgeOri = Gph[DtOri[5][0]][DtOri[5][1]]
    EdgeDst = Gph[DtDst[5][0]][DtDst[5][1]]

    if DtOri[1] == DtDst[1]:
        sln_Dst = abs(DtOri[3][0] - DtDst[3][0]) # delta distance between the points
        # if DetourR > 1.0:
        # adding edges to points
        return sln_Dst
        # if OriginAdd:
        #     Gph.add_edges_from((
        #         ('O', DtOri[5][0], {'weight': DtOri[6], 'cost': DtOri[3][0], 'LineID': DtOri[1]}),
        #         ('O', DtOri[5][1], {'weight': DtOri[6], 'cost': DtOri[3][1], 'LineID': DtOri[1]}),
        #     ))
        #     # Gph.remove_edge(DtOri[5][0], DtOri[5][1]) # because occupies same stuff
        # Gph.add_edges_from((
        #     ('D', DtDst[5][0], {'weight': DtDst[6], 'cost': DtDst[3][0], 'LineID': DtDst[1]}),
        #     ('D', DtDst[5][1], {'weight': DtDst[6], 'cost': DtDst[3][1], 'LineID': DtDst[1]}),
        # ))
        # Gph.remove_edge(DtDst[5][0], DtDst[5][1])
    else:
        # adding edges to points
        if OriginAdd:
            Gph.add_edges_from((
                ('O', DtOri[5][0], {'weight': DtOri[6], 'cost': DtOri[3][0], 'LineID': DtOri[1]}),
                ('O', DtOri[5][1], {'weight': DtOri[6], 'cost': DtOri[3][1], 'LineID': DtOri[1]}),
            ))
            Gph.remove_edge(DtOri[5][0], DtOri[5][1])

        Gph.add_edges_from((
            ('D', DtDst[5][0], {'weight': DtDst[6], 'cost': DtDst[3][0], 'LineID': DtDst[1]}),
            ('D', DtDst[5][1], {'weight': DtDst[6], 'cost': DtDst[3][1], 'LineID': DtDst[1]}),
        ))
        Gph.remove_edge(DtDst[5][0], DtDst[5][1])

    if nx.has_path(Gph, 'O', 'D'):
        ShstLen = nx.shortest_path_length(Gph, 'O', 'D', weight=AttrWgt)
        if OriginAdd:
            Gph.add_edges_from(((DtOri[5][0], DtOri[5][1], EdgeOri),))
            Gph.remove_edges_from((('O', DtOri[5][0]), ('O', DtOri[5][1]),))
            Gph.add_edges_from(((DtDst[5][0], DtDst[5][1], EdgeDst),))
        else:
            Gph.add_edges_from(((DtDst[5][0], DtDst[5][1], EdgeDst),))
        Gph.remove_edges_from((('D', DtDst[5][0]), ('D', DtDst[5][1]),))
        
        return ShstLen
    else:
        if OriginAdd:
            Gph.add_edges_from(((DtOri[5][0], DtOri[5][1], EdgeOri),))
            Gph.remove_edges_from((('O', DtOri[5][0]), ('O', DtOri[5][1]),))
            if sln_Dst is None:
                Gph.add_edges_from(((DtDst[5][0], DtDst[5][1], EdgeDst),))
        else:
            Gph.add_edges_from(((DtDst[5][0], DtDst[5][1], EdgeDst),))
        Gph.remove_edges_from((('D', DtDst[5][0]), ('D', DtDst[5][1]),))
        return None


def graphsim_paths(Gph:nx.Graph, DtOri:tuple, DtDst:tuple, AttrWgt:str="cost", ScDist:float=800.0, DetourR:float=1.0, path_lim:int=20, OriginAdd:bool=True):
    """
    graphsim_paths() \n
    getting path from an origin to a destination \n
    returns nested tuple of paths with line ID, and tuple distance
    """

    # checks if the origin and destination on the same line
    sln_Dst = None
    EdgeOri = Gph[DtOri[5][0]][DtOri[5][1]]
    EdgeDst = Gph[DtDst[5][0]][DtDst[5][1]]
    if DtOri[1] == DtDst[1]:
        sln_Dst = abs(DtOri[3][0] - DtDst[3][0]) # delta distance between the points

        return ((DtOri[1],),), (sln_Dst,)
        # if DetourR > 1.0:
        # adding edges to points
        # if OriginAdd:
        #     Gph.add_edges_from((
        #         ('O', DtOri[5][0], {'weight': DtOri[6], 'cost': DtOri[3][0], 'LineID': DtOri[1]}),
        #         ('O', DtOri[5][1], {'weight': DtOri[6], 'cost': DtOri[3][1], 'LineID': DtOri[1]}),
        #     ))
        #     # Gph.remove_edge(DtOri[5][0], DtOri[5][1]) # because occupies same stuff
        # Gph.add_edges_from((
        #     ('D', DtDst[5][0], {'weight': DtDst[6], 'cost': DtDst[3][0], 'LineID': DtDst[1]}),
        #     ('D', DtDst[5][1], {'weight': DtDst[6], 'cost': DtDst[3][1], 'LineID': DtDst[1]}),
        # ))
        # Gph.remove_edge(DtDst[5][0], DtDst[5][1])
    else:
        # adding edges to points
        if OriginAdd:
            Gph.add_edges_from((
                ('O', DtOri[5][0], {'weight': DtOri[6], 'cost': DtOri[3][0], 'LineID': DtOri[1]}),
                ('O', DtOri[5][1], {'weight': DtOri[6], 'cost': DtOri[3][1], 'LineID': DtOri[1]}),
            ))
            Gph.remove_edge(DtOri[5][0], DtOri[5][1])

        Gph.add_edges_from((
            ('D', DtDst[5][0], {'weight': DtDst[6], 'cost': DtDst[3][0], 'LineID': DtDst[1]}),
            ('D', DtDst[5][1], {'weight': DtDst[6], 'cost': DtDst[3][1], 'LineID': DtDst[1]}),
        ))
        Gph.remove_edge(DtDst[5][0], DtDst[5][1])


    if nx.has_path(Gph, 'O', 'D'):
        ShstLen = nx.shortest_path_length(Gph, 'O', 'D', weight=AttrWgt)
        Udst = []
        Upth = []
        if ShstLen < ScDist:
            if DetourR <= 1.0: # if detour is equal to one, only returns the closest
                if sln_Dst: # situation if falls in the same line
                    if ShstLen > sln_Dst:
                        Udst = (sln_Dst,)
                        Upth = ((DtOri[1],),)

                        if OriginAdd:
                            Gph.add_edges_from(((DtOri[5][0], DtOri[5][1], EdgeOri),))
                            Gph.remove_edges_from((('O', DtOri[5][0]), ('O', DtOri[5][1]),))
                            if sln_Dst is None:
                                Gph.add_edges_from(((DtDst[5][0], DtDst[5][1], EdgeDst),))
                        else:
                            Gph.add_edges_from(((DtDst[5][0], DtDst[5][1], EdgeDst),))
                        Gph.remove_edges_from((('D', DtDst[5][0]), ('D', DtDst[5][1]),))
                        # no need repairing graph because sameline detour same
                        return Upth, Udst
                    
                for pth in nx.shortest_simple_paths(Gph, 'O', 'D', weight=AttrWgt):
                    if len(pth) > 3:
                        pthn = tuple(Gph[a][b]['LineID'] for a,b in zip(pth[:-1], pth[1:]))
                        # pthn = (DtOri[1],) + tuple(Gph[a][b]['LineID'] for a,b in zip(pth[1:-2], pth[2:-1])) + (DtDst[1],)
                    else:
                        pthn = (DtOri[1], DtDst[1])
                    Udst = (ShstLen,)
                    Upth = (pthn,)

                    if OriginAdd:
                        Gph.add_edges_from(((DtOri[5][0], DtOri[5][1], EdgeOri),))
                        Gph.remove_edges_from((('O', DtOri[5][0]), ('O', DtOri[5][1]),))
                        if sln_Dst is None:
                            Gph.add_edges_from(((DtDst[5][0], DtDst[5][1], EdgeDst),))
                    else:
                        Gph.add_edges_from(((DtDst[5][0], DtDst[5][1], EdgeDst),))
                    Gph.remove_edges_from((('D', DtDst[5][0]), ('D', DtDst[5][1]),))

                    return Upth, Udst
            
            memDist = 0
            # if detourR more than one:
            limDist = ShstLen * DetourR
            if sln_Dst: # situational if falls in the same line state
                Udst.append(sln_Dst)
                Upth.append((DtOri[1],))
                for pth in nx.shortest_simple_paths(Gph, 'O', 'D', weight=AttrWgt):
                    pthDist = pathlength(Gph, pth, AttrWgt)
                    if pthDist == memDist or len(Udst) > path_lim:
                        break
                    elif len(pth) == 3:
                        pass
                    elif pthDist < limDist:
                        if len(pth) > 3:
                            pthn = tuple(Gph[a][b]['LineID'] for a,b in zip(pth[:-1], pth[1:]))
                            if pthn[0] == pthn[1]:
                                continue
                        else:
                            pthn = (DtOri[1], DtDst[1])
                        Udst.append(pthDist)
                        Upth.append(pthn) # tupleized
                    else: break
            else:
                for pth in nx.shortest_simple_paths(Gph, 'O', 'D', weight=AttrWgt):
                    pthDist = pathlength(Gph, pth, AttrWgt)
                    if pthDist == memDist or len(Udst) > path_lim:
                        break
                    elif pthDist < limDist:
                        if len(pth) > 3:
                            pthn = tuple(Gph[a][b]['LineID'] for a,b in zip(pth[:-1], pth[1:]))
                            if pthn[0] == pthn[1]:
                                continue
                        else:
                            pthn = (DtOri[1], DtDst[1])
                        Udst.append(pthDist)
                        Upth.append(pthn) # tupleized
                    else: break
        # reinstating back graph
        if OriginAdd:
            Gph.add_edges_from(((DtOri[5][0], DtOri[5][1], EdgeOri),))
            Gph.remove_edges_from((('O', DtOri[5][0]), ('O', DtOri[5][1]),))
            if sln_Dst is None:
                Gph.add_edges_from(((DtDst[5][0], DtDst[5][1], EdgeDst),))
        else:
            Gph.add_edges_from(((DtDst[5][0], DtDst[5][1], EdgeDst),))
        Gph.remove_edges_from((('D', DtDst[5][0]), ('D', DtDst[5][1]),))

        # finalizing outputs
        Upth = tuple(Upth) 
        Udst = tuple(Udst)
        return Upth, Udst
    else:
        if OriginAdd:
            Gph.add_edges_from(((DtOri[5][0], DtOri[5][1], EdgeOri),))
            Gph.remove_edges_from((('O', DtOri[5][0]), ('O', DtOri[5][1]),))
            if sln_Dst is None:
                Gph.add_edges_from(((DtDst[5][0], DtDst[5][1], EdgeDst),))
        else:
            Gph.add_edges_from(((DtDst[5][0], DtDst[5][1], EdgeDst),))
        Gph.remove_edges_from((('D', DtDst[5][0]), ('D', DtDst[5][1]),))
        return None, None
    

def Base_MapPaths(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict):
    '''
    Base_MapPaths(Gdf:gpd.GeoDataFrame, Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)\n
    packed function for path mapping, for number of processes\n
    returns ---'''

    Settings={
        'AttrEdgeID': 'FID',
        'AttrEntID': 'FID',
        'SearchDist': 1500.0,
        'DetourR' : 1.0,
    }
    for k,v in SettingDict.items(): # setting kwargs
        Settings[k] = v
    
    # finding base items first
    SrcD = Settings['SearchDist']
    iOriEntID = list(OriDf.columns).index(Settings['AttrEntID'])
    iOriGeom = list(OriDf.columns).index('geometry')
    iDesEntID = list(DestDf.columns).index(Settings['AttrEntID'])
    iDesGeom = list(DestDf.columns).index('geometry')

    EntriesPtId = tuple(x[0] for x in EntriesPt)

    # outputs
    OptPths = []
    OptDsts = []
    OptIDs = []
    for Oi in range(len(OriDf)):
        Oid = OriDf.iat[Oi, iOriEntID]
        ptO = OriDf.iat[Oi, iOriGeom]
        Odt = None
        try: Odt = EntriesPt[EntriesPtId.index(Oid)]
        except: continue
        # insert the origin point to the graph
        EdgeOri = Gph[Odt[5][0]][Odt[5][1]]
        Gph.add_edges_from((
            ('O', Odt[5][0], {'weight': Odt[6], 'cost': Odt[3][0], 'LineID': Odt[1]}),
            ('O', Odt[5][1], {'weight': Odt[6], 'cost': Odt[3][1], 'LineID': Odt[1]}),
        ))

        for Di in range(len(DestDf)):
            Did = DestDf.iat[Di, iDesEntID]
            if Did == Oid:
                continue
            try: Ddt = EntriesPt[EntriesPtId.index(Did)]
            except: continue

            # insert filter search here
            # will filter based on fly by distance
            if ptO.distance(DestDf.iat[Di, iDesGeom]) > SrcD*1.2:
                continue
            # graphsim_paths(Gph:nx.Graph, DtOri:tuple, DtDst:tuple, AttrWgt:str="cost", ScDist:float=800.0, DetourR:float=1.0, path_lim:int=20, OriginAdd:bool=True)
            # getting paths
            fp, dsts = graphsim_paths(Gph, Odt, Ddt, "cost", SrcD, Settings['DetourR'], OriginAdd=False)
            if fp is None or len(fp) == 0: # pass on, on conditions with paths are not found or other errs
                continue # if not found
            if 0.0 in dsts:
                dstlt = [0]*len(dsts)
                for n, d in enumerate(dsts):
                    if d == 0.0: dstlt[n] = 0.1
                dsts = tuple(dstlt)
            OptPths.append(fp)
            OptDsts.append(dsts)
            OptIDs.append((Oid, Did))
            
        Gph.add_edges_from(((Odt[5][0], Odt[5][1], EdgeOri),))
        Gph.remove_edges_from((('O', Odt[5][0]), ('O', Odt[5][1]),))

    OptPths = tuple(OptPths)
    OptDsts = tuple(OptDsts)
    OptIDs = tuple(OptIDs)
    return OptIDs, OptPths, OptDsts


# package for multiprocessing
def gph_addentries_multi(inpt):
    '''
    packaged graph_addentries for multiprocessing
    graph_addentries(GphDf:gpd.GeoDataFrame, EntryDf:gpd.GeoDataFrame, EntryDist:float=100.0, AttrNodeID:str='FID', AttrEdgeID:str='FID')
    '''
    opt = graph_addentries(inpt[0], inpt[1], inpt[2], inpt[3], inpt[4], inpt[5])
    return opt
