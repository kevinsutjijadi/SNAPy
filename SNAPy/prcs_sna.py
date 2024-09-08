### SNAPy (Spatial Network Analysis Python)
# housing spatial network functions
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
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point, mapping, shape
from shapely.ops import nearest_points
import numpy as np

# importing internal modules
# from .prcs_geom import *
from .prcs_grph import *
from .SGACy.graph import GraphCy
from .SGACy.geom import *


# functions
def Base_BetweenessPatronage_Singular(Gph:GraphCy, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict):
    '''
    Base_BetweenessPatronage(Gph:GraphCy, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)\n
    packed function for multithreading on betweenesspatronage\n
    returns tuple of ((result tuple), (LineID tuple))
    '''

    Settings={
        'OriWgt': 'weight',
        'DestWgt' : 'weight',
        'AttrEdgeID': 'FID',
        'AttrEntID': 'FID',
        'SearchDist' : 1500.0, 
        'DetourR' : 1.0, 
        'AlphaExp' : 0.0,
        'DistMul' : 2.0,
        'EdgeCmin' : 0.9,
        'PathLim' : 2_000,
        'LimCycles' : 1_000_000,
    }
    for k,v in SettingDict.items(): # setting kwargs
        Settings[k] = v
    
    OutAr = np.zeros(Gph.sizeInfo()[1], dtype=float)
    expA = Settings['AlphaExp']
    SrcD = Settings['SearchDist']

    numpaths = 0
    # cycles by oridf
    DetourR = Settings['DetourR']
    DistMul = Settings['DistMul']
    EdgeCmin = Settings['EdgeCmin']
    PathLim = Settings['PathLim']
    LimCycles = Settings['LimCycles']
    DestWgt = np.array(DestDf[Settings['DestWgt']], dtype=float)
    OriWgt = np.array(OriDf[Settings['OriWgt']], dtype=float)
    DidAr = np.array(DestDf[Settings['AttrEntID']], dtype=int)
    OidAr = np.array(OriDf[Settings['AttrEntID']], dtype=int)
    # EntriesPtId = np.array(tuple((x[0] for x in EntriesPt)), dtype=int)

    for Oi in range(len(OriDf)):
        Oid = OidAr[Oi]
        wgtO = OriWgt[Oi]
        # starting individual calculation betweeness
        iterPths = []
        iterDsts = []
        iterWgts = []

        for Di in range(len(DestDf)):
            Did = DidAr[Di]
            if Did == Oid:
                continue
            # print(Ddt[1], Ddt[4], Ddt[3], DestDf.iat[Di, iDesWgt])
            rslt = graphsim_paths(Gph, Oid, Did, SrcD, DetourR, DistMul, EdgeCmin, PathLim, LimCycles)
            
            if rslt is None or len(rslt[0]) == 0: # pass on, on conditions with paths are not found or other errs
                # print((f'\tNone'))
                continue # if not found
            # print(f'\t{min(rslt[0]):,.2f}-{max(rslt[0]):,.2f} {max(rslt[0])/min(rslt[0]):,.4f} || {len(rslt[1])}')
            dsts, fp = rslt
            if 0.0 in dsts:
                dstlt = [0]*len(dsts)
                for n, d in enumerate(dsts):
                    if d == 0.0: 
                        dstlt[n] = 0.1
                    else:
                        dstlt[n] = d
                dsts = tuple(dstlt)
            if len(fp) == 1: # if only one path found, weight will be equal to destination weight
                iterPths.append(fp[0])
                iterDsts.append(dsts[0])
                iterWgts.append(DestWgt[Di])
            else:
                # compiling through
                iterPths += tuple(fp)
                iterDsts += tuple(dsts)
                iterWgts += [DestWgt[Di]]*len(fp)
        # now compiling the results
        if len(iterPths) == 0:
            continue
        # for d, w, p in zip(iterDsts, iterWgts, iterPths):
        #     print(d, w, p)
        numpaths += len(iterPths)
        DistMn = min(iterDsts)
        WgtPth = tuple(wgt*(DistMn/dst)**(1+expA) for dst, wgt in zip(iterDsts, iterWgts)) # weighting calculations based on dist
        WgtPthSm = sum(WgtPth)
        TrafficPth = tuple(wgtO*(w/WgtPthSm) for w in WgtPth) # calculating traffic on each path
        # checking on segments
        for pth, trf in zip(iterPths, TrafficPth):
            for i in pth:
                OutAr[i] += trf
            pass
    print(f'Total Paths {numpaths:,}')
    return OutAr

def Base_BetweenessPatronage_Plural(Gph:GraphCy, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict):
    '''
    Base_BetweenessPatronage(Gdf:gpd.GeoDataFrame, Gph:GraphCy, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)\n
    packed function for multithreading on betweenesspatronage\n
    returns tuple of ((result tuple), (LineID tuple))
    '''

    Settings={
        'OriWgt': 'weight',
        'DestWgt' : 'weight',
        'AttrEdgeID': 'FID',
        'AttrEntID': 'FID',
        'SearchDist' : 1500.0, 
        'DetourR' : 1.0, 
        'AlphaExp' : 0.0,
        'DistMul' : 2.0,
        'EdgeCmin' : 0.9,
        'PathLim' : 2_000,
        'LimCycles' : 1_000_000,
        'Include_Destination': False,
    }
    for k,v in SettingDict.items(): # setting kwargs
        Settings[k] = v
    
    Include_Destination = Settings['Include_Destination']
    OutAr = np.zeros(Gph.sizeInfo()[1], dtype=np.float32)
    if Include_Destination:
        OutEn = np.zeros(DestDf.shape[0], dtype=np.float32)
    expA = Settings['AlphaExp']
    SrcD = Settings['SearchDist']

    numpaths = 0
    DetourR = Settings['DetourR']
    DistMul = Settings['DistMul']
    PathLim = Settings['PathLim']
    LimCycles = Settings['LimCycles']
    # cycles by oridf
    # EntriesPtId = np.array(tuple((x[0] for x in EntriesPt)), dtype=int)
    DidAr = np.array(DestDf[Settings['AttrEntID']], dtype=np.int32)
    DestWgt = np.array(DestDf[Settings['DestWgt']], dtype=np.float32)
    # DestWgt = DestDf[Settings['DestWgt']]
    OriWgt = np.array(OriDf[Settings['OriWgt']], dtype=np.float32)
    OidAr = np.array(OriDf[Settings['AttrEntID']], dtype=np.int32)
    DestinationDatas = tuple((DidAr[d] for d in range(len(DestDf))))
    for Oi in range(len(OriDf)):
        Oid = OidAr[Oi]
        wgtO = OriWgt[Oi]
        iterDsts, iterPths, iterIds = Gph.PathFind_Multi_MultiDest_VirtuEntry(
            Oid,
            DestinationDatas,
            DetourR, 
            SrcD,
            LimCycles,
            DistMul,
            PathLim, 
        ) 
        # now compiling the results
        if len(iterIds) == 0:
            continue
        numpaths += len(iterDsts)
        DistMn = min(iterDsts)
        iterDsts = np.array(iterDsts, dtype=np.float32)
        iterIds = np.array(iterIds, dtype=np.int32)
        # iterWgts = DestWgt.loc[iterIds]
        iterId = np.digitize(iterIds, DidAr)-1
        iterWgts = DestWgt[iterId]
        WgtPth = iterWgts * (DistMn/iterDsts) ** (1+expA)
        WgtPthSm = np.sum(WgtPth)
        TrafficPth = wgtO*(WgtPth/WgtPthSm)
        
        if Include_Destination:
            for pth, trf, id in zip(iterPths, TrafficPth, iterId):
                OutEn[id] += trf
                for i in pth:
                    OutAr[i] += trf
            pass
        else:
            for pth, trf in zip(iterPths, TrafficPth):
                for i in pth:
                    OutAr[i] += trf
    print(f'Total Paths {numpaths:,}')
    if Include_Destination:
        return OutAr, OutEn
    else:
        return OutAr


def Base_ReachN(Gph:GraphCy, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict):
    '''
    Base_Reach Count(Gph:GraphCy, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)\n
    packed function on ReachN\n
    returns tuple of ((result tuple), (PointID tuple))
    '''
    # types of calculation
    Settings={
        'AttrEntID': 'FID',
        'SearchDist': 1500.0,
        'DistMul' : 2.0,
        'LimCycles' : 1_000_000,
    }
    for k,v in SettingDict.items(): # setting kwargs
        Settings[k] = v

    SearchDist = Settings['SearchDist']
    LimCycles = Settings['LimCycles']
    OutAr = np.zeros(len(OriDf), dtype=int)
    OidAr = np.array(OriDf[Settings['AttrEntID']], dtype=int)
    Didtp = tuple(DestDf[Settings['AttrEntID']])
    # cycles by oridf
    for Oi in range(len(OriDf)):
        Oid = OidAr[Oi]

        # starting individual calculation

        rslt = Gph.PathDistComp_MultiDest_VirtuEntry(
            Oid,
            Didtp,
            SearchDist,
            LimCycles,
            NullVal = 0.0
        )
        if rslt is None:
            continue
        rslt = np.array(rslt)
        nulp = rslt[:,0] != 0.0
        OutAr[Oi] = np.sum(nulp)
    return (tuple(OriDf[Settings['AttrEntID']]), tuple(OutAr))


def Base_ReachW(Gph:GraphCy, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict):
    '''
    Base_ReachW(Gph:GraphCy, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)\n
    packed function on Reach sum Weight\n
    returns tuple of ((result tuple), (PointID tuple))
    '''
    # types of calculation
    Settings={
        'AttrEntID': 'FID',
        'SearchDist': 1500.0,
        'DestWgt': 'weight',
        'DistMul' : 2.0,
        'LimCycles' : 1_000_000,
    }
    for k,v in SettingDict.items(): # setting kwargs
        Settings[k] = v
    
    SearchDist = Settings['SearchDist']
    LimCycles = Settings['LimCycles']
    OutAr = np.zeros((len(OriDf), 2), dtype=float)
    DestWgt = np.array(DestDf[Settings['DestWgt']], dtype=float)
    OidAr = np.array(OriDf[Settings['AttrEntID']], dtype=int)
    Didtp = tuple(DestDf[Settings['AttrEntID']])
    # cycles by oridf
    for Oi in range(len(OriDf)):
        Oid = OidAr[Oi]

        # starting individual calculation
        rslt = Gph.PathDistComp_MultiDest_VirtuEntry(
            Oid,
            Didtp,
            SearchDist,
            LimCycles,
            NullVal = 0.0
        )
        if rslt is None:
            continue
        rslt = np.array(rslt)
        nulp = rslt[:,0] != 0.0
        OutAr[Oi][0] = np.sum(nulp)
        OutAr[Oi][1] = np.sum(DestWgt[nulp])
    return (tuple(OriDf[Settings['AttrEntID']]), tuple(OutAr[:,0]), tuple(OutAr[:,1]))


def Base_ReachWD(Gph:GraphCy, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict):
    '''
    Base_ReachWD(Gph:GraphCy, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)\n
    packed function on Reach Weighted Distance\n
    returns tuple of ((result tuple), (PointID tuple))
    '''
    # types of calculation
    Settings={
        'AttrEntID': 'FID',
        'SearchDist': 1500.0,
        'CalcExp': -0.35,
        'CalcComp': 0.6,
        'DestWgt': 'weight',
        'DistMul' : 2.0,
        'LimCycles' : 1_000_000,
    }
    for k,v in SettingDict.items(): # setting kwargs
        Settings[k] = v
    
    SrcD = Settings['SearchDist']
    CalcExp = Settings['CalcExp']
    CalcComp = Settings['CalcComp']

    SearchDist = Settings['SearchDist']
    LimCycles = Settings['LimCycles']
    OutAr = np.zeros((len(OriDf), 2), dtype=float)
    DestWgt = np.array(DestDf[Settings['DestWgt']], dtype=float)
    OidAr = np.array(OriDf[Settings['AttrEntID']], dtype=int)
    Didtp = tuple(DestDf[Settings['AttrEntID']])
    # cycles by oridf
    for Oi in range(len(OriDf)):
        Oid = OidAr[Oi]

        # starting individual calculation

        rslt = Gph.PathDistComp_MultiDest_VirtuEntry(
            Oid,
            Didtp,
            SearchDist,
            LimCycles,
            NullVal = 0.0
        )
        if rslt is None:
            continue
        rslt = np.array(rslt)
        rslt = rslt[:,0]
        nulp = rslt != 0.0
        rslt = rslt[nulp]
        dwgt = DestWgt[nulp]
        nrg = np.arange(dwgt.size)
        nargsort = np.argsort(rslt)
        rslt = rslt[nargsort]
        dwgt = dwgt[nargsort]

        if CalcExp == 0.0:
            OutAr[Oi][1] = np.absolute(dwgt * (-(rslt/SrcD)+1) * (CalcComp**nrg))
        elif CalcExp < 0:
            OutAr[Oi][1] = dwgt * (mt.e**(rslt*CalcExp/SrcD)) * ((-rslt/SrcD) + 1) * (CalcComp**nrg)
        else:
            OutAr[Oi][1] = dwgt * rslt / SrcD * mt.e**(CalcExp*((rslt/SrcD)-1) * (CalcComp**nrg))

        OutAr[Oi][0] = np.sum(nulp)
    return (tuple(OriDf[Settings['AttrEntID']]), tuple(OutAr[:,0]), tuple(OutAr[:,1]))


def Base_ReachND(Gph:GraphCy, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict):
    '''
    Base_ReachWD(Gph:GraphCy, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)\n
    packed function on Reach Weighted Distance\n
    returns tuple of ((result tuple), (PointID tuple))
    '''
    # types of calculation
    Settings={
        'AttrEntID': 'FID',
        'SearchDist': 1500.0,
        'CalcType': 'Linear',
        'DestWgt': 'weight',
        'DistMul' : 2.0,
        'LimCycles' : 1_000_000,
    }
    for k,v in SettingDict.items(): # setting kwargs
        Settings[k] = v
    
    SearchDist = Settings['SearchDist']
    LimCycles = Settings['LimCycles']
    OutAr = np.zeros((len(OriDf), 2), dtype=float)
    OidAr = np.array(OriDf[Settings['AttrEntID']], dtype=int)
    Didtp = tuple(DestDf[Settings['AttrEntID']])
    # cycles by oridf
    for Oi in range(len(OriDf)):
        Oid = OidAr[Oi]

        # starting individual calculation
        rslt = Gph.PathDistComp_MultiDest_VirtuEntry(
            Oid,
            Didtp,
            SearchDist,
            LimCycles,
            NullVal = 0.0
        )
        if rslt is None:
            continue
        rslt = np.array(rslt)
        rslt = rslt[:,0]
        nulp = rslt != 0.0
        rslt = rslt[nulp]

        OutAr[Oi][0] = np.sum(nulp)
        OutAr[Oi][1] = np.min(rslt)
    return (tuple(OriDf[Settings['AttrEntID']]), tuple(OutAr[:,0]), tuple(OutAr[:,1]))


def Base_ReachNDW(Gph:GraphCy, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict):
    '''
    Base_ReachNDW(Gph:GraphCy, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)\n
    packed function on Reach Weighted Distance\n
    returns tuple of ((result tuple), (PointID tuple))
    '''
    # types of calculation
    Settings={
        'AttrEntID': 'FID',
        'SearchDist': 1500.0,
        'CalcType': 'Linear',
        'DestWgt': 'weight',
        'DistMul' : 2.0,
        'LimCycles' : 1_000_000,
    }
    for k,v in SettingDict.items(): # setting kwargs
        Settings[k] = v

    SearchDist = Settings['SearchDist']
    LimCycles = Settings['LimCycles']
    DestWgt = np.array(DestDf[Settings['DestWgt']], dtype=float)
    OutAr = np.zeros((len(OriDf), 3), dtype=float)
    OidAr = np.array(OriDf[Settings['AttrEntID']], dtype=int)
    Didtp = tuple(DestDf[Settings['AttrEntID']])
    # cycles by oridf
    for Oi in range(len(OriDf)):
        Oid = OidAr[Oi]

        # starting individual calculation

        rslt = Gph.PathDistComp_MultiDest_VirtuEntry(
            Oid,
            Didtp,
            SearchDist,
            LimCycles,
            NullVal = 0.0
        )
        if rslt is None:
            continue
        rslt = np.array(rslt)
        rslt = rslt[:,0]
        nulp = rslt != 0.0
        rslt = rslt[nulp]     

        OutAr[Oi][0] = np.sum(nulp)
        OutAr[Oi][1] = np.min(rslt)
        OutAr[Oi][2] = np.sum(DestWgt[nulp])
    return (tuple(OriDf[Settings['AttrEntID']]), tuple(OutAr[:,0]), tuple(OutAr[:,1]), tuple(OutAr[:,2]))


def Base_Straightness(Gph:GraphCy, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict):
    '''
    Base_StraightnessB(Gph:GraphCy, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)\n
    packed function on Straightness Averaged with distance weighting\n
    returns tuple of ((result tuple), (PointID tuple))
    '''
    # types of calculation
    Settings={
        'AttrEntID': 'FID',
        'SearchDist': 1500.0,
        'CalcExp': -0.35,
        'DestWgt': 'weight',
        'DistMul' : 2.0,
        'LimCycles' : 1_000_000,
    }
    for k,v in SettingDict.items(): # setting kwargs
        Settings[k] = v
    
    # OutDf = pd.DataFrame([[0,]], index=list(OriDf[Settings['AttrEntID']]), columns=['rslt'])
    SrcD = Settings['SearchDist']
    CalcExp = Settings['CalcExp']
    SearchDist = Settings['SearchDist']
    LimCycles = Settings['LimCycles']
    DestWgt = np.array(DestDf[Settings['DestWgt']], dtype=float)
    OutAr = np.zeros(len(OriDf), dtype=float)
    OidAr = np.array(OriDf[Settings['AttrEntID']], dtype=int)
    Didtp = tuple(DestDf[Settings['AttrEntID']])
    # cycles by oridf
    for Oi in range(len(OriDf)):
        Oid = OidAr[Oi]

        # starting individual calculation
        rsltS = 0
        rsltW = 0

        rslt = Gph.PathDistComp_MultiDest_VirtuEntry(
            Oid,
            Didtp,
            SearchDist,
            LimCycles,
            NullVal = 0.0
        )
        if rslt is None:
            continue
        rslt = np.array(rslt, dtype=np.float32)
        nulp = rslt[:,0] != 0.0
        rslt = rslt[nulp]
        dwgt = DestWgt[nulp]

        if CalcExp == 0.0:
            wd = dwgt
        elif CalcExp < 0.0:
            wd = dwgt * (mt.e**(rslt[:,0]*CalcExp/SrcD)) * (1-(rslt[:,0]/SrcD))
        else:
            wd = dwgt * (rslt[:,0]/SrcD) * (mt.e**(CalcExp*((rslt[:,0]/SrcD)-1)))
        try:
            rsltS = np.sum(rslt[:,1]/rslt[:,0] * wd)
            rsltW = np.sum(wd)
        except:
            continue
        
        if rsltW > 0:
            rslt = rsltS/rsltW
            OutAr[Oi] = rslt

    # return (tuple(OutDf.index), tuple(OutDf['rslt']),)
    return (tuple(OriDf[Settings['AttrEntID']]), tuple(OutAr))


# multiprocessing packing
def gph_Base_BetweenessPatronage_Singular_multi(inpt):
    '''
    packaged Base_BetweenessPatronage for multiprocessing\n
    Base_BetweenessPatronage(Gdf, Gph, EntriesPt, OriDf, DestDf, SettingDict)
    '''
    opt = Base_BetweenessPatronage_Singular(inpt[0], inpt[1], inpt[2], inpt[3])
    return opt

def gph_Base_BetweenessPatronage_Plural_multi(inpt):
    '''
    packaged Base_BetweenessPatronage for multiprocessing\n
    Base_BetweenessPatronage(Gdf, Gph, EntriesPt, OriDf, DestDf, SettingDict)
    '''
    opt = Base_BetweenessPatronage_Plural(inpt[0], inpt[1], inpt[2], inpt[3])
    return opt

def gph_Base_Reach_multi(inpt:tuple):
    '''
    packaged Base_Reach family for multiprocessing
    Base_Reach(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)
    first item is the type of processing used
    '''
    match inpt[0]:
        case 'N':
            Opt = Base_ReachN(inpt[1], inpt[2], inpt[3], inpt[4])
            return Opt
        case 'W':
            Opt = Base_ReachW(inpt[1], inpt[2], inpt[3], inpt[4])
            return Opt
        case 'WD':
            Opt = Base_ReachWD(inpt[1], inpt[2], inpt[3], inpt[4])
            return Opt
        case 'ND':
            Opt = Base_ReachND(inpt[1], inpt[2], inpt[3], inpt[4])
            return Opt
        case 'NDW':
            Opt = Base_ReachNDW(inpt[1], inpt[2], inpt[3], inpt[4])
            return Opt
        case other:
            return None


def gph_Base_Straightness_multi(inpt:tuple):
    '''
    packaged Base_Straigthness for multiprocessing\n
    Base_StraightnessA(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)
    '''
    Opt = Base_Straightness(inpt[0], inpt[1], inpt[2], inpt[3])
    return Opt
