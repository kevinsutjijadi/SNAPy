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
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point, mapping, shape
from shapely.ops import nearest_points

# importing internal modules
from .prcs_geom import *
from .prcs_grph import *


# functions
def Base_BetweenessPatronage(Gdf:gpd.GeoDataFrame, Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict):
    '''
    Base_BetweenessPatronage(Gdf:gpd.GeoDataFrame, Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)\n
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
        'AlphaExp' : 0
    }
    for k,v in SettingDict.items(): # setting kwargs
        Settings[k] = v
    
    OutDf = pd.DataFrame([[0]], index=list(Gdf[Settings['AttrEdgeID']]), columns=['results'])
    expA = Settings['AlphaExp']
    SrcD = Settings['SearchDist']
    iOriEntID = list(OriDf.columns).index(Settings['AttrEntID'])
    iOriWgt = list(OriDf.columns).index(Settings['OriWgt'])
    iOriGeom = list(OriDf.columns).index('geometry')
    iDesEntID = list(DestDf.columns).index(Settings['AttrEntID'])
    iDesWgt = list(DestDf.columns).index(Settings['DestWgt'])
    iDesGeom = list(DestDf.columns).index('geometry')

    EntriesPtId = tuple(x[0] for x in EntriesPt)

    # cycles by oridf
    for Oi in range(len(OriDf)):
        Oid = OriDf.iat[Oi, iOriEntID]
        wgtO = OriDf.iat[Oi, iOriWgt]
        ptO = OriDf.iat[Oi, iOriGeom]
        try: Odt = EntriesPt[EntriesPtId.index(Oid)]
        except: continue
        # insert the origin point to the graph
        EdgeOri = Gph[Odt[5][0]][Odt[5][1]]
        Gph.add_edges_from((
            ('O', Odt[5][0], {'weight': Odt[6], 'cost': Odt[3][0], 'LineID': Odt[1]}),
            ('O', Odt[5][1], {'weight': Odt[6], 'cost': Odt[3][1], 'LineID': Odt[1]}),
        ))

        # how to keep stuff
        # starting individual calculation betweeness
        iterPths = []
        iterDsts = []
        iterWgts = []

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
            # getting paths
            fp, dsts = graphsim_paths(Gph, Odt, Ddt, "cost", SrcD, Settings['DetourR'], OriginAdd=False)
            if fp is None or len(fp) == 0: # pass on, on conditions with paths are not found or other errs
                continue # if not found
            if 0.0 in dsts:
                dstlt = [0]*len(dsts)
                for n, d in enumerate(dsts):
                    if d == 0.0: dstlt[n] = 0.1
                dsts = tuple(dstlt)
            if len(fp) == 1: # if only one path found, weight will be equal to destination weight
                iterPths.append(fp[0])
                iterDsts.append(dsts[0])
                iterWgts.append(DestDf.iat[Di, iDesWgt])
            else:
                # compiling through
                iterPths += tuple(fp)
                iterDsts += tuple(dsts)
                iterWgts += [DestDf.iat[Di, iDesWgt]]*len(fp)
            
        Gph.add_edges_from(((Odt[5][0], Odt[5][1], EdgeOri),))
        Gph.remove_edges_from((('O', Odt[5][0]), ('O', Odt[5][1]),))
        
        # now compiling the results
        if len(iterPths) == 0:
                continue
        
        DistMn = min(iterDsts)
        WgtPth = tuple(wgt*(DistMn/dst)**(1+expA) for dst, wgt in zip(iterDsts, iterWgts))# weighting calculations based on dist
        WgtPthSm = sum(WgtPth)
        TrafficPth = tuple(wgtO*(w/WgtPthSm) for w in WgtPth) # calculating traffic on each path
        # checking on segments
        for pth, trf in zip(iterPths, TrafficPth):
            for i in pth:
                OutDf.iat[i,0] += trf
            pass
    return (tuple(OutDf.index), tuple(OutDf['results']))


def Base_ReachN(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict):
    '''
    Base_Reach Count(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)\n
    packed function on ReachN\n
    returns tuple of ((result tuple), (PointID tuple))
    '''
    # types of calculation
    Settings={
        'AttrEntID': 'FID',
        'SearchDist': 1500.0,
    }
    for k,v in SettingDict.items(): # setting kwargs
        Settings[k] = v
    
    OutDf = pd.DataFrame([[0,],], index=list(OriDf[Settings['AttrEntID']]), columns=['results'])
    SrcD = Settings['SearchDist']
    iOriEntID = list(OriDf.columns).index(Settings['AttrEntID'])
    iOriGeom = list(OriDf.columns).index('geometry')
    iDesEntID = list(DestDf.columns).index(Settings['AttrEntID'])
    iDesGeom = list(DestDf.columns).index('geometry')

    EntriesPtId = tuple(x[0] for x in EntriesPt)

    # cycles by oridf
    for Oi in range(len(OriDf)):
        Oid = OriDf.iat[Oi, iOriEntID]
        ptO = OriDf.iat[Oi, iOriGeom]
        try: Odt = EntriesPt[EntriesPtId.index(Oid)]
        except: continue
        # insert the origin point in graph
        EdgeOri = Gph[Odt[5][0]][Odt[5][1]]
        Gph.add_edges_from((
            ('O', Odt[5][0], {'weight': Odt[6], 'cost': Odt[3][0], 'LineID': Odt[1]}),
            ('O', Odt[5][1], {'weight': Odt[6], 'cost': Odt[3][1], 'LineID': Odt[1]}),
        ))

        # starting individual calculation
        num = 0
        for Di in range(len(DestDf)):
            Did = DestDf.iat[Di, iDesEntID]
            if Did == Oid:
                continue
            try: Ddt = EntriesPt[EntriesPtId.index(Did)]
            except: continue

            # will fileter based on flyby distance
            if ptO.distance(DestDf.iat[Di, iDesGeom]) >  SrcD*1.2:
                continue
            
            dst = graphsim_dist(Gph, Odt, Ddt, "cost", OriginAdd=False)
            if dst is None or dst > SrcD: continue
            else: num += 1
        
        Gph.add_edges_from(((Odt[5][0], Odt[5][1], EdgeOri),))
        Gph.remove_edges_from((('O', Odt[5][0]), ('O', Odt[5][1]),))

        OutDf.iat[Oi, 0] = num
    return (tuple(OutDf.index), tuple(OutDf['results']),)


def Base_ReachW(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict):
    '''
    Base_ReachW(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)\n
    packed function on Reach sum Weight\n
    returns tuple of ((result tuple), (PointID tuple))
    '''
    # types of calculation
    Settings={
        'AttrEntID': 'FID',
        'SearchDist': 1500.0,
        'DestWgt': 'weight'
    }
    for k,v in SettingDict.items(): # setting kwargs
        Settings[k] = v
    
    OutDf = pd.DataFrame([[0, 0]], index=list(OriDf[Settings['AttrEntID']]), columns=['sumN', 'sumW'])
    SrcD = Settings['SearchDist']
    iOriEntID = list(OriDf.columns).index(Settings['AttrEntID'])
    iOriGeom = list(OriDf.columns).index('geometry')
    iDesEntID = list(DestDf.columns).index(Settings['AttrEntID'])
    iDesWgt = list(DestDf.columns).index(Settings['DestWgt'])
    iDesGeom = list(DestDf.columns).index('geometry')

    EntriesPtId = tuple(x[0] for x in EntriesPt)

    # cycles by oridf
    for Oi in range(len(OriDf)):
        Oid = OriDf.iat[Oi, iOriEntID]
        ptO = OriDf.iat[Oi, iOriGeom]
        try: Odt = EntriesPt[EntriesPtId.index(Oid)]
        except: continue
        # insert the origin point in graph
        EdgeOri = Gph[Odt[5][0]][Odt[5][1]]
        Gph.add_edges_from((
            ('O', Odt[5][0], {'weight': Odt[6], 'cost': Odt[3][0], 'LineID': Odt[1]}),
            ('O', Odt[5][1], {'weight': Odt[6], 'cost': Odt[3][1], 'LineID': Odt[1]}),
        ))

        # starting individual calculation
        num = 0
        wgt = 0.0
        for Di in range(len(DestDf)):
            Did = DestDf.iat[Di, iDesEntID]
            if Did == Oid:
                continue
            try: Ddt = EntriesPt[EntriesPtId.index(Did)]
            except: continue

            # will fileter based on flyby distance
            if ptO.distance(DestDf.iat[Di, iDesGeom]) >  SrcD*1.2:
                continue

            dst = graphsim_dist(Gph, Odt, Ddt, "cost", OriginAdd=False)
            if dst is None or dst > SrcD: continue
            else: 
                num += 1
                wgt += float(DestDf.iat[Di, iDesWgt])
        
        Gph.add_edges_from(((Odt[5][0], Odt[5][1], EdgeOri),))
        Gph.remove_edges_from((('O', Odt[5][0]), ('O', Odt[5][1]),))

        OutDf.iat[Oi, 0] = num
        OutDf.iat[Oi, 1] = wgt
    return (tuple(OutDf.index), tuple(OutDf['sumN']), tuple(OutDf['sumW']),)


def Base_ReachWD(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict):
    '''
    Base_ReachWD(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)\n
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

    }
    for k,v in SettingDict.items(): # setting kwargs
        Settings[k] = v
    
    OutDf = pd.DataFrame([[0, 0]], index=list(OriDf[Settings['AttrEntID']]), columns=['sumN', 'sumW'])
    SrcD = Settings['SearchDist']
    iOriEntID = list(OriDf.columns).index(Settings['AttrEntID'])
    iOriGeom = list(OriDf.columns).index('geometry')
    iDesEntID = list(DestDf.columns).index(Settings['AttrEntID'])
    iDesWgt = list(DestDf.columns).index(Settings['DestWgt'])
    iDesGeom = list(DestDf.columns).index('geometry')
    CalcExp = Settings['CalcExp']
    CalcComp = Settings['CalcComp']

    EntriesPtId = tuple(x[0] for x in EntriesPt)

    # cycles by oridf
    for Oi in range(len(OriDf)):
        Oid = OriDf.iat[Oi, iOriEntID]
        ptO = OriDf.iat[Oi, iOriGeom]
        try: Odt = EntriesPt[EntriesPtId.index(Oid)]
        except: continue
        # insert the origin point in graph
        EdgeOri = Gph[Odt[5][0]][Odt[5][1]]
        Gph.add_edges_from((
            ('O', Odt[5][0], {'weight': Odt[6], 'cost': Odt[3][0], 'LineID': Odt[1]}),
            ('O', Odt[5][1], {'weight': Odt[6], 'cost': Odt[3][1], 'LineID': Odt[1]}),
        ))

        # starting individual calculation
        rsltDt = []
        for Di in range(len(DestDf)):
            Did = DestDf.iat[Di, iDesEntID]
            if Did == Oid:
                continue

            try: Ddt = EntriesPt[EntriesPtId.index(Did)]
            except: continue

            # will filter based on flyby distance
            if ptO.distance(DestDf.iat[Di, iDesGeom]) >  SrcD*1.2:
                continue
            
            dst = graphsim_dist(Gph, Odt, Ddt, "cost", OriginAdd=False)
            if dst is None or dst > SrcD: continue
            else:
                rsltDt.append((dst, float(DestDf.iat[Di, iDesWgt])))

        val = 0
        rsltDt.sort(key = lambda x: x[0])
        if CalcExp == 0.0:
            for n, d in enumerate(rsltDt):
                val += abs(round(
                    d[1] * (-(d[0]/SrcD)+1) * (CalcComp**n), 4
                ))
        elif CalcExp < 0.00:
            for n, d in enumerate(rsltDt):
                val += round(
                    d[1] * (mt.e**(d[0]*CalcExp/SrcD)) * ((-d[0]/SrcD) + 1) * (CalcComp**n), 4
                )
        else:
            for n, d in enumerate(rsltDt):
                val += round(
                    d[1] * d[0] / SrcD * mt.e**(CalcExp*((d[0]/SrcD)-1)) * (CalcComp**n), 4
                )
        
        Gph.add_edges_from(((Odt[5][0], Odt[5][1], EdgeOri),))
        Gph.remove_edges_from((('O', Odt[5][0]), ('O', Odt[5][1]),))

        OutDf.iat[Oi, 0] = len(rsltDt)
        OutDf.iat[Oi, 1] = val
    return (tuple(OutDf.index), tuple(OutDf['sumN']), tuple(OutDf['sumW']),)


def Base_ReachND(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict):
    '''
    Base_ReachWD(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)\n
    packed function on Reach Weighted Distance\n
    returns tuple of ((result tuple), (PointID tuple))
    '''
    # types of calculation
    Settings={
        'AttrEntID': 'FID',
        'SearchDist': 1500.0,
        'CalcType': 'Linear',
        'DestWgt': 'weight',
    }
    for k,v in SettingDict.items(): # setting kwargs
        Settings[k] = v
    
    OutDf = pd.DataFrame([[0, 0]], index=list(OriDf[Settings['AttrEntID']]), columns=['sumN', 'sumND'])
    SrcD = Settings['SearchDist']
    iOriEntID = list(OriDf.columns).index(Settings['AttrEntID'])
    iOriGeom = list(OriDf.columns).index('geometry')
    iDesEntID = list(DestDf.columns).index(Settings['AttrEntID'])
    iDesWgt = list(DestDf.columns).index(Settings['DestWgt'])
    iDesGeom = list(DestDf.columns).index('geometry')

    EntriesPtId = tuple(x[0] for x in EntriesPt)

    # cycles by oridf
    for Oi in range(len(OriDf)):
        Oid = OriDf.iat[Oi, iOriEntID]
        ptO = OriDf.iat[Oi, iOriGeom]
        try: Odt = EntriesPt[EntriesPtId.index(Oid)]
        except: continue
        # insert the origin point in graph
        EdgeOri = Gph[Odt[5][0]][Odt[5][1]]
        Gph.add_edges_from((
            ('O', Odt[5][0], {'weight': Odt[6], 'cost': Odt[3][0], 'LineID': Odt[1]}),
            ('O', Odt[5][1], {'weight': Odt[6], 'cost': Odt[3][1], 'LineID': Odt[1]}),
        ))

        # starting individual calculation
        num = 0
        dist = SrcD
        for Di in range(len(DestDf)):
            Did = DestDf.iat[Di, iDesEntID]
            if Did == Oid:
                continue

            try: Ddt = EntriesPt[EntriesPtId.index(Did)]
            except: continue

            # will filter based on flyby distance
            if ptO.distance(DestDf.iat[Di, iDesGeom]) >  SrcD*1.1:
                continue

            dst = graphsim_dist(Gph, Odt, Ddt, "cost", OriginAdd=False)
            if dst is not None: 
                if dst < SrcD:
                    if dst < dist:
                        dist = dst
                    num += 1
                
        
        Gph.add_edges_from(((Odt[5][0], Odt[5][1], EdgeOri),))
        Gph.remove_edges_from((('O', Odt[5][0]), ('O', Odt[5][1]),))

        OutDf.iat[Oi, 0] = num
        OutDf.iat[Oi, 1] = dist
    return (tuple(OutDf.index), tuple(OutDf['sumN']), tuple(OutDf['sumND']),)


def Base_ReachNDW(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict):
    '''
    Base_ReachNDW(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)\n
    packed function on Reach Weighted Distance\n
    returns tuple of ((result tuple), (PointID tuple))
    '''
    # types of calculation
    Settings={
        'AttrEntID': 'FID',
        'SearchDist': 1500.0,
        'CalcType': 'Linear',
        'DestWgt': 'weight',
    }
    for k,v in SettingDict.items(): # setting kwargs
        Settings[k] = v
    
    OutDf = pd.DataFrame([[0, 0, 0]], index=list(OriDf[Settings['AttrEntID']]), columns=['sumN', 'sumMD', 'sumW'])
    SrcD = Settings['SearchDist']
    iOriEntID = list(OriDf.columns).index(Settings['AttrEntID'])
    iOriGeom = list(OriDf.columns).index('geometry')
    iDesEntID = list(DestDf.columns).index(Settings['AttrEntID'])
    iDesWgt = list(DestDf.columns).index(Settings['DestWgt'])
    iDesGeom = list(DestDf.columns).index('geometry')

    EntriesPtId = tuple(x[0] for x in EntriesPt)

    # cycles by oridf
    for Oi in range(len(OriDf)):
        Oid = OriDf.iat[Oi, iOriEntID]
        ptO = OriDf.iat[Oi, iOriGeom]
        try: Odt = EntriesPt[EntriesPtId.index(Oid)]
        except: continue
        # insert the origin point in graph
        EdgeOri = Gph[Odt[5][0]][Odt[5][1]]
        Gph.add_edges_from((
            ('O', Odt[5][0], {'weight': Odt[6], 'cost': Odt[3][0], 'LineID': Odt[1]}),
            ('O', Odt[5][1], {'weight': Odt[6], 'cost': Odt[3][1], 'LineID': Odt[1]}),
        ))

        # starting individual calculation
        num = 0
        dist = SrcD
        wgt = 0.0
        for Di in range(len(DestDf)):
            Did = DestDf.iat[Di, iDesEntID]
            if Did == Oid:
                continue

            try: Ddt = EntriesPt[EntriesPtId.index(Did)]
            except: continue

            # will filter based on flyby distance
            if ptO.distance(DestDf.iat[Di, iDesGeom]) >  SrcD*1.05:
                continue

            dst = graphsim_dist(Gph, Odt, Ddt, "cost", OriginAdd=False)
            if dst is not None: 
                if dst < SrcD:
                    if dst < dist:
                        dist = dst
                    num += 1
                    wgt += float(DestDf.iat[Di, iDesWgt])        
        
        Gph.add_edges_from(((Odt[5][0], Odt[5][1], EdgeOri),))
        Gph.remove_edges_from((('O', Odt[5][0]), ('O', Odt[5][1]),))

        OutDf.iat[Oi, 0] = num
        OutDf.iat[Oi, 1] = dist
        OutDf.iat[Oi, 2] = wgt
    return (tuple(OutDf.index), tuple(OutDf['sumN']), tuple(OutDf['sumMD']), tuple(OutDf['sumW']))


def Base_Straightness(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict):
    '''
    Base_StraightnessB(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)\n
    packed function on Straightness Averaged with distance weighting\n
    returns tuple of ((result tuple), (PointID tuple))
    '''
    # types of calculation
    Settings={
        'AttrEntID': 'FID',
        'SearchDist': 1500.0,
        'CalcExp': -0.35,
        'DestWgt': 'weight',
    }
    for k,v in SettingDict.items(): # setting kwargs
        Settings[k] = v
    
    OutDf = pd.DataFrame([[0,]], index=list(OriDf[Settings['AttrEntID']]), columns=['rslt'])
    SrcD = Settings['SearchDist']
    iOriEntID = list(OriDf.columns).index(Settings['AttrEntID'])
    iOriGeom = list(OriDf.columns).index('geometry')
    iDesEntID = list(DestDf.columns).index(Settings['AttrEntID'])
    iDesWgt = list(DestDf.columns).index(Settings['DestWgt'])
    iDesGeom = list(DestDf.columns).index('geometry')
    CalcExp = Settings['CalcExp']

    EntriesPtId = tuple(x[0] for x in EntriesPt)

    # cycles by oridf
    for Oi in range(len(OutDf)):
        Oid = OriDf.iat[Oi, iOriEntID]
        ptO = OriDf.iat[Oi, iOriGeom]
        try: Odt = EntriesPt[EntriesPtId[Oid]]
        except: continue
        # insert the origin point in graph
        EdgeOri = Gph[Odt[5][0]][Odt[5][1]]
        Gph.add_edges_from((
            ('O', Odt[5][0], {'weight': Odt[6], 'cost': Odt[3][0], 'LineID': Odt[1]}),
            ('O', Odt[5][1], {'weight': Odt[6], 'cost': Odt[3][1], 'LineID': Odt[1]}),
        ))

        # starting individual calculation
        rsltS = 0
        rsltW = 0
        for Di in range(len(DestDf)):
            Did = DestDf.iat[Di, iDesEntID]
            if Did == Oid:
                continue

            try: Ddt = EntriesPt[EntriesPtId.index(Did)]
            except: continue

            # will filter based on flyby distance
            dstFB = ptO.distance(DestDf.iat[Di, iDesGeom])
            if dstFB >  SrcD*1.0:
                continue
            
            dst = graphsim_dist(Gph, Odt, Ddt, "cost", OriginAdd=False)
            if dst is None or dst == 0.0: continue
            else:
                if CalcExp == 0.0:
                    wd = DestDf.iat[Di, iDesWgt]
                elif CalcExp < 0.0:
                    wd = DestDf.iat[Di, iDesWgt] * (mt.e**(-dst*CalcExp/SrcD)) * (dst/SrcD)
                else:
                    wd = DestDf.iat[Di, iDesWgt] * (dst / SrcD * mt.e**(CalcExp*((dst/SrcD)-1)))
                rsltS += dstFB/dst * wd 
                rsltW += wd
        
        if rsltW > 0:
            rslt = rsltS/rsltW
            OutDf.iat[Oi, 0] = rslt
        
        Gph.add_edges_from(((Odt[5][0], Odt[5][1], EdgeOri),))
        Gph.remove_edges_from((('O', Odt[5][0]), ('O', Odt[5][1]),))

    return (tuple(OutDf.index), tuple(OutDf['rslt']),)


# multiprocessing packing
def gph_Base_BetweenessPatronage_multi(inpt):
    '''
    packaged Base_BetweenessPatronage for multiprocessing\n
    Base_BetweenessPatronage(Gdf, Gph, EntriesPt, OriDf, DestDf, SettingDict)
    '''
    opt = Base_BetweenessPatronage(inpt[0], inpt[1], inpt[2], inpt[3], inpt[4], inpt[5])
    return opt


def gph_Base_MapPaths_multi(inpt):
    '''
    packaged Base_MapPaths for multiprocessing\n
    Base_MapPaths(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)
    '''
    OptIDs, OptPths, OptDsts = Base_MapPaths(inpt[0], inpt[1], inpt[2], inpt[3], inpt[4])
    return (OptIDs, OptPths, OptDsts)


def gph_Base_Reach_multi(inpt:tuple):
    '''
    packaged Base_Reach family for multiprocessing
    Base_Reach(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)
    first item is the type of processing used
    '''
    match inpt[0]:
        case 'N':
            Opt = Base_ReachN(inpt[1], inpt[2], inpt[3], inpt[4], inpt[5])
            return Opt
        case 'W':
            Opt = Base_ReachW(inpt[1], inpt[2], inpt[3], inpt[4], inpt[5])
            return Opt
        case 'WD':
            Opt = Base_ReachWD(inpt[1], inpt[2], inpt[3], inpt[4], inpt[5])
            return Opt
        case 'ND':
            Opt = Base_ReachND(inpt[1], inpt[2], inpt[3], inpt[4], inpt[5])
            return Opt
        case 'NDW':
            Opt = Base_ReachNDW(inpt[1], inpt[2], inpt[3], inpt[4], inpt[5])
            return Opt
        case other:
            return None


def gph_Base_Straightness_multi(inpt:tuple):
    '''
    packaged Base_Straigthness for multiprocessing\n
    Base_StraightnessA(Gph:nx.Graph, EntriesPt:tuple, OriDf:gpd.GeoDataFrame, DestDf:gpd.GeoDataFrame, SettingDict:dict)
    '''
    Opt = Base_Straightness(inpt[0], inpt[1], inpt[2], inpt[3], inpt[4])
    return Opt
