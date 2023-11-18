### SNAPy (Spatial Network Analysis Python)
# routines
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


import os
import pickle
from time import time

# importing dependent libraries
import geopandas as gpd
import pandas as pd
import numpy as np

# importing internal scripts
from .main import *

'''
Script for applying routines of processing for projects with certain calculation and else

'''

# MeasureDf is a tabular data with the following input
# 'OriField' - 'DestField'     - ScrDist - weights - *other arguments
# originName - destinationName -  dist   - weights - *other arguments

# for reach
def ReachAggregate(GraphSm:GraphSims, MeasureDf:pd.DataFrame, **kwargs):
    """
    reach aggregate makes comprehensive reach calculations based on the network
    and the set measurements, outputs resulting hasil
    """

    # base settings
    baseSet = {
        'EntID': 'FID',
        'EdgeID': 'FID',
        'OptSuffix': '',
        'OptPrefix': '',
        'DestWgt': 'weight',
        'CalcExp': 0.35,
        'CalcComp': 0.6,
        'CalcType': 'ND'
    }
    for k,v in kwargs.items():
        baseSet[k] = v

    if baseSet['DestWgt'] not in tuple(MeasureDf.columns):
        MeasureDf[baseSet['DestWgt']] = 1.0
    
    OriField = MeasureDf.columns[0]
    DestField = MeasureDf.columns[1]
    if DestField [-2:] == '.1':
        DestField = DestField [:-2]
    print(f'Starting reach aggregate of {len(MeasureDf)} relations')
    n = 0
    ttl = MeasureDf.shape[0]

    for rw in MeasureDf.iterrows():
        n += 1
        rw = rw[1]
        # iterating per rows
        print(f'\t{n}/{ttl}  Reach from {rw[0]} to {rw[1]}')
        OriIds = tuple(GraphSm.EntriesDf.index[GraphSm.EntriesDf[OriField] == rw[0]])
        DesIds = tuple(GraphSm.EntriesDf.index[GraphSm.EntriesDf[DestField] == rw[1]])

        rsltattr = baseSet['OptPrefix']+str(rw[1])+baseSet['OptSuffix']

        # if rsltattr in GraphSm.EntriesDf.columns:
        #     rn = 1
        #     while rsltattr + f'_{rn}' in GraphSm.EntriesDf.columns:
        #         rn += 1
        #     rsltattr += f'_{rn}'

        if len(OriIds) == 0 or len(DesIds) == 0:
            print(f'\t\tOne or more keys not found')   
            continue
        sets = {
            'SearchDist' : rw[2],
            'DestWgt' : rw[3],
            'CalcExp' : baseSet['CalcExp'],
            'CalcComp' : baseSet['CalcComp'],
            'RsltAttr' : rsltattr
        } # settings


        GraphSm.Reach(OriIds, DesIds, baseSet['CalcType'], **sets)

    print('------------------------\nAggregate Reach Completed')
    return GraphSm


def BetweenessPAggregate(GraphSm:GraphSims, PairsDf:pd.DataFrame, MeasureDf:pd.DataFrame=None, **kwargs):
    """
    reach aggregate makes comprehensive reach calculations based on the network
    and the set measurements, outputs resulting hasil
    """

    # base settings
    baseSet = {
        'EntID': 'FID',
        'EdgeID': 'FID',
        'OptSuffix': '',
        'OptPrefix': '',
        'SearchDist': 700,
        'DetourR' : 1.2, 
        'AlphaExp' : 0.35,
    }
    for k,v in kwargs.items():
        baseSet[k] = v

    if MeasureDf is not None:
        print('Appending Sim Weights')
        simcols = []
        TypeMatch = MeasureDf.columns[0]
        for col in MeasureDf.columns[2:]:
            GraphSm.EntriesDf[col] = (0,)*GraphSm.EntriesDf.shape[0]
            simcols.append(col)
            for rw in MeasureDf.iterrows():
                GraphSm.EntriesDf[col] = np.where(
                    GraphSm.EntriesDf[TypeMatch] == rw[1][0],
                    GraphSm.EntriesDf[rw[1][1]]*MeasureDf.at[rw[0], col],
                    GraphSm.EntriesDf[col]
                )
        print('\tSim Weights added')
    
    print(f'Starting Betweeness aggregate of {len(PairsDf)} relations')
    n = 0
    ttl = PairsDf.shape[0]

    for rw in PairsDf.iterrows():
        n += 1
        rw = rw[1]

        rsltattr = baseSet['OptPrefix']+str(rw[0])+baseSet['OptSuffix']

        if rsltattr in GraphSm.EntriesDf.columns:
            rn = 1
            while rsltattr + f'_{rn}' in GraphSm.EntriesDf.columns:
                rn += 1
            rsltattr += f'_{rn}'

        try:
            SrcDist = rw[3]
        except:
            SrcDist = baseSet['SearchDist']

        try:
            DetourR = rw[4]
        except:
            DetourR = baseSet['DetourR']
        
        try:
            AlphaExp = rw[5]
        except:
            AlphaExp = baseSet['AlphaExp']

        # iterating per rows
        print(f'\t{n}/{ttl}  Reach from {rw[1]} to {rw[2]}')
        sets={
            'OriWgt': rw[1],
            'DestWgt' : rw[2],
            'RsltAttr': rsltattr,
            'SearchDist' : SrcDist, 
            'DetourR' : DetourR, 
            'AlphaExp' : AlphaExp
        }

        GraphSm.BetweenessPatronage(**sets)

    print('------------------------\nAggregate Reach Completed')
    return GraphSm