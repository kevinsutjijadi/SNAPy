### SNAPy (Spatial Network Analysis Python)
# main script, contains the compiled processings
# Kevin Sutjijadi @ 2023

__author__ = "Kevin Sutjijadi"
__copyright__ = "Copyright 2023, Kevin Sutjijadi"
__credits__ = ["Kevin Sutjijadi"]
__version__ = "0.1.0"

# importing standard libraries
import pandas as pd
import scipy.integrate as integrate
import geopandas as gpd
import numpy as np
from scipy.special import erf

"""
UNA supplemental calculations module, 
providing additional statistics and other processing for results of netx_sim module
"""

### NEEDS:
# convert GISVector into graph
# graph editing
# entrance - exit findings from points to network'

def Func_SkewedDistribution(t, p, loc=0, shp=1, skw=0):
    '''
    Func_SkewedDistribution(t:hour, p:ammount/intensity, loc:location, shp:shape, skw:skew)
    function on skewed distribution
    '''
    pos = (t-loc) / shp
    return p / (shp * np.sqrt(2 * np.pi)) * (np.e **(-0.5 * pos**2)) * 0.5 * (1 + erf(skw * pos / np.sqrt(2)))


def Calc_HourlyTrafficSpread(sets, spread=1, h_start=0, h_end=24, cal_ends=True):
    """
    Frml_Hourly_TrafficSpread(list/tuple of [p:ammount, loc:location, shp:shape, skw:skew], spread:time resolution by hour-default 1, h_start, h_end)
    generating an evenly spaced traffic ammount
    """
    h_sets = [h_start + (spread*n) for n in range(int(round((h_end-h_start)/spread,0)))]
    opt = [0]*len(h_sets)
    if not cal_ends:
        for set in sets:
            for n, h in enumerate(h_sets):
                opt[n] += integrate.quad(lambda x: Func_SkewedDistribution(x, set[0]*2, set[1], set[2], set[3]), h-0.5, h+0.5)[0]
    if cal_ends:
        for set in sets:
            for n,h in enumerate(h_sets):
                opt[n] += integrate.quad(lambda x: Func_SkewedDistribution(x, set[0]*2, set[1], set[2], set[3]), h-24.5, h-23.5)[0]
                opt[n] += integrate.quad(lambda x: Func_SkewedDistribution(x, set[0]*2, set[1], set[2], set[3]), h-0.5, h+0.5)[0]
                opt[n] += integrate.quad(lambda x: Func_SkewedDistribution(x, set[0]*2, set[1], set[2], set[3]), h+23.5, h+24.5)[0]
    return opt


def SimTimeDistribute(Gdf, SetDt, spread=1, ApdAtt='HrTrf_'):
    """
    Calc_Traffic(Gdf, SetDt, spread=1)
    calculate segments and datas
    """
    dfcrs = Gdf.crs.to_epsg()
    Opt = [list(Gdf['geometry']),]
    for i in Gdf.index:
        Opt.append(Calc_HourlyTrafficSpread(
            tuple((Gdf.at[i, s[0]], s[1], s[2], s[3]) for s in SetDt), 
            spread
            ))
    Ocl = list(f'{ApdAtt}{n*spread}' for n in range(len(Opt[0])))
    Ocl = ['geometry'] + Ocl
    Odf = gpd.GeoDataFrame(Opt, columns=Ocl, crs=f'EPSG:{dfcrs}')
    return Odf