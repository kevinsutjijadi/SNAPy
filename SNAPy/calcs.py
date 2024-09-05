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


def Func_SkewedDistribution(t, p, loc=0, shp=1, skw=0):
    '''
    Func_SkewedDistribution(t:hour, p:ammount/intensity, loc:location, shp:shape, skw:skew)
    function on skewed distribution
    '''
    pos = (t-loc) / shp
    return p / (shp * np.sqrt(2 * np.pi)) * (np.e **(-0.5 * pos**2)) * (1 + erf(skw * pos / np.sqrt(2)))


def Calc_HourlyTrafficSpread(sets, spread=1, h_start=0, h_end=24, cal_ends=False):
    """
    Frml_Hourly_TrafficSpread(list/tuple of [p:np.array of btwns value, [loc:location, shp:shape, skw:skew]], spread:time resolution by hour-default 1, h_start, h_end)
    generating an evenly spaced traffic ammount
    """
    h_sets = np.array([h_start + (spread*n) for n in range(int(round((h_end-h_start)/spread,0)))])
    opt = np.zeros((sets[0][0].shape[0], h_sets.shape[0]), dtype=np.float32)
    for values, param in sets:
        intfunc = np.vectorize(lambda z, h: integrate.quad(lambda x: Func_SkewedDistribution(x, z, param[0], param[1], param[2]), h-0.5, h+0.5)[0])
        intfuncY = np.vectorize(lambda z, h: integrate.quad(lambda x: Func_SkewedDistribution(x, z, param[0], param[1], param[2]), h-24.5, h-23.5)[0])
        intfuncT = np.vectorize(lambda z, h: integrate.quad(lambda x: Func_SkewedDistribution(x, z, param[0], param[1], param[2]), h+23.5, h+24.5)[0])

        opt += intfunc(values[:,np.newaxis], h_sets)
        if cal_ends:
            opt += intfuncY(values[:,np.newaxis], h_sets)
            opt += intfuncT(values[:,np.newaxis], h_sets)
    return opt


def SimTimeDistribute(Gdf, SetDt, spread=1, ApdAtt='HrTrf_', h_start=0, h_end=24, cal_ends=False, returnArray=False):
    """
    Calc_Traffic(Gdf, SetDt[arr: attribute name on gdf, loc:location, shp:shape, skw:skew] , spread=1)
    calculate segments and datas
    """

    Ocl = list(f'{ApdAtt}{n*spread}' for n in range(int(round(24/spread,0))))
    ncol = len(Gdf.columns)
    GdfC = Gdf.copy()
    for cl in Ocl:
        GdfC[cl] = [0,] * len(GdfC)

    sets = [(np.array(GdfC[s[0]]), (s[1], s[2], s[3])) for s in SetDt]
    rslt = Calc_HourlyTrafficSpread(sets, spread, h_start, h_end, cal_ends)
    # masukin ke anu gdf
    if returnArray:
        return Ocl, rslt
    for n, cl in enumerate(Ocl):
        GdfC[cl] = rslt[:,n]

    return GdfC