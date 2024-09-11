### SNAPy (Spatial Network Analysis Python)
# utility scripts
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

from multiprocessing import Pool
from shapely.geometry import LineString, MultiLineString, Point, mapping, shape
import os
import numpy as np

def MultiProcessPool(func, largs:list, threadcount:int=0):
    """
    MultiProcessPool(func:function, inputs:list, threadcount:int=None)\n
    multiprocessing using Pool from multiprocessing, threadcount defaults to os.cpucount - 1\n
    make sure function only takes in one input, single input item is a chunck
    """
    print(f'Multiprocessing {func.__name__}, on {len(largs)} task chunks')
    if threadcount == 0 or threadcount > os.cpu_count(): 
        threadcount = os.cpu_count()-1
    with Pool(threadcount) as pool:
        output = pool.imap(func, largs)
        pool.close()
        pool.join()
    return output

def colorBR(value):
    return (255,0,0) if value else (0,0,255)

def getcoords(point:Point, z=2):
    if point.has_z:
        return (point.x, point.y, point.z+z)
    else:
        return (point.x, point.y, z)

def NumStringFormat(value, rounding=1) ->str:
    return f"{value:,.{rounding}f}"

def repassColor(foo):
    return (foo[0], foo[1], foo[2])

def rgb_to_hex(rgb_tuple):
    return '#{:02x}{:02x}{:02x}'.format(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2])


def ColorRampMapping(values:np.ndarray, colors:str|list='viridis', vmin=None, vmax=None, distribution='decile', **kwargs):
    """
    Create a color map based on input values and a list of colors.
    
    :param values: 1D numpy array of values to be mapped to colors
    :param colors: List of RGB tuples (values from 0-255), or string 'viridis' or 'spectral'
    :return: numpy array of RGB values (shape: len(values) x 3)
    """
    if type(colors) == str:
        match colors.lower():
            case 'viridis':
                colors = [(68, 1, 84),  # Color 1
                    (72, 40, 120),  # Color 2
                    (62, 74, 137),  # Color 3
                    (49, 104, 142),  # Color 4
                    (38, 130, 142),  # Color 5
                    (53, 183, 121),  # Color 6
                    (253, 231, 37),  # Color 7
                ]
            case 'spectral':
                colors = [
                    (50, 12, 105),  # Color 7
                    (43, 131, 186),  # Color 6
                    (171, 221, 164),  # Color 5
                    (253, 174, 97),  # Color 4
                    (244, 109, 67),  # Color 3
                    (213, 62, 79),  # Color 2
                    (158, 1, 66),  # Color 1    
                ]
            case _:
                print('color ramp not detected, defaulting to viridis')
                colors = [(68, 1, 84),  # Color 1
                    (72, 40, 120),  # Color 2
                    (62, 74, 137),  # Color 3
                    (49, 104, 142),  # Color 4
                    (38, 130, 142),  # Color 5
                    (53, 183, 121),  # Color 6
                    (253, 231, 37),  # Color 7
                ]
    # Normalize the input colors to 0-1 range
    if vmin is None:
        vmin = values.min()
    else:
        values[values<vmin] = vmin
    if vmax is None:
        vmax = values.max()
    else:
        values[values>vmax] = vmax
    
    match distribution:
        case 'linear':
            pass
        case 'quantile'|'decile':
            if distribution == 'quantile':
                ptl = np.array((0, 25, 50, 75, 100))
            else:
                ptl = np.array((0, 10,20,30,40,50,60,70,80,90,100))
            dis = np.percentile(values, ptl)
            vals = values
            v_binmap = np.clip(np.abs(vals[:,np.newaxis] - dis).argmin(axis=1), 0, len(dis)-2)
            issmaller = dis[v_binmap] > vals
            values = v_binmap.astype(np.float16) + (vals - dis[v_binmap-issmaller])/(dis[v_binmap+1-issmaller] - dis[v_binmap-issmaller])
            vmin = 0.0
            vmax = len(ptl)-1
        case _:
            print('distribution case not detecting, reveting to linear')
    colors.append(colors[-1])
    colors = np.array(colors)
    
    vrange = vmax-vmin
    # Create color map
    n_bins = len(colors) - 2
    vloc = np.clip((values - vmin)/vrange * n_bins, 0, n_bins)
    np.nan_to_num(vloc, False, nan=0)
    v_binmap = colors[(vloc//1).astype(int)]

    return (v_binmap + (colors[((vloc//1)+1).astype(int)] - v_binmap)*(vloc[:, np.newaxis]%1)).astype(int)

def ValueRampMapping(values:np.ndarray, vmin=None, vmax=None, Omin=1.0, Omax=5.0, distribution='decile', **kwargs):
    if vmin is None:
        vmin = values.min()
    else:
        values[values<vmin] = vmin
    if vmax is None:
        vmax = values.max()
    else:
        values[values>vmax] = vmax
    
    match distribution:
        case 'linear':
            pass
        case 'quantile'|'decile':
            if distribution == 'quantile':
                ptl = np.array((0, 25, 50, 75, 100))
            else:
                ptl = np.array((0, 10,20,30,40,50,60,70,80,90,100))
            dis = np.percentile(values, ptl)
            vals = values
            v_binmap = np.clip(np.abs(vals[:,np.newaxis] - dis).argmin(axis=1), 0, len(dis)-2)
            issmaller = dis[v_binmap] > vals
            values = v_binmap.astype(np.float16) + (vals - dis[v_binmap-issmaller])/(dis[v_binmap+1-issmaller] - dis[v_binmap-issmaller])
            vmin = 0.0
            vmax = len(ptl)-1
        case _:
            print('distribution case not detecting, reveting to linear')
    
    vrange = vmax-vmin
    # Create color map
    Orange = Omax - Omin

    return Omin + Orange*(values-vmin)/vrange
