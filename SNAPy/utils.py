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
import os

def MultiProcessPool(func, largs:list, threadcount:int=None):
    """
    MultiProcessPool(func:function, inputs:list, threadcount:int=None)\n
    multiprocessing using Pool from multiprocessing, threadcount defaults to os.cpucount - 1\n
    make sure function only takes in one input, single input item is a chunck
    """
    print(f'Multiprocessing {func.__name__}, on {len(largs)} task chunks')
    if threadcount is None: 
        threadcount = os.cpu_count()-1
    with Pool(threadcount) as pool:
        output = pool.imap(func, largs)
        pool.close()
        pool.join()
    return output