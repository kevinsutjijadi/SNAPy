# SNAPy (Spatial Netowk Analysis Python Module)

A package of spatial network analysis tools based on Geopandas dataframe and networkx pathfinding. Developed for regional scale spatial network analysis in Python environment for efficiency and better documentation. With capabilities of multithreading using multiprocessing.pool. Most functions and tools are based from [Urban Network Analysis Toolbox](https://cityform.mit.edu/projects/urban-network-analysis.html) by MIT City Form Lab, but there are some (Reach, Betweeness Patronage) that have different mathematical expression which is elaborated on the [Documentation](https://github.com/kevinsutjijadi/SNAPy/Documentation) files in this repository.  


## Authorship
made by kevinsutjijadi @2023 Jakarta, Indonesia  
Last updated at 2023/09/12

## Installation

 use the package manager [pip](https://pip.pypa.io/en/stable) to install.

 ```bash
 pip install SNAPy
 ```

## Requirements 
- pandas >= 1.5.2
- geopandas >= 0.9.0
- networkx >= 2.7.1
- scipy >= 1.10.0
- numpy >= 1.24.1
- shapely >= 2.0.0

## How to Use

As Spatial Network Analysis is a derivative development of the Social Science Network Analysis, the conceptualization of the network is adjusted where in the object, path network as the medium is supplemented by end nodes that represents buildings, activity points, or Point of interest of the movement. Note that the polygon data of the network pathway is required to be segmented at each intersection, with maximum inaccuracy tollarance of 1e-3 on junctions, further than that distance will not be detected as a junction. Entrance data are not required to be located on the network lines, as there are built in functions to map entrance points into the network.
Per Function documentation can be accessed in [Documentation](https://github.com/kevinsutjijadi/SNAPy/Documentation)
Most analysis tools included will require a network, consisting of polygon geometry pathway networks, and point geometry entrances in any GIS format readable by geopandas. Some testdata is provided in this repository, which contains a part of Jakarta's network line obtained by OSM API, as the following:  

<img src="Documentation/Resources/SmplNetwork.png" width=50%/>


to use library, load data into geopandas geodataframe format, and use Graphsims to load the network information. It will compile and also append how the entrances connect to the network data.

```Python
import geopandas as pd
import SNAPy as sna

dfNetwork = gpd.readfile('NetworkClean.shp') # network dataframe
dfEntries = gpd.readfile('Features.gpkg', layer='Features') # entrance dataframe

nwSim = sna.GraphSims(dfnw, dfpt) # main class for loading network data
```
to save the projected entries data, or access both network data or entrance data, both dataframes can be called and used as a normal Geopandas geodataframe
```Python
nwSim.EntriesDf.tofile("file.gpkg") # saving entries dataframe
nwSim.NetworkDf.tofile("file.gpkg") # saving network dataframe
```
Example analysis and result of the betweenness Patronage
```Python
nwSim.BetweenessPatronage(OriWgt='Capacity', DestWgt='Weight', DetourR=1.2, SearchDist=1200, AlphaExp=0.1, RsltAttr='BtwnP')
```
with the resulting nwSim.Gdf (can be displayed directly in conjunction with folium) as displayed below

<img src="Documentation/Resources/SmplBtwnP.png" width=50%/>  
  
Performance is designed to be RAM efficient, and more faster and reliable than former counterparts, but undeniably there are still a lot of room for development and streamlining process. Upon trial, using all cores of multithreading, process can be 2-10x faster than UNA Toolbox on Rhino and ArcGIS with the cost of multiple cores used. However single threading is slower. for example the following simulation of betweenness patronage of 279 origins and destinations permutaion, with search distance of 1200m and 1.2 detour ratio took 3'14" of multithread process on a i7-13700K without overclocking. Multiple paths may also cause significant time cost, as the same input with 1.0 detour ratio took only 9" to complete (21x faster)

<img src="Documentation/Resources/SmplBtwnP2.png" width=50%/>  


## References
- Sevtsuk, A., & Mekonnen, M. (2012). Urban Network Analysis Toolbox. International Journal of Geomatics and Spatial Analysis, 22(2), 287–305.
- Sevtsuk, A. (2014). Networks of the built environment. In D. Ofenhuber & C. Ratti (Eds.), Decoding the City: Urbanism in the Age of Big Data (p. 192). Birkhäuser.
- Barrat, A., et al. (2004) The architecture of complex weighted networks
- Haggett, P., & Chorley, J. C. (1969) Network Analysis in Geography. London: Butler & Tanner Ltd.
- Freeman, Linton (1977) A set of measures of centrality based on betweenness, Sociometry.
- Sevtsuk, A (2010) Path and Place: A Study of Urban Geometry and Retail Activity in Cambridge and Somerville, MA.
- Dijkstra, E W (1959) A note on two problems in connexion with graphs.
