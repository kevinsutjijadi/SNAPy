---
title: Functions
description: Spatial Network Analysis for Python Module
---

# Functions and Classes

Functions and classes that are contained within the library

## GraphSims   <i> class </i>
`#!python GraphSims(NetworkDf:GeoDataFrame, EntriesDf:GeoDataFrame, **kwargs)`

:   The core process for spatial network analysis is operated/structured within the `#!python Graphsims` Class, which reads Geopandas Geodataframe for entries and network data. The class initialization prepackages and compiles further processing and its subsequent results. main class for network analysis processing, initialization stage will have some heavy processes included. the processes in order

    - adding Feature-ID of network and entrances as attribute if not found
    - building networkx graph using BuildGraph from prcs_grph
    - building entries data, appending with entrance/relation to graph related information.

        !!! info "entries data dumping feature"
            Since processing into the graph relation can take some time, there is the feature that saves and reads dump data from previous run through. The settings of this feature will be explained later at kwargs parameter.
    
    - appending point coordinate as attributes unto EntriesDf



    ##### Parameters

    :   <b>NetworkDf</b> : GeoDataFrame
        :   Geopandas geodataframe of network data, containing edges in lines/polylines data form. Can contain weight multiplier for costs and also vertex classification, with the default 

    :   <b>EntriesDf</b> : GeoDataFrame
        :   Geopandas geodataframe of Entries data, containing nodes in lines/polylines data form. Can contain weight

    :   <b>**kwargs</b> : Dict/keys-values
        :   Parameter arguments, that contain:

            - `#!python "EntDist" float `   default 100.0

                search distance to closest edge of network, further distanced entrances will not be accounted
            
            - `#!python "EntID" string `   default "FID"

                string of attribute/column name for Entrance Feature ID. If not found, will be automatically made in initialization process

            - `#!python "EdgeID" string `   default "FID"

                string of attribute/column name for Network Feature ID. If not found, will be automatically made in initialization process
            
            - `#!python "EdgeCost" string `   default None

                string of attribute/column name for Network cost multiplier, if None, all edges will use the value of "EdgeCostDef" argument
            
            - `#!python "EdgeCostDef" string `   default 1.0

                default value for edge cost, will only be applied if "EdgeCost" is None

            - `#!python "EntryDtDump" bool `   default True

                Switch to use entries data initialization results dumping. If true, then it will search for a pre-existing data dump, or if not found will make one at "Directory" argument location.

            - `#!python "EntryDtDumpOvr" bool `   default False

                Switch to overide pre-existing entries data dump. Only works if "EntryDtDump" is True. If there are updates on network or entries data, this argument is used.
            
            - `#!python "Directory" string `   default "\\dump"

                entries data relation dump location.
            
            - `#!python "Threads" int `   default 0

                specify threads number for multiprocessing. For single core processing, input 1. The default value is 0, if which, will be replaced with (CPU count - 1), which almost fully utilizes CPU processing.

    ##### Use Example
    :
        ```python
        import SNAPy as sna
        import geopandas as gpd

        dfNetwork = gpd.read_file('testdata\\NetworkClean.shp') # network dataframe
        dfEntries = gpd.read_file('testdata\\Features.gpkg', layer='Features') # entrance dataframe
        
        settings = {
            'EntID': 'ID',
            'EdgeID': 'ID',
            'EdgeCost': 'Cost',
            'EntryDtDumpOvr': True,
            'Threads': 8,
        }

        nwSim = sna.GraphSims(dfNetwork, dfEntries, settings) # main class for loading network data
        ```

### GraphSims <i>Items</i>

:   #### .baseSet <i> dict </i>

    :   Dictionary of settings

        Dictionary of processing settings used in initialization or other functions. Complete elements of the dictionary can be read/found at class parameter description.


:   #### .EntriesDf <i> GeoDataFrame </i>

    :   GeoDataFrame of entries from input. From initialization process or reparametrization will be automatically appended intersection point coordinate as 'xPt_X' and 'xPt_Y'.

        built-in entries point based functions such as .Reach, etc will save their results in a specified field name. Which then the variable can be obtained and modified as a GeoDataFrame, which can be edited, saved, filtered.


:   #### .EntriesPt <i> Tuple </i>

    :   contains a nested tuple of each point with information of the matching EntriesDf poin with the graph. Tuple structure of each point contains:
        
        - EntryDf point ID
        - connected line/edge ID
        - distance from entry point to intersection
        - tuple of distance to the start and end nodes
        - point of intersection
        - tuple of connected network node ids
        - weight of edge
        - tuple of spliited shapely point

:   #### .Gph <i> Graph </i>

    :   networkx Graph object from class initialization or reparametrization

:   #### .EntPtDumpDir <i> string </i>

    :   File name for dump file


### GraphSims <i>Functions</i>

:   #### .BetweenessPatronage <i> func </i>
    `#!python BetweenessPatronage(self, OriID:Tuple=None, DestID:Tuple=None, **kwargs)`

    :   returns self.NetworkDf : GeoDataFrame <br>
        <i>GraphSims.EntriesDf with the specified results column</i>

        Function for betweeness patronage, calculating segment traffic weighting from origin capacity, distributed through weightable destinations within the network in a set distance. Further elaboration on the method can be accessed on method documentation.

        !!! info "result return"
            results will be appended to self.NetworkDf, therefore the results can be accesed later so that the function does not need to be assigned into a new variable.
        
        !!! warning "processing duration"
            function has built in multithreading capabilities using multiprocessing.Pool. Please take notice of self.baseSet['Threads'], as the value is also used in this function.
            Larger models, larger search distance, larger detour ratio will be cause exponentially longer processing time.

    ##### Parameters

    :   <b>OriID</b> : Tuple <i>default None</i>
        :   Tuple of Origin point IDs, referencing `#!python GraphSims.EntriesDf` ID attribute specified at class initialization. If None, all 

    :   <b>DestID</b> : Tuple <i>default None</i>
        :   Geopandas geodataframe of Entries data, containing nodes in lines/polylines data form. Can contain weight

    :   <b>**kwargs</b> : Dict/keys-values
        :   Parameter arguments, that contain:

            - `#!python "OriWgt" string `   default 'weight'

                name of the attribute from self.EntriesDf to access weight value for origin points. In the context of Betweeness Patronage, the value will represent the trip unit/person.
                There is a feature that skips any processing if the weighting value is 0.0

                !!! warning "default values"
                    if the attribute name is not found, there will be an automatic adjustment/appandage with the default value of 1
            
            - `#!python "DestWgt" string `   default "weight"

                name of the attribute from self.EntriesDf to access weight value for destination points. In the context of Betweeness Patronage, the value will represent destination preference.
                There is a feature that skips any processing if the weighting value is 0.0
                !!! warning "default values"
                    if the attribute name is not found, there will be an automatic adjustment/appandage with the default value of 1
            
            - `#!python "RsltAttr" string `   default "PatronBtwns"

                Attribute name for the result. will be appended to self.NetworkDf.
            
            - `#!python "SearchDist" float `   default 1200.0

                maximum path distance, corresponding to the data's units
            
            - `#!python "DetourR" float `   default 1.0

                detour ratio for redundant paths, the value will represent the maximum redundant/alternative path length from the shortest path within the maximum search distance. Default value is 1.0 which is provided a switch to only find one shortest path. (if two exact shortest path exist, the pathfinding algorithm will use the smalled FID edge number).
            
            - `#!python "AlphaExp" float `   default 0.0

                Exponent value for inverse distance function. if using default value of 0.0, the calculation turns into a linear inverse distance function.
    
    ##### Use Example
    :   
        ```python
        ### continuation from class initialization
        ### nwSim = sna.GraphSims()


        nwSim.BetweenessPatronage(OriWgt='Capacity', DestWgt='Weight', SearchDist=700, DetourR=1.1) # results does not need to be contained
        nwSim.NetworkDf.sample # results can be accessed from the NetworkDf attribute
        ```

        Function can be run in loops

        ```python
        ### continuation from class initialization
        ### nwSim = sna.GraphSims()

        runs = (
            ('Ent01', 'Dst02', 350.0, 1.0, 'run01'),
            ('Ent02', 'Dst02', 700.0, 1.2, 'run02'),
            ('Ent03', 'Dst03', 700.0, 1.0, 'run03'),
        )

        for rn in runs:
            nwSim.BetweenessPatronage(OriWgt=rn[0], DestWgt=rn[1], SearchDist=rn[2], DetourR=rn[3], RsltAttr=rn[4]) # results does not need to be contained
        nwSim.NetworkDf.to_file("output.gpkg", layer="Entries", crs="EPSG:32748", driver='GPKG') # geoDataFrame can be saved to GIS files
        ```

:   #### .Reach <i> func </i>
    `#!python Reach(self, OriID:Tuple=None, DestID:Tuple=None, Mode:str='N', **kwargs)`

    :   returns self.EntriesDf : GeoDataFrame <br>
        <i>GraphSims.EntriesDf with the specified results column</i>

        Function for Reach related calculations. Further elaboration on the method can be accessed on method documentation. Contains multiple Modes:
        
        - Mode="N" : Count Sum
            for counting destinations within reach distance, results in one new attribute of interger numbers

        - Mode="W" : Weighted Sum
            for counting and summing value based on specific weights of each destination feature, results in two attribute (with the second has a suffix of "_2"). First is interger count, second is the summed weight

        - Mode="WD" : Weighted Distance Sum
            for counting and summing value based on specific weights adjusted with inverse distance function of each destination feature, results in two attribute (with the second has a suffix of "_2"). First is interger count, second is the summed weight

        - Mode="ND" : Count and Distance
            for counting and finding nearest distance to destination. Results in two attribute (with the second has a suffix of "_2"). First is interger count, second is the minimum distance.
        
        - Mode="NDW" : Count, min Distance, and Weight Sum
            for counting and finding nearest distance to destination, and also sum of weight to all destinations in reach. Results in three attribute (with the minimum distance suffix of "_D", and Weight Sum of "_W").

        !!! info "result return"
            results will be appended to self.EntriesDf, therefore the results can be accesed later so that the function does not need to be assigned into a new variable.
        
        !!! warning "processing duration"
            function has built in multithreading capabilities using multiprocessing.Pool. Please take notice of self.baseSet['Threads'], as the value is also used in this function.
            Larger models, larger search distance, will be cause exponentially longer processing time.

    ##### Parameters

    :   <b>OriID</b> : Tuple <i>default None</i>
        :   Tuple of Origin point IDs, referencing `#!python GraphSims.EntriesDf` ID attribute specified at class initialization. If None, all 

    :   <b>DestID</b> : Tuple <i>default None</i>
        :   Geopandas geodataframe of Entries data, containing nodes in lines/polylines data form. Can contain weight

    :   <b>Mode</b> : string <i>default "N"</i>
        :   modes of calculation, see function description.

    :   <b>**kwargs</b> : Dict/keys-values
        :   Parameter arguments, that contain:
            
            - `#!python "DestWgt" string `   default "weight"

                name of the attribute from self.EntriesDf to access weight value for destination points. In the context of Betweeness Patronage, the value will represent destination preference.
                There is a feature that skips any processing if the weighting value is 0.0
                !!! warning "default values"
                    if the attribute name is not found, there will be an automatic adjustment/appandage with the default value of 1
            
            - `#!python "RsltAttr" string `   default "PatronBtwns"

                Attribute name for the result. will be appended to self.NetworkDf.
            
            - `#!python "SearchDist" float `   default 1200.0

                maximum path distance, corresponding to the data's units
            
            - `#!python "CalcExp" float `   default 0.35

                Exponent value for inverse distance function. if using default value of 0.0, the calculation turns into a linear inverse distance function. only for "WD" mode.

            - `#!python "CalcComp" float `   default 0.6

                Compunding multiplier for "WD" mode.
    
    ##### Use Example

    :   
        ```python
        ### continuation from class initialization
        ### nwSim = sna.GraphSims()


        nwSim.Reach(Mode='WD", SearchDist=700, CalcExp=0.1) # results does not need to be contained
        nwSim.EntriesDf.sample # results can be accessed from the EntriesDf attribute
        ```

:   #### .Straightness <i> func </i>
    `#!python Straightness(self, OriID:list=None, DestID:list=None, **kwargs)`

    :   returns self.EntriesDf : GeoDataFrame <br>
        <i>GraphSims.EntriesDf with the specified results column</i>

        Returns average of Straightness value from all objects is flight distance from origin point. Only accounts of objects with findable paths within the network. Untracable paths will not be accounted. Further elaboration on the method can be accessed on method documentation.

        !!! info "result return"
            results will be appended to self.EntriesDf, therefore the results can be accesed later so that the function does not need to be assigned into a new variable.
        
        !!! warning "processing duration"
            function has built in multithreading capabilities using multiprocessing.Pool. Please take notice of self.baseSet['Threads'], as the value is also used in this function.
            Larger models, larger search distance, will be cause exponentially longer processing time.

    ##### Parameters

    :   <b>OriID</b> : Tuple <i>default None</i>
        :   Tuple of Origin point IDs, referencing `#!python GraphSims.EntriesDf` ID attribute specified at class initialization. If None, all 

    :   <b>DestID</b> : Tuple <i>default None</i>
        :   Geopandas geodataframe of Entries data, containing nodes in lines/polylines data form. Can contain weight

    :   <b>**kwargs</b> : Dict/keys-values
        :   Parameter arguments, that contain:
            
            - `#!python "DestWgt" string `   default "weight"

                name of the attribute from self.EntriesDf to access weight value for destination points. In the context of Betweeness Patronage, the value will represent destination preference.
                There is a feature that skips any processing if the weighting value is 0.0
                !!! warning "default values"
                    if the attribute name is not found, there will be an automatic adjustment/appandage with the default value of 1
            
            - `#!python "RsltAttr" string `   default "PatronBtwns"

                Attribute name for the result. will be appended to self.NetworkDf.
            
            - `#!python "SearchDist" float `   default 1200.0

                maximum path distance, corresponding to the data's units
            
            - `#!python "CalcExp" float `   default 0.35

                Exponent value for inverse distance function. if using default value of 0.0, the calculation turns into a linear inverse distance function. only for "WD" mode.

            - `#!python "CalcComp" float `   default 0.6

                Compunding multiplier for "WD" mode.
    
    ##### Use Example

    asdf
    :   
        ```python
        ### continuation from class initialization
        ### nwSim = sna.GraphSims()


        nwSim.Reach(Mode='WD", SearchDist=700, CalcExp=0.1) # results does not need to be contained
        nwSim.EntriesDf.sample # results can be accessed from the EntriesDf attribute
        ```


<br><br>
@October2023