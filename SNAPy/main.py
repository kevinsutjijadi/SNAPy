### SNAPy (Spatial Network Analysis Python)
# main script, contains the compiled processings
# Kevin Sutjijadi @ 2023

__author__ = "Kevin Sutjijadi"
__copyright__ = "Copyright 2023, Kevin Sutjijadi"
__credits__ = ["Kevin Sutjijadi"]
__version__ = "0.2.13"

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
import pydeck as pdk
import warnings

# importing internal scripts
# from .prcs_geom import *
from .prcs_grph import *
from .prcs_sna import *
from .utils import *
from .SGACy.graph import GraphCy
from .SGACy.geom import multilinestring_to_linestring, NetworkCompileIntersections, NetworkSegmentIntersections


### packed functions for multiprocessing
class GraphSims:
    def __init__(self, NetworkDf:gpd.GeoDataFrame, EntriesDf:gpd.GeoDataFrame, **kwargs):
        """
        GraphSims(Gph, Entries)\n
        main class for una simulations, appending destinations\n
        has built-in multithreading. \n
        kwargs included: EntDist, EntID, Verbose, Threads
        """

        # base settings
        self.baseSet = {
            'EntDist': 100,
            'EntID': 'fid',
            'EdgeID': 'fid',
            'AE_Lnlength': None,
            'AE_LnlengthR': None,
            'AN_EdgeCost': None,
            'SizeBuffer': 0.05,
            'Verbose': True,
            'Threads': 0,
            'Warnings':True,
        }
        if self.baseSet['Warnings']:
            warnings.filterwarnings('ignore')
        for k,v in kwargs.items():
            self.baseSet[k] = v
        if self.baseSet['Threads'] == 0:
            self.baseSet['Threads'] = os.cpu_count()-1

        self.epsg = NetworkDf.crs.to_epsg()

        issues = []
        if EntriesDf.crs.is_geographic:
            issues.append("EntriesDf projection is geographic (degrees), please conver to projected")
        if NetworkDf.crs.is_geographic:
            issues.append("NetworkDf projection is geographic (degrees), please conver to projected")
        if EntriesDf.crs.to_epsg() != self.epsg:
            issues.append("EntriesDf and NetworkDf have different projections, please match them using gdb.to_crs()")
        if tuple(NetworkDf.geometry.type.unique()) != ('LineString',):
            if 'MultiLineString' in tuple(NetworkDf.geometry.type):
                try:
                    print('DataFrame contains MultiLineString, exploding to LineString')
                    NetworkDf = multilinestring_to_linestring(NetworkDf)
                except:
                    issues.append("NetworkDf geometry type contains other than LineString and/or MultiLineString, convert it to LineString first")
            else:
                issues.append("NetworkDf geometry type contains other than LineString and/or MultiLineString, convert it to LineString first")
        if tuple(EntriesDf.geometry.type.unique()) != ('Point',):
            issues.append("EntriesDf geoemtry type is not Point, convert it to point")
        if len(issues) > 0:
            for n, i in enumerate(issues):
                print(f'{n}\t'+i)
            raise Exception("SNAPy init failed, resolve the issues to continue")
        
        self.NodeDf = None

        ixpt = NetworkCompileIntersections(NetworkDf)
        if (np.sum(ixpt['JunctCnt'] == 1)/len(ixpt)) > 0.3:
            print("Warning, more than 30% of endpoints are dead ends, segment intersections first?")
            inpt = input('1/3   segment intersections: (y/[n]) (default n-no)')
            if inpt == 'y':
                extln = input('2/3  Extend line ends? (default 0.0)')
                if extln == '': 
                    extln = 0.0 
                else: 
                    try: extln = float(extln)
                    except: extln = 0.0

                dpass = input('3/3  DoublePass check? (y/[n]) (default n-no)')
                if dpass == 'y': 
                    dpass = True 
                else: 
                    dpass = False

                print("segmenting intersections")
                NetworkDf, self.NodeDf = NetworkSegmentIntersections(NetworkDf, True, ExtendLines=extln, DoublePass=dpass)
                NetworkDf = NetworkDf.set_crs(self.epsg, allow_override=True)
                self.NodeDf = self.NodeDf.set_crs(self.epsg, allow_override=True)
                print("Access segmented network at GraphSims.NetworkDf and GraphSims.ixDf, recommended to save both dataframes, future GraphSims runs use segmented datasets")
            else:
                print("ignoring segmentation")

        self.EntriesDf = EntriesDf
        self.LastAtt = None
        
        print(f'GraphSim Class ----------')
        print(f'Projection EPSG:{self.epsg}')

        if self.baseSet['EntID'] not in self.EntriesDf.columns:
            print('EntriesDf EntID not detected, adding from index')
            self.EntriesDf[self.baseSet['EntID']] = range(self.EntriesDf.shape[0])

        if self.baseSet['EdgeID'] not in NetworkDf.columns:
            print('NetworkDf EdgeID not detected, adding from index')
            NetworkDf[self.baseSet['EdgeID']] = range(int(NetworkDf.shape[0]))
    
        tmst = time()
        nwSz = NetworkDf.shape[0]
        self.Gph = GraphCy(int(nwSz*(2+self.baseSet['SizeBuffer'])), int(nwSz*(1+self.baseSet['SizeBuffer'])), len(EntriesDf))
        self.Gph.fromGeopandas_Edges(NetworkDf,
                                    self.baseSet['AE_Lnlength'],
                                    self.baseSet['AE_LnlengthR'],
                                    )
        # future settings
        # self.baseSet['A_LnW'],
        # self.baseSet['A_PtstW'],
        # self.baseSet['A_PtstC'],
        # self.baseSet['A_PtedW'],
        # self.baseSet['A_PtedC'],
        self.NetworkSize = self.Gph.sizeInfo() # returns (nodesize, edgesize)
        print(f'Graph Built with {self.NetworkSize[0]:,} Nodes, {self.NetworkSize[1]:,} Edges')
        self.NetworkDf = NetworkDf
        EntriesDt = graph_addentries(self.NetworkDf, EntriesDf, self.baseSet['EntDist'], self.baseSet['EntID'], self.baseSet['EdgeID'],  self.baseSet['AN_EdgeCost'])
        print(f'Graph mapped {len(EntriesDt):,} Entries')
        
        entryfails = EntriesDt[EntriesDt['lnID'] == -1]
        if entryfails.shape[0] > 0:
            print(f'Detected {entryfails.shape[0]} Entries unable to map to network. Coordinate error or further than EntDist parameter')
            print(f"\tEntry ids are :\n\t{list(entryfails['fid'])}")
        
        self.Gph.frompandas_Entries(EntriesDt)
        self.EntriesDf['xLn_ID'] = EntriesDt['lnID']
        self.EntriesDf['xPt_X'] = [p[0] for p in EntriesDt['ixPt']]
        self.EntriesDf['xPt_Y'] = [p[1] for p in EntriesDt['ixPt']]

        # map!
        self.pdkLayers = []
        self.pdkLyrNm = []
        self.pdkCenter = None
        print(f'Graph initialization finished in {time()-tmst:,.4f} s')


    def __repr__(self) -> str:
        strNwSim = f'GraphSim object of {len(self.NetworkDf)} Segments and {len(self.EntriesDf)} Entries'
        return strNwSim

    def getNodes(self, reset=False) -> gpd.GeoDataFrame:
        if self.NodeDf is None or reset:
            self.NodeDf = self.Gph.getNodes(self.epsg)
        return self.NodeDf
    
    def Threads(self, value:int):
        if value is None:
            return
        if value == 0:
            self.baseSet['Threads'] = os.cpu_count()-1
        else:
            self.baseSet['Threads'] = value

    def Map_BaseLayerInit(self, ContainFields=False):
        """
        Map_LayerBuild
        builds pydeck map base layers consisting of network and entries
        """
        if not ContainFields:
            NetDf = self.NetworkDf[['geometry', self.baseSet['EdgeID']]].copy()
            EntXDf = self.EntriesDf[['geometry', self.baseSet['EntID'], 'xPt_X', 'xPt_Y']].copy()
            EntDf = self.EntriesDf[['geometry', self.baseSet['EntID']]].copy()
        else:
            NetDf = self.NetworkDf.copy()
            EntXDf = self.EntriesDf.copy()
            EntDf = self.EntriesDf.copy()

        NetDf = NetDf.to_crs(4326)
        lyrNetwork = pdk.Layer(
            type="GeoJsonLayer",
            data=NetDf,
            pickable=True,
            get_line_color=[80,80,80],
            get_line_width=0.8,
        )

        EntXDf['geometry'] = EntXDf.apply(lambda x: LineString(((x.geometry.x, x.geometry.y), (x.xPt_X, x.xPt_Y))), axis=1)
        EntXDf = EntXDf.to_crs(4326)
        lyrEntriesX = pdk.Layer(
            type="GeoJsonLayer",
            data=EntXDf,
            pickable=False,
            get_line_color=[255,80,80],
            get_line_width=0.5,
        )

        EntDf = EntDf.to_crs(4326)
        lyrEntries = pdk.Layer(
            type="GeoJsonLayer",
            data=EntDf,
            pickable=True,
            get_fill_color=[255,80,80],
            get_line_width = 0,
            get_radius=1.5
        )
        self.pdkCenter = NetDf.geometry.unary_union.centroid
        print('built base layers on self.pdfLayers. indexed at: 0-base network layer, 1-entries line connection to edges, 2-entries')
        print('able to pop, edit ,or add more layers by directly accessing self.pdfLayers before using self.ShowMap\n')

        self.pdkLayers = [lyrNetwork, lyrEntriesX, lyrEntries]
        self.pdkLyrNm = ['Ntw_Edges', 'Ntw_EntriesX', 'Ntw_Entries']
    
    def Map_PopLayer(self, name:int|str|None=None):
        """
        Map_LayerBuild
        builds pydeck map base layers consisting of network and entries
        """
        if name is None:
            print(f'existing layers: {self.pdkLyrNm}')
            ly = input('input layer name/index: (default enter:skip)')
            if ly is None:
                return
            try:
                name = int(ly)
            except:
                name = ly

        if isinstance(name, int):
            try:
                self.pdkLayers.pop(name)
                self.pdkLyrNm.pop(name)
            except:
                print(f'index {name} doesnt exists')
        else:
            try:
                name = self.pdkLyrNm.index(name)
                self.pdkLayers.pop(name)
                self.pdkLyrNm.pop(name)
            except:
                print(f'layer named {name} does not exist')
        print(f'remaining layers: {self.pdkLyrNm}')
        return

    def Map_LayerAdd(self, layers:dict):

        if len(self.pdkLayers) == 0:
            self.Map_BaseLayerInit()
        
        if 'junction' in layers and 'Ntw_Nodes' not in self.pdkLyrNm:
            if self.NodeDf is None:
                self.getNodes()
            nodes = self.NodeDf.copy().to_crs(4326)
            nodes['DeadEnd'] = nodes['JunctCnt'] < 2
            nodes['color'] = nodes['DeadEnd'].apply(colorBR)
            ly = pdk.Layer(
                    type="GeoJsonLayer",
                    data=nodes,
                    pickable=True,
                    get_fill_color='color',
                    get_line_width = 0,
                    get_radius=1.5,
                )
            self.pdkLayers.append(ly)
            self.pdkLyrNm.append('Ntw_Nodes')
            nodes['coords'] = nodes.geometry.apply(getcoords)
            nodes['JunctCnt'] = nodes['JunctCnt'].astype('string')
            ly = pdk.Layer(
                    type="TextLayer",
                    data=nodes,
                    pickable=False,
                    get_position='coords',
                    get_text='JunctCnt',
                    get_size=14,
                    get_color='color',
                    background=True,
                    get_background_color = [255, 255, 255, 180],
                    get_text_anchor=pdk.types.String("middle"),
                    get_alignment_baseline=pdk.types.String("center"),
                )
            self.pdkLayers.append(ly)
            self.pdkLyrNm.append('Ntw_NodesLbl')
            print('Map layers added junctions info')
            layers.pop('junction')
        # other layers
        if len(layers) == 0:
            return
        colsNetwork = tuple(self.NetworkDf.columns)
        colsEntries = tuple(self.EntriesDf.columns)
        for ly, st in layers.items():
            if ly+'_Ntw' in self.pdkLyrNm or ly+'_Ent' in self.pdkLyrNm:
                continue
            if st is None:
                st = {}
            if 'label' not in st:
                st['label'] = True
            if 'labelsize' not in st:
                st['labelsize'] = 9
            if ly in colsNetwork:
                # if output attribute name in networkdf columns
                dt = self.NetworkDf[['geometry', ly]].copy()
                dt = dt.to_crs(4326)
                dt['color'] = [(int(x[0]), int(x[1]), int(x[2]), 120) for x in ColorRampMapping(np.array(dt[ly]), **st)]
                dt['linewidth'] = ValueRampMapping(np.array(dt[ly]), **st)
                lyr = pdk.Layer(
                    type="GeoJsonLayer",
                    data=dt,
                    pickable=False,
                    get_line_color='color',
                    get_line_width='linewidth',
                )
                self.pdkLayers.append(lyr)
                self.pdkLyrNm.append(ly+'_Ntw')
                if st['label']:
                    dt['coords'] = dt.geometry.interpolate(0.5, normalized=True).apply(getcoords)
                    dt[ly] = dt[ly].apply(NumStringFormat)
                    lyr = pdk.Layer(
                        type="TextLayer",
                        data=dt,
                        pickable=False,
                        get_position='coords',
                        get_text=ly,
                        get_size=st['labelsize'],
                        get_color=[0,0,0],
                        background=True,
                        get_background_color = [255, 255, 255, 180],
                        get_text_anchor=pdk.types.String("middle"),
                        get_alignment_baseline=pdk.types.String("center"),
                    )
                    self.pdkLayers.append(lyr)
                    self.pdkLyrNm.append(ly+'_NtwLbl')
                print(f'Map layers added {ly} as Edges')

            if ly in colsEntries:
                # if output attribute name in entriesdf columns
                dt = self.EntriesDf[['geometry', ly]].copy()
                dt = dt.to_crs(4326)

                dt['color'] = [(int(x[0]), int(x[1]), int(x[2]), 120) for x in ColorRampMapping(np.array(dt[ly]), **st)]
                dt['radius'] = ValueRampMapping(np.array(dt[ly]), **st)*2+2
                lyr = pdk.Layer(
                    type="GeoJsonLayer",
                    data=dt,
                    pickable=False,
                    get_fill_color='color',
                    get_line_width = 0,
                    get_radius='radius',
                )
                self.pdkLayers.append(lyr)
                self.pdkLyrNm.append(ly+'_Ent')
                if st['label']:
                    dt['coords'] = dt.geometry.apply(getcoords)
                    dt[ly] = dt[ly].apply(NumStringFormat)
                    lyr = pdk.Layer(
                        type="TextLayer",
                        data=dt,
                        pickable=False,
                        get_position='coords',
                        get_text=ly,
                        get_size=st['labelsize'],
                        get_color=[0,0,0],
                        background=True,
                        get_background_color = [255, 255, 255, 180],
                        get_text_anchor=pdk.types.String("middle"),
                        get_alignment_baseline=pdk.types.String("center"),
                    )
                    self.pdkLayers.append(lyr)
                    self.pdkLyrNm.append(ly+'_EntLbl')
                print(f'Mapp layers added {ly} as Entries')

    def Map_Show(self, show='base', map_style='light', height=500, width=500, viewZoom=17, viewCenter=None):
        
        if len(self.pdkLayers)==0:
            self.Map_BaseLayerInit()
        
        match show:
            case 'base':
                pass
            case 'junction':
                self.Map_LayerAdd({'junction':None,})
        print(f'Shown Layers:\n\t{self.pdkLyrNm}')
        
        if viewCenter is not None:
            try: self.pdkCenter = Point(viewCenter[0], viewCenter[1])
            except: raise("viewCenter not in acceptable format, in list or tuple of x,y")
        view_state = pdk.ViewState(latitude=self.pdkCenter.y, longitude=self.pdkCenter.x, zoom=viewZoom)
        
        pdkmap = pdk.Deck(layers=self.pdkLayers, initial_view_state=view_state, map_style=map_style, height=height, width=width)
        return pdkmap
           
    def BetweenessPatronage(self, OriID=None, DestID=None, **kwargs):
        """
        betweenesspatronage(OriID=list, DestID=list, **kwargs)\n
        betweeness patronage metric\n
        returns edited self.NetworkDf\n
        if DestID is None, will search through all available destinations

        """
        Settings={
            'OriWgt': 'weight',
            'DestWgt' : 'weight',
            'RsltAttr': 'PatronBtwns',
            'AttrEdgeID': self.baseSet['EdgeID'],
            'AttrEntID': self.baseSet['EntID'],
            'SearchDist' : 1200.0, 
            'DetourR' : 1.0, 
            'AlphaExp' : 0.0,
            'DistMul' : 2.0,
            'EdgeCmin' : 0.9,
            'PathLim' : 2000,
            'LimCycles' : 1_000_000,
            'OpType' : 'P',
            'Include_Destination' : False,
            'Threads' : None
        }
        if kwargs:
            for k,v in kwargs.items():
                Settings[k] = v
        if Settings['Threads'] is not None:
            self.Threads(Settings['Threads'])
        print(f'BetweenessPatronage ---------- \nAs {Settings["RsltAttr"]} from {Settings["OriWgt"]} to {Settings["DestWgt"]}')
        # processing betweeness patronage of a network.
        # collect all relatable origins and destinations

        if Settings['OriWgt'] not in self.EntriesDf.columns:
            self.EntriesDf['OriWgt'] = (1,)*len(self.EntriesDf)
            print(f'field {Settings["OriWgt"]} is not detected, appended with default 1.0 value')
        if Settings['DestWgt'] not in self.EntriesDf.columns:
            self.EntriesDf['DestWgt'] = (1,)*len(self.EntriesDf)
            print(f'field {Settings["OriWgt"]} is not detected, appended with default 1.0 value')

        OriDf = self.EntriesDf[(self.EntriesDf[Settings['OriWgt']]>0)][[self.baseSet['EntID'], Settings['OriWgt'], 'geometry']] # filtering only those above 0
        DestDf = self.EntriesDf[(self.EntriesDf[Settings['DestWgt']]>0)][[self.baseSet['EntID'], Settings['DestWgt'], 'geometry']]   
        try:
            if OriID is not None: # if there are specific OriID
                OriDf = OriDf[(OriDf[self.baseSet['EntID']].isin(OriID))]
            
            if DestID is not None: # if there are specific destID
                DestDf = DestDf[(DestDf[self.baseSet['EntID']].isin(DestID))]
            print(f'Collected {len(OriDf)} Origin and {len(DestDf)} Destinations Point[s]')
        except:
            raise Exception("OriID or DestID not in correct form. Any form of iterable array that works in panda DataFrame.isin() is possible")

        # Base_BetweenessPatronage(Gdf, Gph, EntriesPt, OriDf, DestDf, SettingDict)
        if self.baseSet['Threads'] == 1:
            tmSt = time()
            if Settings['OpType'] == 'P':
                print('Processing with singlethreading & Plural mapping')
                Rslt = Base_BetweenessPatronage_Plural(self.Gph, OriDf, DestDf, Settings)
            else:
                print('Processing with singlethreading & Singular mapping')
                Rslt = Base_BetweenessPatronage_Singular(self.Gph, OriDf, DestDf, Settings)
            print(f'Processing finished in {time()-tmSt:,.3f} seconds')
            # self.NetworkDf[Settings['RsltAttr']] = (0,)*len(Rslt[1])
            # for i, v in zip(Rslt[0], Rslt[1]):
            # self.NetworkDf[Settings['RsltAttr']] = Rslt
            if Settings['Include_Destination']:
               Entryids = np.array(DestDf.index)
               RsltDes = np.zeros(len(self.EntriesDf), dtype=np.float32)
               RsltDes[Entryids] = Rslt[1]
               self.EntriesDf[Settings['RsltAttr']] = RsltDes
               self.NetworkDf[Settings['RsltAttr']] = Rslt[0]
            else:
                self.NetworkDf[Settings['RsltAttr']] = Rslt
        else:
            chunksize = int(round(len(OriDf) / self.baseSet['Threads'], 0)) + 1
            if len(OriDf) > 100:
                chunksize = int(round(chunksize / 2,0))
            largs = [(self.Gph, OriDf[i:i+chunksize], DestDf, Settings) for i in range(0, len(OriDf), chunksize)]
            tmSt = time()
            if Settings['OpType'] == 'P':
                print(f'Processing with multithreading & Plural mapping, with chunksize {chunksize}')
                SubRslt = MultiProcessPool(gph_Base_BetweenessPatronage_Plural_multi, largs)
            else:
                print(f'Processing with multithreading & Singular mapping, with chunksize {chunksize}')
                SubRslt = MultiProcessPool(gph_Base_BetweenessPatronage_Singular_multi, largs)
            
            print(f'Multiprocessing finished in {time()-tmSt:,.3f} seconds')

            Rslt = np.zeros(len(self.NetworkDf), dtype=np.float32)
            if Settings['Include_Destination']:
                Entryids = np.array(DestDf.index)
                RsltDes = np.zeros(len(self.EntriesDf), dtype=np.float32)
                for rslt in SubRslt:
                    Rslt += rslt[0]
                    RsltDes[Entryids] += rslt[1]
                self.EntriesDf[Settings['RsltAttr']] = RsltDes
            else:
                for rslt in SubRslt:
                    Rslt += rslt
            self.NetworkDf[Settings['RsltAttr']] = Rslt
        return self.NetworkDf

    def Reach(self, OriID:list=None, DestID:list=None, Mode:str='N', **kwargs):
        """
        Reach(OriID:list, DestID:list, **kwargs)\n
        Calculating reach, which has multiple modes, as in:\n
        - Reach \'N\'  : number of reachable features on distance,\n
        - Reach \'W\'  : sum of weight on reachable features on distance\n
        - Reach \'WD\' : sum of weight with inverse distance (linear/exponent) with compounding multiplier weights on reachable features on distance\n
        returns returns self.EntriesDf
        """
        Settings={
            'AttrEntID': self.baseSet['EntID'],
            'SearchDist': 1200.0,
            'DestWgt': 'weight',
            'CalcExp': 0.35,
            'CalcComp': 0.6,
            'RsltAttr': 'Reach',
            'LimCycles' : 1_000_000,
            'Threads':None,
        }
        if kwargs:
            for k,v in kwargs.items():
                Settings[k] = v
        if Settings['Threads'] is not None:
            self.Threads(Settings['Threads'])
        print(f'Reach -------------- As {Settings["RsltAttr"]} with {Mode} of {Settings["DestWgt"]}')
        # processing reach valuation of a network
        # collecting all relatable origins and destinations
        OriDf = self.EntriesDf[[self.baseSet['EntID'], 'geometry']]
        DestDf = self.EntriesDf[[self.baseSet['EntID'], 'geometry']]
        if Mode != 'N':
            DestDf[Settings['DestWgt']] = self.EntriesDf[Settings['DestWgt']]

        RsltAttr = Settings['RsltAttr']
        try:
            if OriID is not None: # if there are specific OriID
                OriDf = OriDf[(OriDf[self.baseSet['EntID']].isin(OriID))]

            if DestID is not None: # if there are specific destID
                DestDf = DestDf[(DestDf[self.baseSet['EntID']].isin(DestID))]
            print(f'Collected {len(OriDf)} Origin and {len(DestDf)} Destinations Point[s]')
        except:
            raise Exception("OriID or DestID not in correct form. Any form of iterable array that works in panda DataFrame.isin() is possible")
        
        threads = self.baseSet['Threads']
        if len(OriDf) < 50:
            threads = 1
        if threads == 1: # if single thread
            tmSt = time()
            print('Processing with singlethreading')
            inpt = (Mode, self.Gph, OriDf, DestDf, Settings)
            Rslt = gph_Base_Reach_multi(inpt)
            print(f'Processing finished in {time()-tmSt:,.3f} seconds')

            if RsltAttr not in self.EntriesDf.columns:
                self.EntriesDf[RsltAttr] = (0,)*self.EntriesDf.shape[0]

            if Mode == 'N':
                for i, v in zip(Rslt[0], Rslt[1]):
                    self.EntriesDf.at[i, RsltAttr] = v
            elif Mode == 'NDW':
                if f'{RsltAttr}_W' not in self.EntriesDf.columns:
                    self.EntriesDf[f'{RsltAttr}_D'] = (0,)*self.EntriesDf.shape[0]
                    self.EntriesDf[f'{RsltAttr}_W'] = (0,)*self.EntriesDf.shape[0]
                for i, v, w, x in zip(Rslt[0], Rslt[1], Rslt[2], Rslt[3]):
                    self.EntriesDf.at[i, RsltAttr] = v
                    self.EntriesDf.at[i, f'{RsltAttr}_D'] = w
                    self.EntriesDf.at[i, f'{RsltAttr}_W'] = x
            else:
                if f'{RsltAttr}_W' not in self.EntriesDf.columns:
                    self.EntriesDf[f'{RsltAttr}_W'] = (0,)*self.EntriesDf.shape[0]
                for i, v, w in zip(Rslt[0], Rslt[1], Rslt[2]):
                    self.EntriesDf.at[i, RsltAttr] = v
                    self.EntriesDf.at[i, f'{RsltAttr}_W'] = w

        else:
            chunksize = int(round(len(OriDf) / self.baseSet['Threads'], 0)) + 1
            print(f'Processing with multithreading, with chunksize {chunksize}')
            tmSt = time()
            if len(OriDf) > 400:
                chunksize = int(round(chunksize / 2,0))
            largs = [(Mode, self.Gph, OriDf[i:i+chunksize], DestDf, Settings) for i in range(0, len(OriDf), chunksize)]
            SubRslt = MultiProcessPool(gph_Base_Reach_multi, largs)
            print(f'Multiprocessing finished in {time()-tmSt:,.3f} seconds')

            if RsltAttr not in self.EntriesDf.columns:
                self.EntriesDf[RsltAttr] = (0,)*self.EntriesDf.shape[0]
                
            if Mode == 'N':
                for rslt in SubRslt:
                    for i, v in zip(rslt[0], rslt[1]):
                        self.EntriesDf.at[i, RsltAttr] = v
            elif Mode == 'NDW':
                if f'{RsltAttr}_W' not in self.EntriesDf.columns:
                    self.EntriesDf[f'{RsltAttr}_D'] = (0,)*self.EntriesDf.shape[0]
                    self.EntriesDf[f'{RsltAttr}_W'] = (0,)*self.EntriesDf.shape[0]
                for rslt in SubRslt:
                    for i, v, w, x in zip(rslt[0], rslt[1], rslt[2], rslt[3]):
                        self.EntriesDf.at[i, RsltAttr] = v
                        self.EntriesDf.at[i, f'{RsltAttr}_D'] = w
                        self.EntriesDf.at[i, f'{RsltAttr}_W'] = x
            else:
                if f'{RsltAttr}_W' not in self.EntriesDf.columns:
                    self.EntriesDf[f'{RsltAttr}_W'] = (0,)*self.EntriesDf.shape[0]
                for rslt in SubRslt:
                    for i, v, w in zip(rslt[0], rslt[1], rslt[2]):
                        self.EntriesDf.at[i, RsltAttr] = v
                        self.EntriesDf.at[i, f'{RsltAttr}_W'] = w
        return self.EntriesDf

    def Straightness(self, OriID:list=None, DestID:list=None, **kwargs):
        """
        Straightness(OriID:list, DestID:list, **kwargs)\n
        Calculating straightness average, can be distance weighted or inverse distance weighted\n
        returns self.EntriesDf
        """
        Settings={
            'AttrEntID': self.baseSet['EntID'],
            'SearchDist': 1500.0,
            'CalcExp': 0.35,
            'DestWgt': 'weight',
            'RsltAttr': 'Straightness',
            'LimCycles' : 1_000_000,
            'Threads':None,
        }
        if kwargs:
            for k,v in kwargs.items():
                Settings[k] = v
        if Settings['Threads'] is not None:
            self.Threads(Settings['Threads'])
        print(f'Straightness Average -------------\n As {Settings["RsltAttr"]} of {Settings["DestWgt"]}')
        # processing reach valuation of a network
        # collecting all relatable origins and destinations
        OriDf = self.EntriesDf[[self.baseSet['EntID'], 'geometry']]
        DestDf = self.EntriesDf[[self.baseSet['EntID'], Settings['DestWgt'], 'geometry']]

        RsltAttr = Settings['RsltAttr']

        try:
            if OriID is not None: # if there are specific OriID
                OriDf = OriDf[(OriDf[self.baseSet['EntID']].isin(OriID))]

            if DestID is not None: # if there are specific destID
                DestDf = DestDf[(DestDf[self.baseSet['EntID']].isin(DestID))]
            print(f'Collected {len(OriDf)} Origin and {len(DestDf)} Destinations Point[s]')
        except:
            raise Exception("OriID or DestID not in correct form. Any form of iterable array that works in panda DataFrame.isin() is possible")

        DestDf[Settings['DestWgt']] = DestDf[Settings['DestWgt']].astype('float32')
        
        if self.baseSet['Threads'] == 1: # if single thread
            tmSt = time()
            print('Processing with singlethreading')
            Rslt = gph_Base_Straightness_multi((self.Gph, OriDf, DestDf, Settings))
            print(f'Processing finished in {time()-tmSt:,.3f} seconds')

            if RsltAttr not in self.EntriesDf.columns:
                self.EntriesDf[RsltAttr] = (0,)*self.EntriesDf.shape[0]

            for i, v in zip(Rslt[0], Rslt[1]):
                self.EntriesDf.at[i, RsltAttr] = v

        else:
            chunksize = int(round(len(OriDf) / self.baseSet['Threads'], 0)) + 1
            print(f'Processing with multithreading, with chunksize {chunksize}')
            tmSt = time()
            if len(OriDf) > 100:
                chunksize = int(round(chunksize / 4 , 0))
            largs = [(self.Gph, OriDf[i:i+chunksize], DestDf, Settings) for i in range(0, len(OriDf), chunksize)]
            SubRslt = MultiProcessPool(gph_Base_Straightness_multi, largs, self.baseSet['Threads'])
            print(f'Multiprocessing finished in {time()-tmSt:,.3f} seconds')

            if RsltAttr not in self.EntriesDf.columns:
                self.EntriesDf[RsltAttr] = (0,)*self.EntriesDf.shape[0]

            for rslt in SubRslt:
                for i, v in zip(rslt[0], rslt[1]):
                    self.EntriesDf.at[i, RsltAttr] = v
        return self.EntriesDf

    def PathReach(self, OriID:list, distance:float=800, joined:bool=False, incl_nodes=False, showmap=False, skip_layerinit=True, pdkupdate=False, **kwargs):
        """
        PathReach(OriID:list, Distance:float=800, **kwargs)\n
        Returning dataframe of edges\n
        returns geoDataFrame of split lines
        """

        def line_subpart(geom, vector):
            if abs(vector) == 1.0:
                return geom
            if vector < 0.0:
                vec = 1 + vector
            else:
                vec = vector
            dist = vec * geom.length
            coords = list(geom.coords)

            if len(coords) == 2:
                cp = geom.interpolate(dist)
                if vector < 0.0:
                    return LineString([(cp.x, cp.y)] + [coords[1],])
                else:
                    return LineString([coords[0],] + [(cp.x, cp.y)])
                
            for i, c in enumerate(coords):
                pd = geom.project(Point(c))
                if pd > dist:
                    cp = geom.interpolate(dist)
                    if vector < 0.0:
                        return LineString([(cp.x, cp.y)] + coords[i:])
                    else:
                        return LineString(coords[:i] + [(cp.x, cp.y)])
        if showmap and not incl_nodes:
            incl_nodes = True
            print('if showmap, incl_nodes must be true')

        if incl_nodes:
            self.getNodes()

        if joined:
            rslt = self.Gph.PathReachMulti_VirtuEntry(tuple(OriID), distance, incl_nodes)
            if incl_nodes:
                rsltNd = rslt[1]
                rslt = rslt[0]
                nid = [x[0] for x in rsltNd]
                nodedf = self.NodeDf.loc[nid].copy()
                nodedf['Distance'] = [x[1] for x in rsltNd]
            eid = [x[0] for x in rslt]
            edf = self.NetworkDf.loc[eid][['geometry', self.baseSet['EdgeID']]]
            edf['vector'] = [x[1] for x in rslt]
            edf['geometry'] = edf.apply(lambda x: line_subpart(x.geometry, x.vector), axis=1)
        else:
            edf = None
            nodedf = None
            for oid in OriID:
                rslt = self.Gph.PathReachMulti_VirtuEntry((oid,), distance, incl_nodes)
                if incl_nodes:
                    rsltNd = rslt[1]
                    rslt = rslt[0]
                    nid = [x[0] for x in rsltNd]
                    nodedfs = self.NodeDf.loc[nid].copy()
                    nodedfs['Distance'] = [x[1] for x in rsltNd]
                    if nodedf is None:
                        nodedf = nodedfs.copy()
                    else:
                        nodedf = gpd.GeoDataFrame(pd.concat([nodedf, nodedfs], ignore_index=True))
                eid = [x[0] for x in rslt]
                edfs = self.NetworkDf.loc[eid][['geometry', self.baseSet['EdgeID']]]
                edfs['vector'] = [x[1] for x in rslt]
                edfs['geometry'] = edfs.apply(lambda x: line_subpart(x.geometry, x.vector), axis=1)
                edfs['Oid'] = oid
                if edf is None:
                    edf = edfs.copy()
                else:
                    edf = gpd.GeoDataFrame(pd.concat([edf, edfs], ignore_index=True))
        
        edf.set_crs(self.epsg, inplace=True, allow_override=True)
        if incl_nodes:
            nodedf.set_crs(self.epsg, inplace=True, allow_override=True)
        
        if showmap:
            if len(self.pdkLyrNm) == 0 and not skip_layerinit:
                print('setting base init layer')
                self.Map_BaseLayerInit()
            MapLayers = self.pdkLayers.copy()
            edt = edf.copy().to_crs(4326)
            lyr = pdk.Layer(
                    type="GeoJsonLayer",
                    data=edt,
                    pickable=True,
                    get_line_color=[255,80,80,180],
                    get_line_width=3,
                )
            MapLayers.append(lyr)
            ndt = nodedf.copy().to_crs(4326)
            lyr = pdk.Layer(
                    type="GeoJsonLayer",
                    data=ndt,
                    pickable=True,
                    get_fill_color=[255,80,80],
                    get_line_width = 0,
                    get_radius=2,
                )
            MapLayers.append(lyr)
            ndt['coords'] = ndt.geometry.apply(getcoords)
            ndt['Distance'] = ndt.Distance.apply(lambda x: f'{x:,.1f}')
            lyr = pdk.Layer(
                    type="TextLayer",
                    data=ndt,
                    pickable=False,
                    get_position='coords',
                    get_text='Distance',
                    get_size=8,
                    get_color=[0,0,0],
                    background=True,
                    get_background_color = [255, 255, 255, 180],
                    get_text_anchor=pdk.types.String("middle"),
                    get_alignment_baseline=pdk.types.String("center"),
                )
            MapLayers.append(lyr)

            entO = self.EntriesDf.loc[OriID].copy().to_crs(4326)
            lyr = pdk.Layer(
                    type="GeoJsonLayer",
                    data=entO,
                    pickable=True,
                    get_fill_color=[255,80,80],
                    get_line_width = 0,
                    get_radius=4,
                )
            MapLayers.append(lyr)
            if pdkupdate:
                return MapLayers
            if self.pdkCenter is None:
                self.pdkCenter = entO.geometry.unary_union.centroid
            view_state = pdk.ViewState(latitude=self.pdkCenter.y, longitude=self.pdkCenter.x, zoom=15)
            pdkmap = pdk.Deck(layers=MapLayers, initial_view_state=view_state, map_style='light')
            print('to see map, please set/access index 0 of result')
            return pdkmap, edf, nodedf

        if incl_nodes:
            return edf, nodedf
        return edf

