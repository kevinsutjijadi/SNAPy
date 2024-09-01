### SNAPy (Spatial Network Analysis Python)
# main script, contains the compiled processings
# Kevin Sutjijadi @ 2023

__author__ = "Kevin Sutjijadi"
__copyright__ = "Copyright 2023, Kevin Sutjijadi"
__credits__ = ["Kevin Sutjijadi"]
__version__ = "0.2.0"

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

# importing internal scripts
# from .prcs_geom import *
from .prcs_grph import *
from .prcs_sna import *
from .utils import MultiProcessPool
from .SGACy.graph import GraphCy
from .SGACy.geom import *


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
            'SizeBuffer': 1.0,
            'Verbose': True,
            'EntryDtDump': True,
            'EntryDtDumpOvr': False,
            'Threads': 0,
            'DumpDir': '\\dump',
            'DumpEnt' : 'EntryDtDump.pkl',
        }

        for k,v in kwargs.items():
            self.baseSet[k] = v
        if self.baseSet['Threads'] == 0:
            self.baseSet['Threads'] = os.cpu_count()-1

        if EntriesDf.crs.is_geographic:
            raise Exception("EntriesDf projection is geographic (degrees), please conver to projected")
        if NetworkDf.crs.is_geographic:
            raise Exception("NetworkDf projection is geographic (degrees), please conver to projected")
        if EntriesDf.crs.to_epsg() != NetworkDf.crs.to_epsg():
            raise Exception("EntriesDf and NetworkDf have different projections, please match them using gdb.to_crs()")

        if self.baseSet['EntID'] not in EntriesDf:
            entid = self.baseSet['EntID']
            print(f'Entries ID: {entid} not found, remaking from index')
            EntriesDf[entid] = EntriesDf.index
        if self.baseSet['EdgeID'] not in NetworkDf:
            edgid = self.baseSet['EdgeID']
            print(f'Network ID: {edgid} not found, remaking from index')
            NetworkDf[edgid] = NetworkDf.index

        self.dumpdir = self.baseSet['DumpDir']
        self.EntPtDumpDir = self.baseSet['DumpEnt']
        self.EntriesDf = EntriesDf
        self.LastAtt = None
        
        
        print(f'GraphSim Class ----------')

        if NetworkDf.geometry[0].geom_type == 'MultiLineString':
            NetworkDf = NetworkDf.explode()
            print('Network data is multilinestring, exploded')
            NetworkDf.index = range(int(NetworkDf.shape[0]))

        if self.baseSet['EntID'] not in self.EntriesDf.columns:
            print('EntriesDf EntID not detected, adding from index')
            self.EntriesDf[self.baseSet['EntID']] = range(self.EntriesDf.shape[0])

        if self.baseSet['EdgeID'] not in NetworkDf.columns:
            print('NetworkDf EdgeID not detected, adding from index')
            NetworkDf[self.baseSet['EdgeID']] = range(int(NetworkDf.shape[0]))
        
        self.Gph = GraphCy(int(NetworkDf.size*self.baseSet['SizeBuffer']+2), int(NetworkDf.size*self.baseSet['SizeBuffer']+4), len(EntriesDf))
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

        if 'bbox' not in NetworkDf.columns:
            NetworkDf['bbox'] = NetworkDf.apply(lambda x: bbox(x.geometry), axis=1)
        

        if self.baseSet['EntryDtDump']:# if dump
            if not os.path.exists(self.dumpdir):
                os.mkdir(self.dumpdir)
            if os.path.exists(f'{self.dumpdir}\\{self.EntPtDumpDir}') and not self.baseSet['EntryDtDumpOvr']:
                print('Pickled EntriesPt File Detected, using it instead')
                with open(f'{self.dumpdir}\\{self.EntPtDumpDir}', 'rb') as op:
                    EntriesDt = pickle.load(op)
                print(f'Found {len(EntriesDt)} Pickled Entry Points at {self.dumpdir}')
            else:    
                if self.baseSet['Threads'] == 1:
                    EntriesDt = graph_addentries(NetworkDf, EntriesDf, self.baseSet['EntDist'], self.baseSet['EntID'], self.baseSet['EdgeID'], self.baseSet['AN_EdgeCost'] )
                else:
                    chunksize = int(round(len(self.EntriesDf) / self.baseSet['Threads'], 0)) + 1
                    largs = tuple((NetworkDf, self.EntriesDf[i:i+chunksize], self.baseSet['EntDist'], self.baseSet['EntID'], self.baseSet['EdgeID'], self.baseSet['AN_EdgeCost']) for i in range(0, len(self.EntriesDf), chunksize))
                    EntPt = MultiProcessPool(gph_addentries_multi, largs)
                    EntriesPt = []
                    for ent in EntPt:
                        EntriesPt += list(ent)
                    EntriesDt = tuple(EntriesPt)
                print(f'Pickling {len(EntriesDt)} Entry Points')
                with open(f'{self.dumpdir}\\{self.EntPtDumpDir}', 'wb') as op:
                    pickle.dump(EntriesDt, op)
                print('Pickling EntriesPt Successfull')
        else:
            if self.baseSet['Threads'] == 1:
                EntriesDt = graph_addentries(self.NetworkDf, EntriesDf, self.baseSet['EntDist'], self.baseSet['EntID'], self.baseSet['EdgeCost'])
                
            else:
                chunksize = int(round(len(self.EntriesDf) / self.baseSet['Threads'], 0)) + 1
                largs = [(NetworkDf, self.EntriesDf[i:i+chunksize], self.baseSet['EntDist'], self.baseSet['EntID'], self.baseSet['EdgeID'], self.baseSet['EdgeCost']) for i in range(0, len(self.EntriesDf), chunksize)]
                EntPt = MultiProcessPool(gph_addentries_multi, largs)
                EntriesPt = []
                for ent in EntPt:
                    EntriesPt += list(ent)
                EntriesDt = tuple(EntriesPt)
        EntriesDt = tuple(((d[0], d[1], d[2], tuple(d[3]), d[4], d[5]) for d in EntriesDt)) # i have no idea why edist is converted to list
        self.Gph.addEntries(tuple(EntriesDt))
        self.EntriesDf['xLn_ID'] = [dt[1] for dt in EntriesDt]
        self.EntriesDf['xPt_X'] = [dt[4][0] for dt in EntriesDt]
        self.EntriesDf['xPt_Y'] = [dt[4][1] for dt in EntriesDt]
        try: self.EntriesDf['xPt_Z'] = [dt[4][2] for dt in EntriesDt]
        except: self.EntriesDf['xPt_Z'] = [0.0 for dt in EntriesDt]

        # map!
        self.pdkLayers = []
        self.pdkCenter = None


    def __repr__(self) -> str:
        strNwSim = f'GraphSim object of {len(self.NetworkDf)} Segments and {len(self.EntriesDf)} Entries'
        return strNwSim

    def Map_BaseLayerInit(self, ShowLayer=None):
        """
        Map_LayerBuild
        builds pydeck map base layers consisting of network and entries
        """
        NetDf = self.NetworkDf[['geometry', self.baseSet['EdgeID']]].copy()
        NetDf = NetDf.to_crs(4326)
        lyrNetwork = pdk.Layer(
            type="GeoJsonLayer",
            data=NetDf,
            pickable=True,
            get_line_color=[80,80,80],
            get_line_width=0.5,
        )

        EntXDf = self.EntriesDf.copy()
        EntXDf['geometry'] = EntXDf.apply(lambda x: LineString(((x.geometry.x, x.geometry.y), (x.xPt_X, x.xPt_Y))), axis=1)
        EntXDf = EntXDf.to_crs(4326)
        lyrEntriesX = pdk.Layer(
            type="GeoJsonLayer",
            data=EntXDf,
            pickable=False,
            get_line_color=[255,80,80],
            get_line_width=1,
        )

        EntDf = self.EntriesDf.copy()
        EntDf = EntDf.to_crs(4326)
        lyrEntries = pdk.Layer(
            type="GeoJsonLayer",
            data=EntDf,
            pickable=True,
            get_fill_color=[255,80,80],
            get_line_width = 0,
            get_radius=3
        )
        self.pdkCenter = NetDf.geometry.unary_union.centroid
        print('built base layers on self.pdfLayers. indexed at: 0-base network layer, 1-entries line connection to edges, 2-entries')
        print('able to pop, edit ,or add more layers by directly accessing self.pdfLayers before using self.ShowMap\n')

        self.pdkLayers = [lyrNetwork, lyrEntriesX, lyrEntries]
        
    def Map_Show(self, tooltip=None, map_style='light', height=500, width=500):
        
        if len(self.pdkLayers)==0:
            self.Map_BaseLayerInit()

        view_state = pdk.ViewState(latitude=self.pdkCenter.y, longitude=self.pdkCenter.x, zoom=16)
        if tooltip is None:
            tooltip={
                "html": "<b>id:</b>{"+self.baseSet['EdgeID']+"}",
                "style": {
                        "backgroundColor": "white",
                        "color": "red",
                        "fontFamily": "Helvetica",

                }
            }
        pdkmap = pdk.Deck(layers=self.pdkLayers, initial_view_state=view_state, tooltip=tooltip, map_style=map_style, height=height, width=width)
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
            'PathLim' : 200,
            'LimCycles' : 1_000_000,
            'OpType' : 'P'
        }
        if kwargs:
            for k,v in kwargs.items():
                Settings[k] = v
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

        if OriID is not None: # if there are specific OriID
            OriDf = OriDf[(OriDf[self.baseSet['EntID']].isin(OriID))]
        
        if DestID is not None: # if there are specific destID
            DestDf = DestDf[(DestDf[self.baseSet['EntID']].isin(DestID))]
        print(f'Collected {len(OriDf)} Origin and {len(DestDf)} Destinations Point[s]')

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
            # self.NetworkDf[Settings['RsltAttr']] = (0,)*len(self.NetworkDf)
            Rslt = np.zeros(len(self.NetworkDf), dtype=float)
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
        }
        if kwargs:
            for k,v in kwargs.items():
                Settings[k] = v
        
        print(f'Reach -------------- As {Settings["RsltAttr"]} with {Mode} of {Settings["DestWgt"]}')
        # processing reach valuation of a network
        # collecting all relatable origins and destinations
        OriDf = self.EntriesDf[[self.baseSet['EntID'], 'geometry']]
        DestDf = self.EntriesDf[[self.baseSet['EntID'], 'geometry']]
        if Mode != 'N':
            DestDf[Settings['DestWgt']] = self.EntriesDf[Settings['DestWgt']]

        RsltAttr = Settings['RsltAttr']

        if OriID is not None: # if there are specific OriID
            OriDf = OriDf[(OriDf[self.baseSet['EntID']].isin(OriID))]

        if DestID is not None: # if there are specific destID
            DestDf = DestDf[(DestDf[self.baseSet['EntID']].isin(DestID))]
        print(f'Collected {len(OriDf)} Origin and {len(DestDf)} Destinations Point[s]')
        
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
        }
        if kwargs:
            for k,v in kwargs.items():
                Settings[k] = v
        
        print(f'Straightness Average -------------\n As {Settings["RsltAttr"]} of {Settings["DestWgt"]}')
        # processing reach valuation of a network
        # collecting all relatable origins and destinations
        OriDf = self.EntriesDf[[self.baseSet['EntID'], 'geometry']]
        DestDf = self.EntriesDf[[self.baseSet['EntID'], Settings['DestWgt'], 'geometry']]

        RsltAttr = Settings['RsltAttr']

        if OriID is not None: # if there are specific OriID
            OriDf = OriDf[(OriDf[self.baseSet['EntID']].isin(OriID))]

        if DestID is not None: # if there are specific destID
            DestDf = DestDf[(DestDf[self.baseSet['EntID']].isin(DestID))]
        print(f'Collected {len(OriDf)} Origin and {len(DestDf)} Destinations Point[s]')

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
