### SNAPy (Spatial Network Analysis Python)
# main script, contains the compiled processings
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
import networkx as nx
import pandas as pd

# importing internal scripts
from .prcs_geom import *
from .prcs_grph import *
from .prcs_sna import *
from .utils import MultiProcessPool



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
            'EntID': 'FID',
            'EdgeID': 'FID',
            'Verbose': True,
            'EntryDtDump': True,
            'EntryDtDumpOvr': False,
            'Threads': 0,
            'DumpDir': '\\dump',
            'DumpFl': 'EntryDtDump'
        }
        for k,v in kwargs.items():
            self.baseSet[k] = v
        if self.baseSet['Threads'] == 0:
            self.baseSet['Threads'] = os.cpu_count()-1

        self.dumpdir = self.baseSet['DumpDir']
        self.EntPtDumpDir = f'{self.baseSet["DumpFl"]}.pkl'
        self.EntriesDf = EntriesDf

        print(f'GraphSim Class ----------')
        if self.baseSet['EntID'] not in self.EntriesDf.columns:
            print('EntriesDf EntID not detected, adding from index')
            self.EntriesDf[self.baseSet['EntID']] = self.EntriesDf.index

        if self.baseSet['EdgeID'] not in NetworkDf.columns:
            print('NetworkDf EdgeID not detected, adding from index')
            NetworkDf[self.baseSet['EdgeID']] = NetworkDf.index
        
        self.Gph, self.Pid, self.NetworkDf = BuildGraph(NetworkDf)
        print('Graph Built')

        genEntryDt = True
        genPickle = self.baseSet['EntryDtDump']

        if not os.path.exists(self.dumpdir):
            os.mkdir(self.dumpdir)

        if os.path.exists(f'{self.dumpdir}\\{self.EntPtDumpDir}') and not self.baseSet['EntryDtDumpOvr']:

            print('Pickled EntriesPt File Detected, using it instead')
            with open(f'{self.dumpdir}\\{self.EntPtDumpDir}', 'rb') as op:
                self.EntriesPt = pickle.load(op)
            print(f'Found {len(self.EntriesPt)} Pickled Entry Points at {self.dumpdir}')
            genPickle = False
            genEntryDt = False
        else:    
            genPickle = True
        
        if genEntryDt:
            if self.baseSet['Threads'] == 1:
                self.EntriesPt = graph_addentries(self.NetworkDf, EntriesDf, self.baseSet['EntDist'], self.baseSet['EntID'], self.baseSet['EdgeID'])
            else:
                chunksize = int(round(len(self.EntriesDf) / self.baseSet['Threads'], 0)) + 1
                largs = tuple((NetworkDf, self.EntriesDf[i:i+chunksize], self.baseSet['EntDist'], self.baseSet['EntID'], self.baseSet['EdgeID']) for i in range(0, len(self.EntriesDf), chunksize))
                EntPt = MultiProcessPool(gph_addentries_multi, largs)
                EntriesPt = []
                for ent in EntPt:
                    EntriesPt += list(ent)
                self.EntriesPt = tuple(EntriesPt)
        
        self.EntriesDf['xPt_X'] = tuple(dt[4].x for dt in self.EntriesPt)
        self.EntriesDf['xPt_Y'] = tuple(dt[4].y for dt in self.EntriesPt)

        print('Entries to Network Rebuilt')
        if genPickle:
            print(f'Pickling {len(self.EntriesPt)} Entry Points')
            with open(f'{self.dumpdir}\\{self.EntPtDumpDir}', 'wb') as op:
                pickle.dump(self.EntriesPt, op)
            print('Pickling EntriesPt Successfull')


    def reGenEntry(self, EntryDtDumpOvr=True):
        """
        rerun genEntryDt for reparametrizing entries point
        """
        if self.baseSet['Threads'] == 1:
            self.EntriesPt = graph_addentries(self.NetworkDf, self.EntriesDf, self.baseSet['EntDist'], self.baseSet['EntID'], self.baseSet['EdgeID'])
        else:
            chunksize = int(round(len(self.EntriesDf) / self.baseSet['Threads'], 0)) + 1
            largs = tuple((self.NetworkDf, self.EntriesDf[i:i+chunksize], self.baseSet['EntDist'], self.baseSet['EntID'], self.baseSet['EdgeID']) for i in range(0, len(self.EntriesDf), chunksize))
            EntPt = MultiProcessPool(gph_addentries_multi, largs)
            EntriesPt = []
            for ent in EntPt:
                EntriesPt += list(ent)
            self.EntriesPt = tuple(EntriesPt)

        if EntryDtDumpOvr:
            print(f'Pickling {len(self.EntriesPt)} Entry Points')
            with open(f'{self.dumpdir}\\{self.EntPtDumpDir}', 'wb') as op:
                pickle.dump(self.EntriesPt, op)
            print('Pickling EntriesPt Successfull')


    def reBuild(self, EntriesDf:gpd.GeoDataFrame=None, NetworkDf:gpd.GeoDataFrame=None, **kwargs):
        """
        reNetwork()\n
        updating network, with recalculating entries data
        """
        for k,v in kwargs.items():
            self.baseSet[k] = v
        if self.baseSet['Threads'] == 0:
            self.baseSet['Threads'] = os.cpu_count()-1
        
        self.dumpdir = self.baseSet['DumpDir']
        self.EntPtDumpDir = f'{self.baseSet["DumpFl"]}.pkl'

        print(f'GraphSim Rebuild --------------')
        if EntriesDf is not None:
            print('rebuilding EntriesDf')
            self.EntriesDf = EntriesDf
            if self.baseSet['EntID'] not in self.EntriesDf.columns:
                print('EntriesDf EntID not detected, adding from index')
                self.EntriesDf[self.baseSet['EntID']] = self.EntriesDf.index

        if NetworkDf is not None:
            print('rebuilding NetworkDf')
            if self.baseSet['EdgeID'] not in NetworkDf.columns:
                print('NetworkDf EdgeID not detected, adding from index')
                NetworkDf[self.baseSet['EdgeID']] = NetworkDf.index
            
            self.Gph, self.Pid, self.NetworkDf = BuildGraph(NetworkDf)
            print('Graph Rebuilt')
        
        if self.baseSet['Threads'] == 1:
            self.EntriesPt = graph_addentries(self.NetworkDf, EntriesDf, self.baseSet['EntDist'], self.baseSet['EntID'], self.baseSet['EdgeID'])
        else:
            chunksize = int(round(len(self.EntriesDf) / self.baseSet['Threads'], 0)) + 1
            largs = tuple((NetworkDf, self.EntriesDf[i:i+chunksize], self.baseSet['EntDist'], self.baseSet['EntID'], self.baseSet['EdgeID']) for i in range(0, len(self.EntriesDf), chunksize))
            EntPt = MultiProcessPool(gph_addentries_multi, largs)
            EntriesPt = []
            for ent in EntPt:
                EntriesPt += list(ent)
            self.EntriesPt = tuple(EntriesPt)

        self.EntriesDf['xPt_X'] = tuple(dt[4].x for dt in self.EntriesPt)
        self.EntriesDf['xPt_Y'] = tuple(dt[4].y for dt in self.EntriesPt)
        print('Entries to Network Rebuilt')

        if self.baseSet['EntryDtDump'] or self.baseSet['EntryDtDumpOvr']:
            print(f'Pickling {len(self.EntriesPt)} Entry Points')
            with open(f'{self.dumpdir}\\{self.EntPtDumpDir}', 'wb') as op:
                pickle.dump(self.EntriesPt, op)
            print('Pickling EntriesPt Successfull')


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
            'AlphaExp' : 0
        }
        if kwargs:
            for k,v in kwargs.items():
                Settings[k] = v
        print(f'BetweenessPatronage ---------- \nAs {Settings["RsltAttr"]} from {Settings["OriWgt"]} to {Settings["DestWgt"]}')
        # processing betweeness patronage of a network.
        # collect all relatable origins and destinations
        OriDf = self.EntriesDf[(self.EntriesDf[Settings['OriWgt']]>0)][[self.baseSet['EntID'], Settings['OriWgt'], 'geometry']] # filtering only those above 0
        DestDf = self.EntriesDf[(self.EntriesDf[Settings['DestWgt']]>0)][[self.baseSet['EntID'], Settings['DestWgt'], 'geometry']]   

        if OriID is not None: # if there are specific OriID
            OriDf = OriDf[(OriDf[self.baseSet['EntID']].isin(OriID))]
        print(f'Collected {len(OriDf)} Origin Point[s]')
        
        if DestID is not None: # if there are specific destID
            DestDf = DestDf[(DestDf[self.baseSet['EntID']].isin(DestID))]
        print(f'Collected {len(DestDf)} Destination Point[s]')

        # Base_BetweenessPatronage(Gdf, Gph, EntriesPt, OriDf, DestDf, SettingDict)
        if self.baseSet['Threads'] == 1:
            tmSt = time()
            print('Processing with singlethreading')
            Rslt = Base_BetweenessPatronage(self.NetworkDf, self.Gph, self.EntriesPt, OriDf, DestDf, Settings)
            print(f'Processing finished in {time()-tmSt:,.3f} seconds')
            self.NetworkDf[Settings['RsltAttr']] = (0,)*len(Rslt[1])
            for v, i in zip(Rslt[0], Rslt[1]):
                self.NetworkDf.at[i, Settings['RsltAttr']] = v

        else:
            chunksize = int(round(len(OriDf) / self.baseSet['Threads'], 0)) + 1
            if len(OriDf) > 400:
                chunksize = int(round(chunksize / 2,0))
            largs = [(self.NetworkDf, self.Gph, self.EntriesPt, OriDf[i:i+chunksize], DestDf, Settings) for i in range(0, len(OriDf), chunksize)]
            print(f'Processing with multithreading, with chunksize {chunksize}')
            tmSt = time()
            SubRslt = MultiProcessPool(gph_Base_BetweenessPatronage_multi, largs)
            print(f'Multiprocessing finished in {time()-tmSt:,.3f} seconds')
            self.NetworkDf[Settings['RsltAttr']] = (0,)*len(self.NetworkDf)
            for rslt in SubRslt:
                for i, v in zip(rslt[0], rslt[1]):
                    self.NetworkDf.at[i, Settings['RsltAttr']] += v
        return self.NetworkDf


    def Reach(self, OriID:list=None, DestID:list=None, Mode:str='N', **kwargs):
        """
        Reach(OriID:list, DestID:list, **kwargs)\n
        Calculating reach, which has multiple modes, as in:\n
        - Reach \'N\'  : number of reachable features on distance,\n
        - Reach \'W\'  : sum of weight on reachable features on distance\n
        - Reach \'WD\' : sum of weight with inverse distance (linear/exponent) with compounding multiplier weights on reachable features on distance\n
        returns tuple of FID and results of entry points
        """
        Settings={
            'AttrEntID': self.baseSet['EntID'],
            'SearchDist': 1500.0,
            'DestWgt': 'weight',
            'CalcType': 'Linear',
            'CalcExp': 0.35,
            'CalcComp': 0.6,
            'RsltAttr': 'Reach',
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
        print(f'Collected {len(OriDf)} Origin Point[s]')

        if DestID is not None: # if there are specific destID
            DestDf = DestDf[(DestDf[self.baseSet['EntID']].isin(DestID))]
        print(f'Collected {len(DestDf)} Destinations Point[s]')
        
        if self.baseSet['Threads'] == 1: # if single thread
            tmSt = time()
            print('Processing with singlethreading')
            inpt = (Mode, self.Gph, self.EntriesPt, OriDf, DestDf, Settings)
            Rslt = gph_Base_Reach_multi(inpt)
            print(f'Processing finished in {time()-tmSt:,.3f} seconds')
            self.EntriesDf[RsltAttr] = (0,)*self.EntriesDf.shape[0]
            if Mode == 'N':
                for i, v in zip(Rslt[0], Rslt[1]):
                    self.EntriesDf.at[i, RsltAttr] = v
            else:
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
            largs = [(Mode, self.Gph, self.EntriesPt, OriDf[i:i+chunksize], DestDf, Settings) for i in range(0, len(OriDf), chunksize)]
            SubRslt = MultiProcessPool(gph_Base_Reach_multi, largs)
            print(f'Multiprocessing finished in {time()-tmSt:,.3f} seconds')
            self.EntriesDf[RsltAttr] = (0,)*self.EntriesDf.shape[0]
            if Mode == 'N':
                for rslt in SubRslt:
                    for i, v in zip(rslt[0], rslt[1]):
                        self.EntriesDf.at[i, RsltAttr] = v
            else:
                self.EntriesDf[f'{RsltAttr}_W'] = (0,)*self.EntriesDf.shape[0]
                for rslt in SubRslt:
                    for i, v, w in zip(rslt[0], rslt[1], rslt[2]):
                        self.EntriesDf.at[i, RsltAttr] = v
                        self.EntriesDf.at[i, f'{RsltAttr}_W'] = w
        return self.EntriesDf


    def Straightness(self, OriID:list=None, DestID:list=None, Mode='A', **kwargs):
        """
        Straightness(OriID:list, DestID:list, **kwargs)\n
        Calculating straightness average\n
        returns tuple of FID and results of entry points
        """
        Settings={
            'AttrEntID': self.baseSet['EntID'],
            'SearchDist': 1500.0,
            'DestWgt': 'weight',
            'CalcType': 'Linear',
            'RsltAttr': 'Straightness',
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
        print(f'Collected {len(OriDf)} Origin Point[s]')

        if DestID is not None: # if there are specific destID
            DestDf = DestDf[(DestDf[self.baseSet['EntID']].isin(DestID))]
        print(f'Collected {len(DestDf)} Destinations Point[s]')
        
        if self.baseSet['Threads'] == 1: # if single thread
            tmSt = time()
            print('Processing with singlethreading')
            Rslt = gph_Base_Straightness_multi((Mode, self.Gph, self.EntriesPt, OriDf, DestDf, Settings))
            print(f'Processing finished in {time()-tmSt:,.3f} seconds')
            self.EntriesDf[RsltAttr] = (0,)*self.EntriesDf.shape[0]
            for i, v in zip(Rslt[0], Rslt[1]):
                self.EntriesDf.at[i, RsltAttr] = v

        else:
            chunksize = int(round(len(OriDf) / self.baseSet['Threads'], 0)) + 1
            print(f'Processing with multithreading, with chunksize {chunksize}')
            tmSt = time()
            if len(OriDf) > 400:
                chunksize = int(round(chunksize / 2 , 0))
            largs = [(Mode, self.Gph, self.EntriesPt, OriDf[i:i+chunksize], DestDf, Settings) for i in range(0, len(OriDf), chunksize)]
            SubRslt = MultiProcessPool(gph_Base_Straightness_multi, largs)
            print(f'Multiprocessing finished in {time()-tmSt:,.3f} seconds')
            self.EntriesDf[RsltAttr] = (0,)*self.EntriesDf.shape[0]
            for rslt in SubRslt:
                for i, v in zip(rslt[0], rslt[1]):
                    self.EntriesDf.at[i, RsltAttr] = v
        return self.EntriesDf


    def MapPaths(self, OriID:list=None, DestID:list= None, **kwargs):
        """
        MapPaths(OriID:list=None, DestID:list= None, **kwargs)\n
        Mapping paths for future processing, results can be pickled
        """
        Settings={
            'AttrEdgeID': self.baseSet['EdgeID'],
            'AttrEntID': self.baseSet['EntID'],
            'SearchDist' : 1200.0, 
            'DetourR' : 1.0, 
            'DumpSt' : True,
            'DumpOvr': False,
            'DumpFl' : 'PathMap.pkl'
        }
        if kwargs:
            for k,v in kwargs.items():
                Settings[k] = v
        # processing betweeness patronage of a network.
        # collect all relatable origins and destinations
        OriDf = self.EntriesDf[[self.baseSet['EntID'], 'geometry']]
        DestDf = self.EntriesDf[[self.baseSet['EntID'], 'geometry']]   
        DumpFl = Settings['DumpFl']

        if OriID is not None: # if there are specific OriID
            OriDf = OriDf[(OriDf[self.baseSet['EntID']].isin(OriID))]
        print(f'Collected {len(OriDf)} Origin Point[s]')
        
        if DestID is not None: # if there are specific destID
            DestDf = DestDf[(DestDf[self.baseSet['EntID']].isin(DestID))]
        print(f'Collected {len(DestDf)} Destination Point[s]')
        OptIDs = []
        OptPths = []
        OptDsts = []

        if os.path.exists(f'{self.dumpdir}\\{DumpFl}') and (Settings['DumpSt'] and not Settings['DumpOvr']):
            print('Pickled PathMap File Detected, using it instead')
            with open(f'{self.dumpdir}\\{self.EntPtDumpDir}', 'rb') as op:
                DumpData = pickle.load(op)
            print(f'Found {len(DumpData[0])} Pickled path maps')
            OptIDs = DumpData[0]
            OptPths = DumpData[1]
            OptDsts = DumpData[2]
        elif self.baseSet['Threads'] == 1:
            tmSt = time()
            print('Processing with singlethreading')
            OptIDs, OptPths, OptDsts = Base_MapPaths(self.Gph, self.EntriesPt, OriDf, DestDf, Settings)
            print(f'Processing finished in {time()-tmSt:,.3f} seconds')
        else:
            chunksize = int(round(len(OriDf) / self.baseSet['Threads'], 0)) + 1
            if len(OriDf) > 400:
                chunksize = int(round(chunksize / 2,0))
            largs = [(self.Gph, self.EntriesPt, OriDf[i:i+chunksize], DestDf, Settings) for i in range(0, len(OriDf), chunksize)]
            print(f'Processing with multithreading, with chunksize {chunksize}')
            tmSt = time()
            SubRslt = MultiProcessPool(gph_Base_MapPaths_multi, largs)
            print(f'Multiprocessing finished in {time()-tmSt:,.3f} seconds')
            for rslt in SubRslt:
                OptIDs += list(rslt[0])
                OptPths += list(rslt[1])
                OptDsts += list(rslt[2])
        
        self.MappedPaths = (OptIDs, OptPths, OptDsts)
        if Settings['DumpSt'] and Settings['DumpOvr']:
            print(f'Pickling {len(self.MappedPaths[0])} path maps')
            with open(f'{self.dumpdir}\\{DumpFl}', 'wb') as op:
                pickle.dump(self.MappedPaths, op)
            print('Pickling PathMap Successfull')
        return OptIDs, OptPths, OptDsts
