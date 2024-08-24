# distutils: language = c++

### SGACy (Spatial Graph Algorithm Cython)
# graph class script
# Kevin Sutjijadi @ 2024

cimport cython
from libcpp.queue cimport priority_queue
from libcpp.algorithm cimport sort, find
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.string cimport string
from libc.math cimport sqrt
import geopandas as gpd
from libc.stdlib cimport malloc, realloc, free
from typing import List
from libc.stdint cimport int32_t, uint32_t
from libc.string cimport memset

# Main graph class, functions somewhat similar to NetworkX's graph, but some changes/specifics are made to adjust for spatial-oriented functionality.

cdef struct Point3d:
    float x
    float y
    float z

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Point3d MakePoint3d(float& x, float& y, float z= 0.0) nogil:
    cdef Point3d pt
    pt.x = x
    pt.y = y
    pt.z = z
    return pt

cdef struct Line3d:
    int nPts
    Point3d* Pts

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int* Int_tuple_array(tuple input_tup):
    cdef int* arr
    cdef int i = <int>len(input_tup)
    cdef int n

    arr = <int*>malloc(i * sizeof(int))
    for n in range(i):
        arr[n] = input_tup[n]
    return arr

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline tuple Int_array_tuple(int* input_arr, int size):
    cdef int n
    return tuple((input_arr[n] for n in range(size)))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float* Float_tuple_array(tuple input_tup):
    cdef float* arr
    cdef int i = <int>len(input_tup)
    cdef int n

    arr = <float*>malloc(i * sizeof(float))
    for n in range(i):
        arr[n] = input_tup[n]
    return arr

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline tuple Float_array_tuple(float* input_arr, int size):
    cdef int n
    return tuple((input_arr[n] for n in range(size)))

cdef struct Node:
    int idx     # index number
    float[3] pt    # coordinate
    float w     # weight
    float c     # cost
    int[10] Eid    # connected edges IDs

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Node Node_make(tuple NodeDt):
    cdef Node nd
    nd.idx = NodeDt[0]
    nd.pt = Float_tuple_array(NodeDt[1])
    nd.w = NodeDt[2]
    nd.c = NodeDt[3]
    nd.Eid = Int_tuple_array(NodeDt[4])
    return nd


@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple Node_tuple(Node NodeDt):
    cdef tuple tup
    tup = (
        NodeDt.idx,
        Float_array_tuple(NodeDt.pt, 3),
        NodeDt.w,
        NodeDt.c,
        Int_array_tuple(NodeDt.Eid, 10)
    )
    return tup

cdef struct Edge:
    int idx        # index number
    int NidO       # origin Node ID
    int NidD       # destination Node ID
    float len      # base length
    float lenR     # reverse length (for bidirectional computing)
    float w        # weight

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Edge Edge_make(tuple EdgeDt):
    cdef Edge ed
    ed.idx = EdgeDt[0]
    ed.NidO = EdgeDt[1]
    ed.NidD = EdgeDt[2]
    ed.len = EdgeDt[3]
    ed.lenR = EdgeDt[4]
    ed.w = EdgeDt[5]
    return ed

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple Edge_tuple(Edge EdgeDt):
    cdef tuple tup
    tup = (
        EdgeDt.idx,
        EdgeDt.NidO,
        EdgeDt.NidD,
        EdgeDt.len,
        EdgeDt.lenR,
        EdgeDt.w,
    )
    return tup

cdef struct NodeReach:
    int Nid # node id
    float Dist # minimum distance found
    int Eid # from where/what edge it is reached
    int NidO  # from where/what node it is reached
    float Weight # the weight for priority
    int pathindex # only used for multipaths, as pathvectorindex


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int NodeReach_CmpL(const NodeReach& a, const NodeReach& b) nogil:
    return a.Weight <= b.Weight

cdef class PriorityQueue_NR:
    # during writing, I haven't figured out how to put priority_queue on cython.
    # this is temporary class specifically for NodeQueue priority queue
    cdef vector[NodeReach] NodeQueue
    
    def __cinit__(self):
        pass
    
    def __str__(self) -> str:
        return f'PriorityQueue Size {self.NodeQueue.size()}'

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void push(self, const NodeReach& item):
        # i dont really need to re-sort everything, things are already sorted.
        # deprecated version
        # self.NodeQueue.push_back(item)
        # sort(self.NodeQueue.begin(), self.NodeQueue.end(), NodeReach_CmpL)
        # this is wayyyy more effective than sort
        if self.NodeQueue.empty():
            self.NodeQueue.push_back(item)
            return

        cdef size_t size = self.NodeQueue.size()
        cdef int n
        for n in range(size):
            if self.NodeQueue[n].Weight > item.Weight:
                self.NodeQueue.insert(self.NodeQueue.begin()+n, item)
                return
        self.NodeQueue.push_back(item)
    
    cdef NodeReach top(self):
        item = self.NodeQueue.front()
        return item
    
    cdef NodeReach bot(self):
        item = self.NodeQueue.back()
        return item

    cdef NodeReach pop_top(self):
        item = self.NodeQueue.front()
        self.NodeQueue.erase(self.NodeQueue.begin())
        return item

    cdef NodeReach pop_bot(self):
        item = self.NodeQueue.back()
        self.NodeQueue.pop_back()
        return item

    cdef bint empty(self):
        return self.NodeQueue.empty()
    
    cdef size_t size(self):
        return self.NodeQueue.size()
    
    cdef void clear(self):
        self.NodeQueue.clear()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef vector[int] EdgesReach_Eididx(const vector[pair[int, float]]& vec, const int& value):
    cdef vector[int] v
    cdef int i
    cdef size_t n = vec.size()

    v.reserve(n)
    for i in range(n):
        if vec[i].first == value:
            # v.push_back(vec[i].first)
            v.push_back(i)
    return v

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint Find_IntVector(vector[int]& vec, const int val) nogil:
    cdef vector[int].iterator it = find(vec.begin(), vec.end(), val)
    return it != vec.end()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline vector[int] Pop_IntVectorVector(vector[vector[int]]& vec, const int index):
    if index >= vec.size():
        raise IndexError("Index out of range")
    cdef vector[int] item = vec[index]
    vec.erase(vec.begin() + index)
    return item

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float dist2d(const Node& N1, const Node& N2) nogil:
    """
    Node distance on 2d (only basing on x and y coordinate)

    Args:
        N1: origin Node struct
        N2: destination Node struct
    
    returns float
    """
    return sqrt((N2.pt[0] - N1.pt[0])**2 + (N2.pt[1] - N1.pt[1])**2)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float dist3d(const Node& N1, const Node& N2) nogil:
    """
    Node distance on 3d

    Args:
        N1: origin Node struct
        N2: destination Node struct
    
    returns float
    """
    cdef float dx = N2.pt[0] - N1.pt[0]
    cdef float dy = N2.pt[1] - N1.pt[1]
    cdef float dz = N2.pt[2] - N1.pt[2]
    return sqrt(dx * dx + dy * dy + dz * dz)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float dist3d_pt(const Point3d& N1, const Point3d& N2) nogil:
    """
    Node distance on 3d

    Args:
        N1: origin Node struct
        N2: destination Node struct
    
    returns float
    """
    cdef float dx = N2.x - N1.x
    cdef float dy = N2.y - N1.y
    cdef float dz = N2.z - N1.z
    return sqrt(dx * dx + dy * dy + dz * dz)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float dist3d_ar(const float[3]& N1, const float[3]& N2) nogil:
    """
    Node distance on 3d

    Args:
        N1: origin Node struct
        N2: destination Node struct
    
    returns float
    """
    cdef float dx = N2[0] - N1[0]
    cdef float dy = N2[1] - N1[1]
    cdef float dz = N2[2] - N1[2]
    return sqrt(dx * dx + dy * dy + dz * dz)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float dist3d_Py(tuple[float, float, float]& p1, tuple[float, float, float]& p2):
    """
    Node distance on 3d

    Args:
        N1: origin Node struct
        N2: destination Node struct
    
    returns float
    """
    cdef float dx = p2[0] - p1[0]
    cdef float dy = p2[1] - p1[1]
    cdef float dz = p2[2] - p1[2]
    return sqrt(dx * dx + dy * dy + dz * dz)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint SamePoint3d(Point3d& pt1, Point3d& pt2) nogil:
    if pt1.x == pt2.x and pt1.y == pt2.y and pt1.z == pt2.z:
        return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int checkclosePt(Point3d& pt, vector[Point3d]& ptLt, float tol=1e-3) nogil:
    cdef int n
    for n in range(ptLt.size()):
        if SamePoint3d(pt, ptLt[n]):
            return n
        if dist3d_pt(pt, ptLt[n]) < tol:
            return n
    return -1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int pySmallestMissing(list[int] numbers):
    """
    Finds the smallest non-negative integer not contained in the list.

    Args:
        numbers: python list of interger numbers
        n: The number of elements in the list.

    Returns:
        The smallest non-negative integer not contained in the list, 
        or -1 if all non-negative integers are present.
    """

    # Set of seen numbers (efficient for membership checks)
    cdef set[int] seen = set()
    cdef int n
    cdef int i
    
    n = int(len(list(numbers)))
    # quick exit for already sorted list of ids
    if numbers[n-1] == n-2:
        return n
    if n == 0:
        return 0
    # Add all elements from the list to the set
    for i in range(n):
        seen.add(numbers[i])

    # Check for non-negative integers starting from 0
    for i in range(n + 1):
        if i not in seen:
            return i

    # All non-negative integers are present in the list
    return -1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void arrAppendInt(int* ary, int arysize, int value):
    cdef int n
    for n in range(arysize):
        if ary[n] == -1:
            break
    ary[n] = value

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void arrPopInt(int* ary, int arysize, int value):
    cdef int n
    for n in range(arysize):
        if ary[n] == value:
            ary[n] = -1
            break

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int arrCountVals(int* ary, int arysize, int val=-1):
    cdef int count
    cdef int n
    count = 0
    for n in range(arysize):
        if ary[n] != val:
            count += 1
    return count

cdef class GraphCy:

    cdef Node* nodes
    cdef Edge* edges
    cdef int* _nodesIds
    cdef int* _edgesIds
    cdef int Nnodes
    cdef int Nedges
    cdef int EidN
    
    cdef NodeReach* nodeVisited

    def __cinit__(self, nodesize:int = 100, edgesize:int = 100, EidN:int = 10):
        self.Nnodes = nodesize
        self.nodes = <Node*>malloc(nodesize * sizeof(Node))
        self._nodesIds = <int*>malloc(nodesize * sizeof(int))

        cdef Node node
        node.idx = -1
        for n in range(nodesize):
            self.nodes[n] = node
            self._nodesIds[n] = -1

        self.Nedges = edgesize
        self.edges = <Edge*>malloc(edgesize * sizeof(Edge))
        self._edgesIds = <int*>malloc(edgesize * sizeof(int))

        cdef Edge edge
        edge.idx = -1
        for n in range(edgesize):
            self.edges[n] = edge
            self._edgesIds[n] = -1

        self.nodeVisited = <NodeReach*>malloc(self.Nnodes * sizeof(NodeReach))

        self.EidN = EidN

    def __repr__(self) -> str:
        cdef int regNodes
        cdef int regEdges
        cdef int n
        regNodes = 0
        regEdges = 0
        for n in range(self.Nnodes):
            regNodes += (self._nodesIds[n] != -1)
        for n in range(self.Nedges):
            regEdges += (self._edgesIds[n] != -1)
        return f"GraphCy object of {regNodes:,} Nodes,  {regEdges:,} Edges"
    
    def __getitem__(self, key:int) -> tuple:
        # gets edges
        cdef Edge ed = self.edges[key]
        return tuple((ed.idx, ed.NidO, ed.NidD, ed.len, ed.lenR, ed.w))
    
    def __dealloc__(self):
        free(self.nodes)
        free(self.edges)
        free(self._nodesIds)
        free(self._edgesIds)
        free(self.nodeVisited)
    
    def __reduce_ex__(self, protocol):
        cdef int n
        cdef tuple nodes = tuple((Node_tuple(self.nodes[n]) for n in range(self.Nnodes)))
        cdef tuple edges = tuple((Edge_tuple(self.edges[n]) for n in range(self.Nedges)))
        cdef tuple nodeids = Int_array_tuple(self._nodesIds, self.Nnodes)
        cdef tuple edgeids = Int_array_tuple(self._edgesIds, self.Nedges)
        return (GraphCy._reconstruct, (nodes, edges, nodeids, edgeids, self.Nnodes, self.Nedges, self.EidN))

    cdef void reinstateGraph(self, tuple nodes, tuple edges, tuple nodeids, tuple edgeids):
        cdef int n
        for n in range(self.Nnodes):
            self.nodes[n] = Node_make(nodes[n])
            self._nodesIds[n] = nodeids[n]
        for n in range(self.Nedges):
            self.edges[n] = Edge_make(edges[n])
            self._edgesIds[n] = edgeids[n]
        return

    @staticmethod
    def _reconstruct(nodes, edges, nodeids, edgeids, Nnodes, Nedges, EidN):
        cdef GraphCy gcy = GraphCy(Nnodes, Nedges, EidN)
        gcy.reinstateGraph(nodes, edges, nodeids, edgeids)
        return gcy
        
    def sizeInfo(self) -> tuple[int, int]:
        cdef int regNodes
        cdef int regEdges
        cdef int n
        regNodes = 0
        regEdges = 0
        for n in range(self.Nnodes):
            regNodes += (self._nodesIds[n] != -1)
        for n in range(self.Nedges):
            regEdges += (self._edgesIds[n] != -1)
        return regNodes, regEdges
    
    def arraySizeInfo(self) -> tuple[int, int]:
        return (self.Nnodes, self.Nedges)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef str C_reallocNodes(self, int& size):

        # copyin existing nodes
        cdef Node* oldNodes = <Node*>malloc(self.Nnodes * sizeof(Node))
        cdef int* oldNodeIds = <int*>malloc(self.Nnodes * sizeof(int))
        cdef int n
        
        oldNodes = self.nodes
        oldNodeIds = self._nodesIds

        cdef int oldNodeSize = self.Nnodes
        # making new size
        self.nodes = <Node*>malloc(size * sizeof(Node))
        self._nodesIds = <int*>malloc(size * sizeof(int))
        self.Nnodes = size
        self.nodeVisited = <NodeReach*>malloc(self.Nnodes * sizeof(NodeReach))

        cdef Node nodeR
        nodeR.idx = -1
        for n in range(size):
            self.nodes[n] = nodeR
            self._nodesIds[n] = -1
        
        if oldNodeSize > size:
            oldNodeSize = size

        for n in range(oldNodeSize):
            self.nodes[n] = oldNodes[n]
            self._nodesIds[n] = oldNodeIds[n]
        
        free(oldNodes)
        free(oldNodeIds)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void C_reallocEdges(self, int& size):

        # copyin existing edge
        cdef Edge* oldEdges = <Edge*>malloc(self.Nedges * sizeof(Edge))
        cdef int* oldEdgeIds = <int*>malloc(self.Nedges * sizeof(int))
        for n in range(self.Nedges):
            oldEdges[n] = self.edges[n]
            oldEdgeIds[n] = self._edgesIds[n]
        cdef int oldedgesize = self.Nedges

        # making new size
        self.edges = <Edge*>malloc(size * sizeof(Edge))
        self._edgesIds = <int*>malloc(size * sizeof(int))
        self.Nedges = size
        
        cdef Edge edge
        edge.idx = -1
        for n in range(size):
            self.edges[n] = edge
            self._edgesIds[n] = -1
        
        if oldedgesize > size:
            oldedgesize = size
        
        for n in range(oldedgesize):
            self.edges[n] = oldEdges[n]
            self._edgesIds[n] = oldEdgeIds[n]
    
    def reallocNodes(self, size:int|None=None):
        cdef int regNodes = self.sizeInfo()[0]
        if size == None:
            self.C_reallocNodes(regNodes)
        else:
            if size < regNodes:
                print('Warning, resizing nodes array smaller than ammount of nodes')
            self.C_reallocNodes(size)
    
    def reallocEdges(self, size:int|None=None):
        cdef int regEdges = self.sizeInfo()[1]
        if size == None:
            self.C_reallocEdges(regEdges)
        else:
            if size < regEdges:
                print('Warning, resizing nodes array smaller than ammount of nodes')
            self.C_reallocEdges(size)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void C_Reset_NodeVisited(self, int val = -1):
        for n in range(self.Nnodes):
            self.nodeVisited[n].Nid = val
    

    def get_NodeVisited(self):
        outtup = tuple((self.nodeVisited[n].Nid for n in range(self.Nnodes)))
        return outtup
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void C_addNode(self, int& idx, float& cX, float& cY, float cZ=0.0, float w=1.0, float c=0.0):
        # create Node instance
        cdef Node node
        # Insert node values
        # if idx is None or idx in self._nodesIds:
        #     idx = find_smallest_missing(self._nodesIds)
        node.idx = idx
        node.pt = (cX, cY, cZ)
        node.w = w
        node.c = c
        
        cdef int n
        for n in range(self.EidN):
            node.Eid[n] = -1

        self.nodes[idx] = node
        self._nodesIds[idx] = idx
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void C_removeEdge(self, int& idx):
        """
        removing edge, edits out node Eid.
        """
        cdef int NidO
        cdef int NidD
        NidO = self.edges[idx].NidO
        NidD = self.edges[idx].NidD
        arrPopInt(self.nodes[NidO].Eid, self.EidN, idx)
        arrPopInt(self.nodes[NidD].Eid, self.EidN, idx)
        self._edgesIds[idx] = -1

        # makes sure if no edges connected to related nodes to be dropped.
        if arrCountVals(self.nodes[NidO].Eid, self.EidN, -1) == self.EidN:
            self._nodesIds[NidO] = -1
        if arrCountVals(self.nodes[NidD].Eid, self.EidN, -1) == self.EidN:
            self._nodesIds[NidD] = -1
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void C_removeNode(self, int& idx):
        # self.nodes[idx] = NULL
        cdef int n
        self._nodesIds[idx] = -1
        # removing related edges, not needed?
        # for n in range(self.EidN):
        #     if self.nodes[idx].Eid[n] != -1:
        #         self.removeEdge(self.nodes[idx].Eid[n])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void C_addEdge(self, int& idx, int& NidO, int& NidD, float& len, float& lenR, float& w):
        # edge add on an higher hierarchi of add_node, with spatial network focusing on edges. make sure paired with add_node
        cdef Edge edge

        edge.idx = idx
        edge.NidO = NidO
        edge.NidD = NidD
        edge.len = len
        edge.lenR = lenR
        edge.w = w

        # node appending information of connected edge
        arrAppendInt(self.nodes[NidO].Eid, self.EidN, idx)
        arrAppendInt(self.nodes[NidD].Eid, self.EidN, idx)

        self.edges[idx] = edge
        self._edgesIds[idx] = idx
    
    def addEdge(self, ptOinfo:tuple, ptDinfo:tuple, lninfo:tuple) -> None:
        # input of endpoints and line
        # input tuple information structure:
        #   ptXinfo (id, coords, weight, cost)
        #   lninfo (id, length, lengthR, weight)
        # adding nodes of endpoints
        cdef float[3] ptOcoord
        cdef float[3] ptDcoord
        if self._nodesIds[ptOinfo[0]] == -1:
            ptOcoord = ptOinfo[1]
            self.C_addNode(
                idx = ptOinfo[0], 
                cX  = ptOcoord[0],
                cY  = ptOcoord[1],
                cZ  = ptOcoord[2],
                w   = ptOinfo[2],
                c   = ptOinfo[3])
        
        if self._nodesIds[ptDinfo[0]] == -1:
            ptDcoord = ptDinfo[1]
            self.C_addNode(
                idx = ptDinfo[0], 
                cX  = ptDcoord[0],
                cY  = ptDcoord[1],
                cZ  = ptDcoord[2],
                w   = ptDinfo[2],
                c   = ptDinfo[3])
        
        # adding edge
        self.C_addEdge(
            idx = lninfo[0],
            NidO = ptOinfo[0],
            NidD = ptDinfo[0],
            len = lninfo[1],
            lenR = lninfo[2],
            w = lninfo[3])
    
    def addNode(self, ptinfo:tuple) -> None:
        cdef float[3] ptOcoord
        if self._nodesIds[ptinfo[0]] == -1:
            coord = ptinfo[1]
            if len(coord) == 2: # if 2d point turn into 3d
                ptOcoord = (coord[0], coord[1], 0.0)
            else:
                ptOcoord = coord
            self.C_addNode(
                idx = ptinfo[0], 
                cX  = ptOcoord[0],
                cY  = ptOcoord[1],
                cZ  = ptOcoord[2],
                w   = ptinfo[2],
                c   = ptinfo[3])
    
    def changeNode(self, ptinfo:tuple) -> None:
        cdef float[3] ptOcoord
        coord = ptinfo[1]
        if len(coord) == 2: # if 2d point turn into 3d
            ptOcoord = (coord[0], coord[1], 0.0)
        else:
            ptOcoord = coord
        self.C_addNode(
            idx = ptinfo[0], 
            cX  = ptOcoord[0],
            cY  = ptOcoord[1],
            cZ  = ptOcoord[2],
            w   = ptinfo[2],
            c   = ptinfo[3])

    def addEdgefromNodes(self, ptOid:int, ptDid:int, lninfo:tuple) -> None:
        # making sure the connecting nodes already exist
        if self._nodesIds[ptOid] != -1 and self._nodesIds[ptDid] != -1:
            self.C_addEdge(
                idx = lninfo[0],
                NidO = ptOid,
                NidD = ptDid,
                len = lninfo[1],
                lenR = lninfo[2],
                w = lninfo[3])
    
    def addNodes(self, ptsinfo:tuple) -> None:
        cdef float[3] ptOcoord
        cdef bint threeD = True
        if len(ptsinfo[1]) == 2: # if 2d point turn into 3d
            threeD = False
        for pt in ptsinfo:
            if self._nodesIds[pt[0]] == -1:
                if threeD:
                    ptOcoord = pt[1]
                else:
                    ptOcoord = pt[1] + (0.0,)
                self.C_addNode(
                    idx = pt[0], 
                    cX  = ptOcoord[0],
                    cY  = ptOcoord[1],
                    cZ  = ptOcoord[2],
                    w   = pt[2],
                    c   = pt[3])
    
    def fromGeopandas_Edges(self, dfNetwork:gpd.GeoDataFrame,
        A_Lnlength:str|None = None,
        A_LnlengthR:str|None = None,
        A_LnW:str|None = None,
        A_PtstW:str|None = None,
        A_PtstC:str|None = None,
        A_PtedW:str|None = None,
        A_PtedC:str|None = None,
        ) -> None:
        # edgesInfo consist of
        # (ptoid, ptdid, lninfo(edgeid, len, lenR, weight))
        cdef int index
        cdef Point3d lnSt
        cdef Point3d lnEd
        cdef vector[Point3d] pointCoords
        cdef int pointidCnt = 0
        cdef int edgeidCnt = 0
        cdef int idSt
        cdef int idEd
        cdef bint State_3d = True
        cdef int ckSt
        cdef int dfSize = int(dfNetwork.size)
        
        dfKeys = tuple(dfNetwork.columns)
        # checking if fields exists
        if A_Lnlength != None:
            if A_Lnlength not in dfKeys:
                print('Atribute for line length not found, using geometric weight')
                Lnlength = tuple(dfNetwork.geometry.length)
            else:
                Lnlength = tuple(dfNetwork[A_Lnlength])
        else:
            Lnlength = tuple(dfNetwork.geometry.length)
        if A_LnlengthR != None:
            if A_LnlengthR not in dfKeys:
                print('Atribute for line length reverse not found, using same length as lnlength')
                LnlengthR = Lnlength
            else:
                LnlengthR = tuple(dfNetwork[A_LnlengthR])
        else:
            LnlengthR = Lnlength
        if A_LnW != None:
            if A_LnW not in dfKeys:
                print('Attribute for line weight not found, defaulting to 1.0')
                LnW = (1.0,)*dfSize
            else:
                LnW = tuple(dfNetwork[A_LnW])
        else:
            LnW = (1.0,)*dfSize
        if A_PtstW != None:
            if A_PtstW not in dfKeys:
                print('Edge point start weight attribute not found, defaulting to 1.0')
                PtstW = (1.0,)*dfSize
            else:
                PtstW = tuple(dfNetwork[A_PtstW])
        else:
            PtstW = (1.0,)*dfSize
        if A_PtstC != None:
            if A_PtstC not in dfKeys:
                print('Edge point start cost attribute not found, defaulting to 0.0')
                PtstC = (0.0,)*dfSize
            else:
                PtstC = tuple(dfNetwork[A_PtstC])
        else:
            PtstC = (0.0,)*dfSize
        if A_PtedW != None:
            if A_PtedW not in dfKeys:
                print('Edge point end weight attribute not found, defaulting to 1.0')
                PtedW = (1.0,)*dfSize
            else:
                PtedW = tuple(dfNetwork[A_PtedW])
        else:
            PtedW = (1.0,)*dfSize
        if A_PtedC != None:
            if A_PtedC not in dfKeys:
                print('Edge point end cost attribute not found, defaulting to 0.0')
                PtedC = (0.0,)*dfSize
            else:
                PtedC = tuple(dfNetwork[A_PtedC])
        else:
            PtedC = (0.0,)*dfSize


        if len(dfNetwork.geometry[0].coords[0]) == 2:
            State_3d = False # checks dimension
        
        for index, row in dfNetwork.iterrows():
            if State_3d:
                lnSt = MakePoint3d(row.geometry.coords[0][0], row.geometry.coords[0][1], row.geometry.coords[0][2])
                lnEd = MakePoint3d(row.geometry.coords[-1][0], row.geometry.coords[-1][1], row.geometry.coords[-1][2])
            else: # for 2d points, appending 3d
                lnSt = MakePoint3d(row.geometry.coords[0][0], row.geometry.coords[0][1], 0.0)
                lnEd = MakePoint3d(row.geometry.coords[-1][0], row.geometry.coords[-1][1], 0.0)

            ckSt = checkclosePt(lnSt, pointCoords)
            if ckSt == -1:
                pointCoords.push_back(lnSt)
                idSt = pointidCnt
                self.C_addNode(idSt, lnSt.x, lnSt.y, lnSt.z, PtstW[edgeidCnt], PtstC[edgeidCnt])
                pointidCnt += 1
            else:
                idSt = ckSt
            
            ckEd = checkclosePt(lnEd, pointCoords)
            if ckEd == -1:
                pointCoords.push_back(lnEd)
                idEd = pointidCnt
                self.C_addNode(idEd, lnEd.x, lnEd.y, lnEd.z, PtedW[edgeidCnt], PtedC[edgeidCnt])
                pointidCnt += 1
            else:
                idEd = ckEd 
            
            self.C_addEdge(index, idSt, idEd, Lnlength[edgeidCnt], LnlengthR[edgeidCnt], LnW[edgeidCnt]) # add edge
            edgeidCnt += 1

        # print(f'Add edges from geopandas successfull, added {pointidCnt:,} nodes, and {edgeidCnt:,} edges')

    def removeEdge(self, idx:int) -> None:
        self.C_removeEdge(idx)
    
    def removeEdges(self, ids:tuple) -> None:
        cdef int n
        for n in ids:
            self.C_removeEdge(n)

    def removeNode(self, idx:int) -> None:
        self.C_removeNode(idx)

    def PathLength(self, edges:tuple) -> float:
        cdef float length = 0.0
        cdef int n
        cdef Edge elook
        cdef Edge flook
        # it has to know if it is reversed or not
        if len(edges) == 1:
            return length
        for n in range(len(edges)):
            if n == len(edges)-1:
                elook = self.edges[edges[n]]
                flook = self.edges[edges[n-1]]
                if elook.NidO == flook.NidO or elook.NidO == flook.NidD:
                    length += elook.len
                else:
                    length += elook.lenR
            else:
                elook = self.edges[edges[n]]
                flook = self.edges[edges[n+1]]
                if elook.NidD == flook.NidO or elook.NidD == flook.NidD:
                    length += elook.len
                else:
                    length += elook.lenR
            
        return length

    def PathFind_Dijkstra(self, int NidO, int NidD, float LimDist = 10_000.0, int LimCycle = 10_000) -> tuple[float, tuple[int]]:
        """
        Find smallest distance and edges traversed between two nodes in the graph.

        Parameters
        ------------
        NidO : int
            starting node ID
        NidD : int
            destination node ID
        LimDist : float, default 10,000.0
            distance limit before giving up the pathfinding process
        LimCycle : int, default 10,000
            number of cycles of priority_queue before giving up the pathfinding process
        
        Returns
        -----------
        length: float
            if a path is found, it will return with the length of shortest possible distance of the path
        
        Edge IDs: tuple[int]
            tuple of edge ids of found shortest path.
        
        if path is not found, it will return (-1.0,(-1,))
        
        Notes
        -----------
        The algorithm has bidirectional capability and also node cost added.
        """
        cdef float BaseDist = dist3d(self.nodes[NidO], self.nodes[NidD])
        if BaseDist*0.9 > LimDist:
            return -1.0

        self.C_Reset_NodeVisited()
        
        cdef int cycles
        cycles = 1
        cdef bint keepGoing
        keepGoing = True

        # cdef vector[NodeReach] OpenNodes
        cdef PriorityQueue_NR OpenNodes
        cdef NodeReach startNode
        startNode.Nid = NidO
        startNode.Eid = -1
        startNode.Dist = 0.0
        startNode.NidO = -1
        startNode.Weight = 0.0
        OpenNodes = PriorityQueue_NR()
        OpenNodes.push(startNode)

        cdef NodeReach NodeReach_T
        cdef int NidF
        cdef float len
        cdef Edge EdgeC
        cdef NodeReach NodeCheck

        cdef vector[int] pth
        cdef int nlook

        while keepGoing:
            cycles += 1
            # check paths from OpenNodes
            NodeCheck = OpenNodes.pop_top()

            if NodeCheck.Nid == NidD:
                break

            for i in range(self.EidN):
                Eid = self.nodes[NodeCheck.Nid].Eid[i]
                if Eid == -1 or Eid == NodeCheck.Eid or self._edgesIds[Eid] == -1:
                    continue
                # possible new path
                # new node?
                EdgeC = self.edges[Eid]
                if EdgeC.NidO == NodeCheck.Nid:
                    len = EdgeC.len
                    NidF = EdgeC.NidD
                else :
                    len = EdgeC.lenR
                    NidF = EdgeC.NidO

                if self._nodesIds[NidF] == -1:
                    continue
                
                len += NodeCheck.Dist + self.nodes[NidF].c

                if len > LimDist:
                    continue

                NodeReach_T.Nid = NidF
                NodeReach_T.Dist = len
                NodeReach_T.Eid = Eid
                NodeReach_T.NidO = NodeCheck.Nid
                NodeReach_T.Weight = len

                # check to visited nodes
                if self.nodeVisited[NidF].Nid == -1 : # if node havent been visited
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)
                elif self.nodeVisited[NidF].Dist > len: # if visited node has a higher distance
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)

            if cycles > LimCycle or OpenNodes.top().Dist > LimDist:
                # if reaches cycle count limit
                # if the minimum distance in open nodes is more than the distance limit
                keepGoing = False
            elif OpenNodes.empty():
                keepGoing = False

        if not keepGoing:
            return (-1.0,(-1,))
        else:
            # if found path, retracing steps
            nlook = NidD
            while nlook != NidO:
                pth.insert(pth.begin(),self.nodeVisited[nlook].Eid)
                nlook = self.nodeVisited[nlook].NidO
            # pth.insert(pth.begin(),nlook) # inserting the NidO
            return (NodeCheck.Dist, tuple(pth))
    
    def PathDist_Dijkstra(self, int NidO, int NidD, float LimDist = 10_000.0, int LimCycle = 10_000) -> float:
        """
        Find smallest distance between two nodes in the graph.

        Parameters
        ------------
        NidO : int
            starting node ID
        NidD : int
            destination node ID
        LimDist : float, default 10,000.0
            distance limit before giving up the pathfinding process
        LimCycle : int, default 10,000
            number of cycles of priority_queue before giving up the pathfinding process
        
        Returns
        -----------
        length: float
            if a path is found, it will return with the length of shortest possible distance of the path
            if a path is not found, it will return -1.0
        
        Notes
        -----------
        The algorithm has bidirectional capability and also node cost added.
        """
        cdef float BaseDist = dist3d(self.nodes[NidO], self.nodes[NidD])
        if BaseDist*0.9 > LimDist:
            return -1.0

        self.C_Reset_NodeVisited()
        
        cdef int cycles
        cycles = 1
        cdef bint keepGoing
        keepGoing = True

        # cdef vector[NodeReach] OpenNodes
        cdef PriorityQueue_NR OpenNodes
        cdef NodeReach startNode
        startNode.Nid = NidO
        startNode.Eid = -1
        startNode.Dist = 0.0
        startNode.NidO = -1
        startNode.Weight = 0.0
        OpenNodes = PriorityQueue_NR()
        OpenNodes.push(startNode)

        cdef NodeReach NodeReach_T
        cdef int NidF
        cdef float len
        cdef Edge EdgeC
        cdef NodeReach NodeCheck

        while keepGoing:
            cycles += 1
            # check paths from OpenNodes
            NodeCheck = OpenNodes.pop_top()

            if NodeCheck.Nid == NidD:
                return NodeCheck.Dist

            for i in range(self.EidN):
                Eid = self.nodes[NodeCheck.Nid].Eid[i]
                if Eid == -1 or Eid == NodeCheck.Eid or self._edgesIds[Eid] == -1:
                    continue
                # possible new path
                # new node?
                EdgeC = self.edges[Eid]
                if EdgeC.NidO == NodeCheck.Nid:
                    len = EdgeC.len
                    NidF = EdgeC.NidD
                else :
                    len = EdgeC.lenR
                    NidF = EdgeC.NidO

                if self._nodesIds[NidF] == -1:
                    continue

                len += NodeCheck.Dist + self.nodes[NidF].c

                if len > LimDist:
                    continue

                NodeReach_T.Nid = NidF
                NodeReach_T.Dist = len
                NodeReach_T.Eid = Eid
                NodeReach_T.NidO = NodeCheck.Nid
                NodeReach_T.Weight = len

                # check to visited nodes
                if self.nodeVisited[NidF].Nid == -1 : # if node havent been visited
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)
                elif self.nodeVisited[NidF].Dist > len: # if visited node has a higher distance
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)

            if cycles > LimCycle or OpenNodes.top().Dist > LimDist:
                # if reaches cycle count limit
                # if the minimum distance in open nodes is more than the distance limit
                keepGoing = False
            elif OpenNodes.empty():
                keepGoing = False

        return -1.0

    def PathFind_AStar(self, int NidO, int NidD, float LimDist = 10_000.0, int LimCycle = 10_000, float DistMul = 2.0) -> tuple[float, tuple[int]]:
        """
        Find smallest distance and edges traversed between two nodes in the graph. Using Astar principle from 3d location information.

        Parameters
        ------------
        NidO : int
            starting node ID
        NidD : int
            destination node ID
        LimDist : float, default 10,000.0
            distance limit before giving up the pathfinding process
        LimCycle : int, default 10,000
            number of cycles of priority_queue before giving up the pathfinding process
        DistMul : float, default 1.0
            A* weighting multiplier for the remaining cartesian distance
        
        Returns
        -----------
        length: float
            if a path is found, it will return with the length of shortest possible distance of the path
        
        Edge IDs: tuple[int]
            tuple of edge ids of found shortest path.
        
        if path is not found, it will return (-1.0,(-1,))
        
        Notes
        -----------
        The algorithm has bidirectional capability and also node cost added.
        PriorityQueue weighting using:
            minimum(distance_NetworkDistance + (distance_RemainingCartesianDistance x Distmul) )
        """
        cdef float BaseDist = dist3d(self.nodes[NidO], self.nodes[NidD])
        if BaseDist*0.9 > LimDist:
            return -1.0

        self.C_Reset_NodeVisited()
        
        cdef int cycles
        cycles = 1
        cdef bint keepGoing
        keepGoing = True

        # cdef vector[NodeReach] OpenNodes
        cdef PriorityQueue_NR OpenNodes
        cdef NodeReach startNode
        startNode.Nid = NidO
        startNode.Eid = -1
        startNode.Dist = 0.0
        startNode.NidO = -1
        startNode.Weight = 0.0
        OpenNodes = PriorityQueue_NR()
        OpenNodes.push(startNode)

        cdef NodeReach NodeReach_T
        cdef int NidF
        cdef float len
        cdef Edge EdgeC
        cdef NodeReach NodeCheck
        cdef Node NodeTarget
        NodeTarget = self.nodes[NidD]

        cdef vector[int] pth
        cdef int nlook

        while keepGoing:
            cycles += 1
            # check paths from OpenNodes
            NodeCheck = OpenNodes.pop_top()

            if NodeCheck.Nid == NidD:
                break

            for i in range(self.EidN):
                Eid = self.nodes[NodeCheck.Nid].Eid[i]
                if Eid == -1 or Eid == NodeCheck.Eid or self._edgesIds[Eid] == -1:
                    continue
                # possible new path
                # new node?
                EdgeC = self.edges[Eid]
                if EdgeC.NidO == NodeCheck.Nid:
                    len = EdgeC.len
                    NidF = EdgeC.NidD
                else :
                    len = EdgeC.lenR
                    NidF = EdgeC.NidO

                if self._nodesIds[NidF] == -1:
                    continue
                
                len += NodeCheck.Dist + self.nodes[NidF].c

                if len > LimDist:
                    continue

                NodeReach_T.Nid = NidF
                NodeReach_T.Dist = len
                NodeReach_T.Eid = Eid
                NodeReach_T.NidO = NodeCheck.Nid
                NodeReach_T.Weight = len + (dist3d(self.nodes[NidF], NodeTarget)- BaseDist) * DistMul

                # check to visited nodes
                if self.nodeVisited[NidF].Nid == -1 : # if node havent been visited
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)
                elif self.nodeVisited[NidF].Dist > len: # if visited node has a higher distance
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)

            if cycles > LimCycle or OpenNodes.top().Dist > LimDist:
                # if reaches cycle count limit
                # if the minimum distance in open nodes is more than the distance limit
                keepGoing = False
            elif OpenNodes.empty():
                keepGoing = False

        if not keepGoing:
            return (-1.0,(-1,))
        else:
            # if found path, retracing steps
            nlook = NidD
            while nlook != NidO:
                pth.insert(pth.begin(),self.nodeVisited[nlook].Eid)
                nlook = self.nodeVisited[nlook].NidO
            # pth.insert(pth.begin(),nlook) # inserting the NidO
            return (NodeCheck.Dist, tuple(pth))
    
    def PathFind_AStar_VirtuEntry(self, 
            const int& EidO, tuple[float, float, float] PtO, const tuple[float, float] DstO,
            const int& EidD, tuple[float, float, float] PtD, const tuple[float, float] DstD,  
            float LimDist = 10_000.0, int LimCycle = 10_000, float DistMul = 2.0) -> tuple[float, tuple[int]]:
        """
        Find smallest distance and edges traversed between two nodes in the graph. Using Astar principle from 3d location information.
        Virtual Entries, located on an edge, represented by edge id, node coordinate, and distance cost to each end

        Parameters
        ------------
        EidO : int
            Edge Id
        EidD : int
            Edge Destination
        LimDist : float, default 10,000.0
            distance limit before giving up the pathfinding process
        LimCycle : int, default 10,000
            number of cycles of priority_queue before giving up the pathfinding process
        DistMul : float, default 1.0
            A* weighting multiplier for the remaining cartesian distance
        
        Returns
        -----------
        length: float
            if a path is found, it will return with the length of shortest possible distance of the path
        
        Edge IDs: tuple[int]
            tuple of edge ids of found shortest path.
        
        if path is not found, it will return (-1.0,(-1,))
        
        Notes
        -----------
        The algorithm has bidirectional capability and also node cost added.
        PriorityQueue weighting using:
            minimum(distance_NetworkDistance + (distance_RemainingCartesianDistance x Distmul) )
        """
        if self._edgesIds[EidO] == -1 or self._edgesIds[EidD] == -1:
            return None
        cdef float[3] PointD = (PtD[0], PtD[1], PtD[2])
        cdef float[3] PointO = (PtO[0], PtO[1], PtO[2])
        cdef float BaseDist = dist3d_ar(PointO, PointD)
        if BaseDist*0.9 > LimDist:
            return -1.0

        self.C_Reset_NodeVisited()
        
        cdef int cycles
        cycles = 1
        cdef bint keepGoing
        keepGoing = True

        # cdef vector[NodeReach] OpenNodes
        cdef PriorityQueue_NR OpenNodes
        cdef NodeReach startNode
        OpenNodes = PriorityQueue_NR()
        
        # for Origin EdgeOrigin
        startNode.Nid = self.edges[EidO].NidO
        startNode.Eid = EidO
        startNode.Dist = DstO[0]
        startNode.NidO = -1
        startNode.Weight = DstO[0] + (dist3d_ar(self.nodes[startNode.Nid].pt, PointD)- BaseDist) * DistMul
        OpenNodes.push(startNode)
        # for destination edgeorigin
        startNode.Nid = self.edges[EidO].NidD
        startNode.Eid = EidO
        startNode.Dist = DstO[1]
        startNode.NidO = -1
        startNode.Weight = DstO[1] + (dist3d_ar(self.nodes[startNode.Nid].pt, PointD)- BaseDist) * DistMul
        OpenNodes.push(startNode)

        cdef NodeReach NodeReach_T
        cdef int NidF
        cdef float len
        cdef Edge EdgeC
        cdef NodeReach NodeCheck

        cdef vector[int] pth
        cdef int nlook

        while keepGoing:
            cycles += 1
            # check paths from OpenNodes
            NodeCheck = OpenNodes.pop_top()

            if NodeCheck.Eid == EidD:
                if NodeCheck.Nid == self.edges[EidD].NidO:
                    NodeCheck.Dist += DstD[0]
                else:
                    NodeCheck.Dist += DstD[1]
                break

            if NodeCheck.Eid == EidD:
                    if NodeCheck.Nid == self.edges[EidD].NidO:
                        NodeCheck.Dist -= self.edges[EidD].lenR - DstD[1]
                    else:
                        NodeCheck.Dist -= self.edges[EidD].len - DstD[0]

            for i in range(self.EidN):
                Eid = self.nodes[NodeCheck.Nid].Eid[i]
                if Eid == -1 or Eid == NodeCheck.Eid or self._edgesIds[Eid] == -1:
                    continue
                # possible new path
                # new node?
                EdgeC = self.edges[Eid]
                if EdgeC.NidO == NodeCheck.Nid:
                    len = EdgeC.len
                    NidF = EdgeC.NidD
                else :
                    len = EdgeC.lenR
                    NidF = EdgeC.NidO

                if self._nodesIds[NidF] == -1:
                    continue
                
                len += NodeCheck.Dist + self.nodes[NidF].c

                if len > LimDist:
                    continue

                NodeReach_T.Nid = NidF
                NodeReach_T.Dist = len
                NodeReach_T.Eid = Eid
                NodeReach_T.NidO = NodeCheck.Nid
                NodeReach_T.Weight = len + (dist3d_ar(self.nodes[NidF].pt, PointD)- BaseDist) * DistMul

                # check to visited nodes
                if self.nodeVisited[NidF].Nid == -1 : # if node havent been visited
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)
                elif self.nodeVisited[NidF].Dist > len: # if visited node has a higher distance
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)

            if cycles > LimCycle or OpenNodes.top().Dist > LimDist:
                # if reaches cycle count limit
                # if the minimum distance in open nodes is more than the distance limit
                keepGoing = False
            elif OpenNodes.empty():
                keepGoing = False

        if not keepGoing:
            return (-1.0,(-1,))
        else:
            # if found path, retracing steps
            nlook = NodeCheck.Nid
            pth.push_back(EidD)
            while nlook != self.edges[Eid].NidO or nlook != self.edges[Eid].NidD:
                pth.insert(pth.begin(),self.nodeVisited[nlook].Eid)
                nlook = self.nodeVisited[nlook].NidO
            # pth.insert(pth.begin(),nlook) # inserting the NidO
            pth.insert(pth.begin(), EidO)
            return (NodeCheck.Dist, tuple(pth))
    
    def PathDist_AStar(self, int NidO, int NidD, float LimDist = 10_000.0, int LimCycle = 10_000, float DistMul = 2.0) -> float:
        """
        Find smallest distance between two nodes in the graph. Using Astar principle from 3d location information.

        Parameters
        ------------
        NidO : int
            starting node ID
        NidD : int
            destination node ID
        LimDist : float, default 10,000.0
            distance limit before giving up the pathfinding process
        LimCycle : int, default 10,000
            number of cycles of priority_queue before giving up the pathfinding process
        DistMul : float, default 1.0
            A* weighting multiplier for the remaining cartesian distance
        
        Returns
        -----------
        length: float
            if a path is found, it will return with the length of shortest possible distance of the path
            if a path is not found, it will return -1.0
        
        Notes
        -----------
        The algorithm has bidirectional capability and also node cost added.
        PriorityQueue weighting using:
            minimum(distance_NetworkDistance + (distance_RemainingCartesianDistance x Distmul) )
        """
        cdef float BaseDist = dist3d(self.nodes[NidO], self.nodes[NidD])
        if BaseDist*0.9 > LimDist:
            return -1.0

        self.C_Reset_NodeVisited()
        
        cdef int cycles
        cycles = 1
        cdef bint keepGoing
        keepGoing = True

        cdef PriorityQueue_NR OpenNodes
        cdef NodeReach startNode
        startNode.Nid = NidO
        startNode.Eid = -1
        startNode.Dist = 0.0
        startNode.NidO = -1
        startNode.Weight = 0.0
        OpenNodes = PriorityQueue_NR()
        OpenNodes.push(startNode)

        cdef NodeReach NodeReach_T
        cdef int NidF
        cdef float len
        cdef Edge EdgeC
        cdef NodeReach NodeCheck
        cdef Node NodeTarget
        NodeTarget = self.nodes[NidD]

        while keepGoing:
            cycles += 1
            # check paths from OpenNodes
            NodeCheck = OpenNodes.pop_top()

            if NodeCheck.Nid == NidD:
                return NodeCheck.Dist

            for i in range(self.EidN):
                Eid = self.nodes[NodeCheck.Nid].Eid[i]
                if Eid == -1 or Eid == NodeCheck.Eid or self._edgesIds[Eid] == -1:
                    continue
                # possible new path
                # new node?
                EdgeC = self.edges[Eid]
                if EdgeC.NidO == NodeCheck.Nid:
                    len = EdgeC.len
                    NidF = EdgeC.NidD
                else :
                    len = EdgeC.lenR
                    NidF = EdgeC.NidO

                if self._nodesIds[NidF] == -1:
                    continue

                len += NodeCheck.Dist + self.nodes[NidF].c

                if len > LimDist:
                    continue

                NodeReach_T.Nid = NidF
                NodeReach_T.Dist = len
                NodeReach_T.Eid = Eid
                NodeReach_T.NidO = NodeCheck.Nid
                NodeReach_T.Weight = len + (dist3d(self.nodes[NidF], NodeTarget)- BaseDist) * DistMul

                # check to visited nodes
                if self.nodeVisited[NidF].Nid == -1 : # if node havent been visited
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)
                elif self.nodeVisited[NidF].Dist > len: # if visited node has a higher distance
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)

            if cycles > LimCycle or OpenNodes.top().Dist > LimDist:
                # if reaches cycle count limit
                # if the minimum distance in open nodes is more than the distance limit
                keepGoing = False
            elif OpenNodes.empty():
                keepGoing = False

        return -1.0

    def PathDist_AStar_VirtuEntry(self, 
            const int& EidO, tuple[float, float, float] PtO, const tuple[float, float] DstO,
            const int& EidD, tuple[float, float, float] PtD, const tuple[float, float] DstD, 
            float LimDist = 10_000.0, 
            int LimCycle = 10_000, 
            float DistMul = 2.0) -> float:
        """
        Find smallest distance between two nodes in the graph. Using Astar principle from 3d location information.

        Parameters
        ------------
        NidO : int
            starting node ID
        NidD : int
            destination node ID
        LimDist : float, default 10,000.0
            distance limit before giving up the pathfinding process
        LimCycle : int, default 10,000
            number of cycles of priority_queue before giving up the pathfinding process
        DistMul : float, default 1.0
            A* weighting multiplier for the remaining cartesian distance
        
        Returns
        -----------
        length: float
            if a path is found, it will return with the length of shortest possible distance of the path
            if a path is not found, it will return -1.0
        
        Notes
        -----------
        The algorithm has bidirectional capability and also node cost added.
        PriorityQueue weighting using:
            minimum(distance_NetworkDistance + (distance_RemainingCartesianDistance x Distmul) )
        """
        # checking edges exists
        if self._edgesIds[EidO] == -1 or self._edgesIds[EidD] == -1:
            return None
        cdef float[3] PointD = (PtD[0], PtD[1], PtD[2])
        cdef float[3] PointO = (PtO[0], PtO[1], PtO[2])
        cdef float BaseDist = dist3d_ar(PointO, PointD)
        if BaseDist*0.9 > LimDist:
            return -1.0

        if EidD == EidO:
            return abs(DstD[0] - DstO[1])

        self.C_Reset_NodeVisited()
        
        cdef int cycles
        cycles = 1
        cdef bint keepGoing
        keepGoing = True

        # cdef vector[NodeReach] OpenNodes
        cdef PriorityQueue_NR OpenNodes
        cdef NodeReach startNode
        OpenNodes = PriorityQueue_NR()
        
        # for Origin EdgeOrigin
        startNode.Nid = self.edges[EidO].NidO
        startNode.Eid = EidO
        startNode.Dist = DstO[0]
        startNode.NidO = -1
        startNode.Weight = DstO[0] + (dist3d_ar(self.nodes[startNode.Nid].pt, PointD)- BaseDist) * DistMul
        OpenNodes.push(startNode)
        # for destination edgeorigin
        startNode.Nid = self.edges[EidO].NidD
        startNode.Eid = EidO
        startNode.Dist = DstO[1]
        startNode.NidO = -1
        startNode.Weight = DstO[1] + (dist3d_ar(self.nodes[startNode.Nid].pt, PointD)- BaseDist) * DistMul
        OpenNodes.push(startNode)

        cdef NodeReach NodeReach_T
        cdef int NidF
        cdef float len
        cdef Edge EdgeC
        cdef NodeReach NodeCheck

        while keepGoing:
            cycles += 1
            # check paths from OpenNodes
            NodeCheck = OpenNodes.pop_top()

            if NodeCheck.Eid == EidD:
                    if NodeCheck.Nid == self.edges[EidD].NidO:
                        return NodeCheck.Dist - self.edges[EidD].lenR + DstD[1]
                    else:
                        return NodeCheck.Dist - self.edges[EidD].len + DstD[0]

            for i in range(self.EidN):
                Eid = self.nodes[NodeCheck.Nid].Eid[i]
                if Eid == -1 or Eid == NodeCheck.Eid or self._edgesIds[Eid] == -1:
                    continue

                # possible new path
                # new node?
                EdgeC = self.edges[Eid]
                if EdgeC.NidO == NodeCheck.Nid:
                    len = EdgeC.len
                    NidF = EdgeC.NidD
                else :
                    len = EdgeC.lenR
                    NidF = EdgeC.NidO

                if self._nodesIds[NidF] == -1:
                    continue
                
                len += NodeCheck.Dist + self.nodes[NidF].c

                if len > LimDist:
                    continue

                NodeReach_T.Nid = NidF
                NodeReach_T.Dist = len
                NodeReach_T.Eid = Eid
                NodeReach_T.NidO = NodeCheck.Nid
                NodeReach_T.Weight = len + (dist3d_ar(self.nodes[NidF].pt, PointD)- BaseDist) * DistMul

                # check to visited nodes
                if self.nodeVisited[NidF].Nid == -1 : # if node havent been visited
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)
                elif self.nodeVisited[NidF].Dist > len: # if visited node has a higher distance
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)

            if cycles > LimCycle or OpenNodes.top().Dist > LimDist:
                # if reaches cycle count limit
                # if the minimum distance in open nodes is more than the distance limit
                keepGoing = False
            elif OpenNodes.empty():
                keepGoing = False

        return -1.0
    
    def PathDistComp_AStar_VirtuEntry(self, 
            const int& EidO, tuple[float, float, float] PtO, const tuple[float, float] DstO,
            const int& EidD, tuple[float, float, float] PtD, const tuple[float, float] DstD, 
            float LimDist = 10_000.0, 
            int LimCycle = 10_000, 
            float DistMul = 2.0) -> tuple[float, float]:
        """
        Find smallest distance between two nodes in the graph. Using Astar principle from 3d location information.

        Parameters
        ------------
        NidO : int
            starting node ID
        NidD : int
            destination node ID
        LimDist : float, default 10,000.0
            distance limit before giving up the pathfinding process
        LimCycle : int, default 10,000
            number of cycles of priority_queue before giving up the pathfinding process
        DistMul : float, default 1.0
            A* weighting multiplier for the remaining cartesian distance
        
        Returns
        -----------
        length: float
            if a path is found, it will return with the length of shortest possible distance of the path
            if a path is not found, it will return -1.0
        
        Notes
        -----------
        The algorithm has bidirectional capability and also node cost added.
        PriorityQueue weighting using:
            minimum(distance_NetworkDistance + (distance_RemainingCartesianDistance x Distmul) )
        """
        # checking edges exists
        if self._edgesIds[EidO] == -1 or self._edgesIds[EidD] == -1:
            return (None, None)
        cdef float[3] PointD = (PtD[0], PtD[1], PtD[2])
        cdef float[3] PointO = (PtO[0], PtO[1], PtO[2])
        cdef float BaseDist = dist3d_ar(PointO, PointD)
        if BaseDist*0.9 > LimDist:
            return (-1.0, -1.0)

        if EidD == EidO:
            return (abs(DstD[0] - DstO[1]), BaseDist)

        self.C_Reset_NodeVisited()
        
        cdef int cycles
        cycles = 1
        cdef bint keepGoing
        keepGoing = True

        # cdef vector[NodeReach] OpenNodes
        cdef PriorityQueue_NR OpenNodes
        cdef NodeReach startNode
        OpenNodes = PriorityQueue_NR()
        
        # for Origin EdgeOrigin
        startNode.Nid = self.edges[EidO].NidO
        startNode.Eid = EidO
        startNode.Dist = DstO[0]
        startNode.NidO = -1
        startNode.Weight = DstO[0] + (dist3d_ar(self.nodes[startNode.Nid].pt, PointD)- BaseDist) * DistMul
        OpenNodes.push(startNode)
        # for destination edgeorigin
        startNode.Nid = self.edges[EidO].NidD
        startNode.Eid = EidO
        startNode.Dist = DstO[1]
        startNode.NidO = -1
        startNode.Weight = DstO[1] + (dist3d_ar(self.nodes[startNode.Nid].pt, PointD)- BaseDist) * DistMul
        OpenNodes.push(startNode)

        cdef NodeReach NodeReach_T
        cdef int NidF
        cdef float len
        cdef Edge EdgeC
        cdef NodeReach NodeCheck

        while keepGoing:
            cycles += 1
            # check paths from OpenNodes
            NodeCheck = OpenNodes.pop_top()

            if NodeCheck.Eid == EidD:
                    if NodeCheck.Nid == self.edges[EidD].NidO:
                        return ((NodeCheck.Dist - self.edges[EidD].lenR + DstD[1]), BaseDist)
                    else:
                        return ((NodeCheck.Dist - self.edges[EidD].len + DstD[0]), BaseDist)

            for i in range(self.EidN):
                Eid = self.nodes[NodeCheck.Nid].Eid[i]
                if Eid == -1 or Eid == NodeCheck.Eid or self._edgesIds[Eid] == -1:
                    continue

                # possible new path
                # new node?
                EdgeC = self.edges[Eid]
                if EdgeC.NidO == NodeCheck.Nid:
                    len = EdgeC.len
                    NidF = EdgeC.NidD
                else :
                    len = EdgeC.lenR
                    NidF = EdgeC.NidO

                if self._nodesIds[NidF] == -1:
                    continue
                
                len += NodeCheck.Dist + self.nodes[NidF].c

                if len > LimDist:
                    continue

                NodeReach_T.Nid = NidF
                NodeReach_T.Dist = len
                NodeReach_T.Eid = Eid
                NodeReach_T.NidO = NodeCheck.Nid
                NodeReach_T.Weight = len + (dist3d_ar(self.nodes[NidF].pt, PointD)- BaseDist) * DistMul

                # check to visited nodes
                if self.nodeVisited[NidF].Nid == -1 : # if node havent been visited
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)
                elif self.nodeVisited[NidF].Dist > len: # if visited node has a higher distance
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)

            if cycles > LimCycle or OpenNodes.top().Dist > LimDist:
                # if reaches cycle count limit
                # if the minimum distance in open nodes is more than the distance limit
                keepGoing = False
            elif OpenNodes.empty():
                keepGoing = False

        return (-1.0, -1.0)

    def PathReach(self, int NidO, float LimDist = 1_000.0, int LimCycle = 10_000) -> tuple[tuple[int, float]]:
        """
        Finds Edges within reach

        Parameters
        ------------
        NidO : int
            starting node ID
        LimDist : float, default 10,000.0
            distance limit before giving up the pathfinding process
        LimCycle : int, default 10,000
            number of cycles of priority_queue before giving up the pathfinding process
        
        Returns
        -----------
        length: float
            if a path is found, it will return with the length of shortest possible distance of the path
        
        Edge IDs: tuple[int]
            tuple of edge ids of found shortest path.
        
        if path is not found, it will return (-1.0,(-1,))
        
        Notes
        -----------
        The algorithm has bidirectional capability and also node cost added.
        """

        self.C_Reset_NodeVisited()
        
        cdef int cycles
        cycles = 1
        cdef bint keepGoing
        keepGoing = True

        cdef PriorityQueue_NR OpenNodes
        cdef NodeReach startNode
        startNode.Nid = NidO
        startNode.Eid = -1
        startNode.Dist = 0.0
        startNode.NidO = -1
        startNode.Weight = 0.0
        OpenNodes = PriorityQueue_NR()
        OpenNodes.push(startNode)

        cdef NodeReach NodeReach_T
        cdef int NidF
        cdef float len
        cdef Edge EdgeC
        cdef NodeReach NodeCheck
        cdef int i
        cdef int m
        cdef vector[pair[int, float]] EdgesFringe
        cdef vector[pair[int, float]] EdgesReach
        cdef pair[int, float] v
        cdef float edgeVector
        cdef float remainingDist
        cdef size_t Eididx
        cdef vector[int] EdgesReach_EidVec
        cdef int Eid
        cdef float lenE

        while keepGoing:
            cycles += 1
            # check paths from OpenNodes
            NodeCheck = OpenNodes.pop_top()
            for i in range(self.EidN):
                Eid = self.nodes[NodeCheck.Nid].Eid[i]
                if Eid == -1:
                    break
                if Eid == NodeCheck.Eid or self._edgesIds[Eid] == -1:
                    continue
                # possible new path
                # new node?
                EdgeC = self.edges[Eid]
                if EdgeC.NidO == NodeCheck.Nid:
                    lenE = EdgeC.len
                    NidF = EdgeC.NidD
                    edgeVector = 1.0
                else :
                    lenE = EdgeC.lenR
                    NidF = EdgeC.NidO
                    edgeVector = -1.0

                if self._nodesIds[NidF] == -1:
                    continue

                len = lenE + NodeCheck.Dist + self.nodes[NidF].c

                if len > LimDist:
                    remainingDist = (LimDist - NodeCheck.Dist) / lenE
                    if remainingDist > <float>1.0:
                        remainingDist = <float>1.0

                    # I'm sorry this section have so much brances and confusing variable names.
                    # theres just so many particular cases that cant be untangled
                    
                    remainingDist = remainingDist * edgeVector
                    EdgesReach_EidVec = EdgesReach_Eididx(EdgesFringe, Eid)
                    if EdgesReach_EidVec.size() == 0:
                        EdgesFringe.push_back(pair[int, float](Eid, remainingDist))
                    elif EdgesReach_EidVec.size() == 1:
                        Eididx = EdgesReach_EidVec[0]
                        if EdgesFringe[Eididx].second != 1.0 and EdgesFringe[Eididx].second != -1.0:
                            if remainingDist == -1.0 or remainingDist == 1.0:
                                EdgesFringe.erase(EdgesFringe.begin()+Eididx)
                                EdgesFringe.push_back(pair[int, float](Eid, remainingDist))
                            elif remainingDist > 0.0:
                                if EdgesFringe[Eididx].second > 0.0 and remainingDist > EdgesFringe[Eididx].second:
                                    EdgesFringe.erase(EdgesFringe.begin()+Eididx) # popping out shorter
                                    EdgesFringe.push_back(pair[int, float](Eid, remainingDist))
                                elif EdgesFringe[Eididx].second < 0.0:
                                    if (remainingDist - EdgesFringe[Eididx].second) > 1.0:
                                        EdgesFringe.erase(EdgesFringe.begin()+Eididx)
                                        EdgesFringe.push_back(pair[int, float](Eid, edgeVector))
                                    else:
                                        EdgesFringe.push_back(pair[int, float](Eid, remainingDist))
                            else:
                                if EdgesFringe[Eididx].second > 0.0:
                                    if (EdgesFringe[Eididx].second - remainingDist) > 1.0:
                                        EdgesFringe.erase(EdgesFringe.begin()+Eididx)
                                        EdgesFringe.push_back(pair[int, float](Eid, edgeVector))
                                    else:
                                        EdgesFringe.push_back(pair[int, float](Eid, remainingDist))
                                elif EdgesFringe[Eididx].second < 0.0 and remainingDist < EdgesFringe[Eididx].second:
                                    EdgesFringe.erase(EdgesFringe.begin()+Eididx)
                                    EdgesFringe.push_back(pair[int, float](Eid, remainingDist))
                    else: # elif EdgesReach_EidVec.size() == 2, should not be possible more than 2
                        if remainingDist == -1.0 or remainingDist == 1.0:
                            EdgesFringe.erase(EdgesFringe.begin()+EdgesReach_EidVec[0])
                            EdgesFringe.erase(EdgesFringe.begin()+EdgesReach_Eididx(EdgesFringe, Eid)[0])
                            EdgesFringe.push_back(pair[int, float](Eid, remainingDist))
                        elif edgeVector == 1.0:
                            if EdgesFringe[EdgesReach_EidVec[0]].second > 0.0:
                                if (remainingDist - EdgesFringe[EdgesReach_EidVec[1]].second) > 1.0:
                                    EdgesFringe.erase(EdgesFringe.begin()+EdgesReach_EidVec[0])
                                    EdgesFringe.erase(EdgesFringe.begin()+EdgesReach_Eididx(EdgesFringe, Eid)[0])
                                    EdgesFringe.push_back(pair[int, float](Eid, edgeVector))
                                elif remainingDist > EdgesFringe[EdgesReach_EidVec[0]].second:
                                    EdgesFringe.erase(EdgesFringe.begin()+ EdgesReach_EidVec[0])
                                    EdgesFringe.push_back(pair[int, float](Eid, remainingDist))
                            else:
                                if (remainingDist - EdgesFringe[EdgesReach_EidVec[0]].second) > 1.0:
                                    EdgesFringe.erase(EdgesFringe.begin()+EdgesReach_EidVec[0])
                                    EdgesFringe.erase(EdgesFringe.begin()+EdgesReach_Eididx(EdgesFringe, Eid)[0])
                                    EdgesFringe.push_back(pair[int, float](Eid, edgeVector))
                                elif remainingDist > EdgesFringe[EdgesReach_EidVec[1]].second:
                                    EdgesFringe.erase(EdgesFringe.begin()+ EdgesReach_EidVec[1])
                                    EdgesFringe.push_back(pair[int, float](Eid, remainingDist))
                        else: # edgeVector == -1.0
                            if EdgesFringe[EdgesReach_EidVec[0]].second > 0.0:
                                if (EdgesFringe[EdgesReach_EidVec[0]].second - remainingDist) > 1.0:
                                    EdgesFringe.erase(EdgesFringe.begin()+EdgesReach_EidVec[0])
                                    EdgesFringe.erase(EdgesFringe.begin()+EdgesReach_Eididx(EdgesFringe, Eid)[0])
                                    EdgesFringe.push_back(pair[int, float](Eid, edgeVector))
                                elif remainingDist < EdgesFringe[EdgesReach_EidVec[1]].second:
                                    EdgesFringe.erase(EdgesFringe.begin()+ EdgesReach_EidVec[1])
                                    EdgesFringe.push_back(pair[int, float](Eid, remainingDist))
                            else:
                                if (EdgesFringe[EdgesReach_EidVec[1]].second - remainingDist) > 1.0:
                                    EdgesFringe.erase(EdgesFringe.begin()+EdgesReach_EidVec[0])
                                    EdgesFringe.erase(EdgesFringe.begin()+EdgesReach_Eididx(EdgesFringe, Eid)[0])
                                    EdgesFringe.push_back(pair[int, float](Eid, edgeVector))
                                elif remainingDist < EdgesFringe[EdgesReach_EidVec[0]].second:
                                    EdgesFringe.erase(EdgesFringe.begin()+ EdgesReach_EidVec[0])
                                    EdgesFringe.push_back(pair[int, float](Eid, remainingDist))
                    continue

                NodeReach_T.Nid = NidF
                NodeReach_T.Dist = len
                NodeReach_T.Eid = Eid
                NodeReach_T.NidO = NodeCheck.Nid
                NodeReach_T.Weight = len

                # check to visited nodes
                if self.nodeVisited[NidF].Nid == -1 : # if node havent been visited
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)
                elif self.nodeVisited[NidF].Dist > len: # if visited node has a higher distance
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)

            if cycles > LimCycle:
                # if reaches cycle count limit
                keepGoing = False
            elif OpenNodes.empty():
                keepGoing = False
        
        cdef int EdgesReachSize = 0

        for i in range(self.Nnodes):
            if self.nodeVisited[i].Nid == -1:
                continue
            NodeReach_T = self.nodeVisited[i]

            if self.edges[NodeReach_T.Eid].NidO == NodeReach_T.Nid:
                edgeVector = 1.0
            else:
                edgeVector = -1.0
            EdgesReachSize += 1
            EdgesReach.push_back(pair[int, float](NodeReach_T.Eid, edgeVector))
        outtup = tuple(EdgesReach[n] for n in range(EdgesReach.size()))

        EdgesReach.insert(EdgesReach.end(), EdgesFringe.begin(), EdgesFringe.end())

        outtup = tuple((v.first, v.second) for v in EdgesReach)
        EdgesReach.clear()
        return outtup

    cdef void C_NodeMap_AStar(
                self, 
                const int& NidO, const int& NidD, 
                float DistMin = -1.0, 
                float DistMulLim = 1.2, 
                bint ReverseEdge = 0, 
                float LimDist = 10_000.0, 
                int LimCycle = 10_000, 
                float DistMul = 1.0, 
                float EdgeCmin = 0.9):
        """
        Mapout NodeReach on the minimum distance to every possible nodes.
        Using Astar principles.

        Parameters
        ------------
        NidO : int
            starting node ID
        NidD : int
            destination node ID
        DistMin : float, default -1.0
            Minimum distance between nodes if already found, if not the function will use PathFind_AStar to find the minimum distance.
        DistMulLim : float, default 1.2
            distance search limit multiplier for distance filter. i.e. if the minimum distance is 100m, it will propable nodes with propable paths up to 120m
        ReverseEdge : bint, default False
            on bidirectional system, using a reversed Len-LenR for pathfinding.
        LimDist : float, default 10,000.0
            distance limit before giving up the pathfinding process
        LimCycle : int, default 10,000
            number of cycles of priority_queue before giving up the pathfinding process
        DistMul : float, default 1.0
            A* weighting multiplier for the remaining cartesian distance
        EdgeCmin : float, default 0.9
            minimum edge cost multiplier, important if there are costs smaller than 1.0

        Returns
        -----------
        NodeReach*: array with size of self.Nnodes
            an array of mapped in range nodes with minimum reached distances with propable path within the distance limit
            from and to the origin-destination node.
        
        Notes
        -----------
        As a supporting function for multipath.
        """
        self.C_Reset_NodeVisited()

        # START
        cdef int cycles
        cycles = 1
        cdef bint keepGoing
        keepGoing = True
        cdef float BaseDist = dist3d(self.nodes[NidO], self.nodes[NidD])
        # cdef vector[NodeReach] OpenNodes
        cdef PriorityQueue_NR OpenNodes
        cdef NodeReach startNode
        startNode.Nid = NidO
        startNode.Eid = -1
        startNode.Dist = 0.0
        startNode.NidO = -1
        startNode.Weight = 0.0
        self.nodeVisited[NidO] = startNode
        OpenNodes = PriorityQueue_NR()
        OpenNodes.push(startNode)

        cdef NodeReach NodeReach_T
        cdef int NidF
        cdef float len
        cdef Edge EdgeC
        cdef NodeReach NodeCheck
        cdef Node NodeTarget
        NodeTarget = self.nodes[NidD]

        cdef vector[int] pth
        cdef int nlook
        cdef int i
        cdef float DistC_Target
        
        # check distmin
        if DistMin == -1.0:
            DistMin = self.PathDist_AStar(NidO, NidD, LimDist, LimCycle, DistMul)
            if DistMin == -1.0:
                return
        
        DistMin = DistMin * DistMulLim

        while keepGoing:
            cycles += 1
            # check paths from OpenNodes
            NodeCheck = OpenNodes.pop_top()

            for i in range(self.EidN):
                Eid = self.nodes[NodeCheck.Nid].Eid[i]
                if Eid == -1:
                    break
                if Eid == NodeCheck.Eid or self._edgesIds[Eid] == -1:
                    continue
                # possible new path
                # new node?
                EdgeC = self.edges[Eid]
                if not ReverseEdge:
                    if EdgeC.NidO == NodeCheck.Nid:
                        len = EdgeC.len
                        NidF = EdgeC.NidD
                    else :
                        len = EdgeC.lenR
                        NidF = EdgeC.NidO
                else:
                    if EdgeC.NidO == NodeCheck.Nid:
                        len = EdgeC.lenR
                        NidF = EdgeC.NidD
                    else :
                        len = EdgeC.len
                        NidF = EdgeC.NidO
                
                len += NodeCheck.Dist + self.nodes[NidF].c
                DistC_Target = dist3d(self.nodes[NidF], NodeTarget)

                if (len + DistC_Target*EdgeCmin) > DistMin: # check if there is still a propable remaining distance to target
                    continue

                NodeReach_T.Nid = NidF
                NodeReach_T.Dist = len
                NodeReach_T.Eid = Eid
                NodeReach_T.NidO = NodeCheck.Nid
                NodeReach_T.Weight = len + (DistC_Target - BaseDist) * DistMul

                # check to visited nodes
                if self.nodeVisited[NidF].Nid == -1 or self.nodeVisited[NidF].Dist > len: # if node havent been visited
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)
            if OpenNodes.empty():
                break
            elif cycles > LimCycle or OpenNodes.top().Dist > LimDist:
                # if reaches cycle count limit
                # if the minimum distance in open nodes is more than the distance limit
                break
        return

    cdef void C_NodeMap_AStar_VirtuEntry(
                self, 
                const int& EidO, const float[3]& PtO, const float[2]& DstO, 
                const int& EidD, const float[3]& PtD, const float[2]& DstD, 
                float DistMin = -1.0, 
                float DistMulLim = 1.2, 
                bint ReverseEdge = 0, 
                float LimDist = 10_000.0, 
                int LimCycle = 10_000, 
                float DistMul = 2.0, 
                float EdgeCmin = 0.9):
        """
        Mapout NodeReach on the minimum distance to every possible nodes.
        Using Astar principles.

        Parameters
        ------------
        NidO : int
            starting node ID
        NidD : int
            destination node ID
        DistMin : float, default -1.0
            Minimum distance between nodes if already found, if not the function will use PathFind_AStar to find the minimum distance.
        DistMulLim : float, default 1.2
            distance search limit multiplier for distance filter. i.e. if the minimum distance is 100m, it will propable nodes with propable paths up to 120m
        ReverseEdge : bint, default False
            on bidirectional system, using a reversed Len-LenR for pathfinding.
        LimDist : float, default 10,000.0
            distance limit before giving up the pathfinding process
        LimCycle : int, default 10,000
            number of cycles of priority_queue before giving up the pathfinding process
        DistMul : float, default 1.0
            A* weighting multiplier for the remaining cartesian distance
        EdgeCmin : float, default 0.9
            minimum edge cost multiplier, important if there are costs smaller than 1.0

        Returns
        -----------
        NodeReach*: array with size of self.Nnodes
            an array of mapped in range nodes with minimum reached distances with propable path within the distance limit
            from and to the origin-destination node.
        
        Notes
        -----------
        As a supporting function for multipath.
        """
        self.C_Reset_NodeVisited()

        # START
        cdef int cycles
        cycles = 1
        cdef bint keepGoing
        keepGoing = True

        # cdef vector[NodeReach] OpenNodes
        cdef float DistC_Target
        cdef PriorityQueue_NR OpenNodes
        cdef NodeReach startNode
        OpenNodes = PriorityQueue_NR()

        cdef float BaseDist = dist3d_ar(PtO, PtD)

        # for Oriding edgeorigin
        startNode.Nid = self.edges[EidO].NidO
        startNode.Eid = EidO
        startNode.Dist = DstO[0]
        startNode.NidO = -1
        startNode.Weight = DstO[0] + (dist3d_ar(self.nodes[startNode.Nid].pt, PtO)-BaseDist) * DistMul
        self.nodeVisited[startNode.Nid] = startNode
        OpenNodes.push(startNode)
        # for destination edgeorigin
        startNode.Nid = self.edges[EidO].NidD
        startNode.Eid = EidO
        startNode.Dist = DstO[1]
        startNode.NidO = -1
        startNode.Weight = DstO[1] + (dist3d_ar(self.nodes[startNode.Nid].pt, PtD)-BaseDist) * DistMul
        self.nodeVisited[startNode.Nid] = startNode
        OpenNodes.push(startNode)

        cdef NodeReach NodeReach_T
        cdef int NidF
        cdef float len
        cdef Edge EdgeC
        cdef NodeReach NodeCheck
        cdef vector[int] pth
        cdef int nlook
        cdef int i
        
        # check distmin
        if DistMin == -1.0:
            DistMin = self.PathDist_AStar_VirtuEntry(EidO, (PtO[0], PtO[1], PtO[2]), (DstO[0], DstO[1]), 
                                                    EidD, (PtD[0], PtD[1], PtD[2]), (DstD[0], DstD[1]),
                                                    LimDist, LimCycle, DistMul)
            if DistMin == -1.0:
                return 
        
        DistMin = DistMin * DistMulLim

        while keepGoing:
            cycles += 1
            # check paths from OpenNodes
            NodeCheck = OpenNodes.pop_top()

            for i in range(self.EidN):
                Eid = self.nodes[NodeCheck.Nid].Eid[i]
                if Eid == -1:
                    break
                if Eid == NodeCheck.Eid or self._edgesIds[Eid] == -1:
                    continue
                # possible new path
                # new node?
                EdgeC = self.edges[Eid]
                if not ReverseEdge:
                    if EdgeC.NidO == NodeCheck.Nid:
                        len = EdgeC.len
                        NidF = EdgeC.NidD
                    else :
                        len = EdgeC.lenR
                        NidF = EdgeC.NidO
                else:
                    if EdgeC.NidO == NodeCheck.Nid:
                        len = EdgeC.lenR
                        NidF = EdgeC.NidD
                    else :
                        len = EdgeC.len
                        NidF = EdgeC.NidO
                
                len += NodeCheck.Dist + self.nodes[NidF].c
                DistC_Target = dist3d_ar(self.nodes[NidF].pt, PtD)

                if (len + DistC_Target*EdgeCmin) > DistMin: # check if there is still a propable remaining distance to target
                    continue

                NodeReach_T.Nid = NidF
                NodeReach_T.Dist = len
                NodeReach_T.Eid = Eid
                NodeReach_T.NidO = NodeCheck.Nid
                NodeReach_T.Weight = len + (DistC_Target-BaseDist) * DistMul

                # check to visited nodes
                if self.nodeVisited[NidF].Nid == -1 or self.nodeVisited[NidF].Dist > len: # if node havent been visited
                    self.nodeVisited[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)
            if OpenNodes.empty():
                break
            elif cycles > LimCycle:
                break
        return

    cdef NodeReach* C_NodeMap_VirtuEntry(
                self, 
                const int& EidO, const float[3]& PtO, const float[2]& DstO,   
                bint ReverseEdge = 0, 
                float LimDist = 10_000.0, 
                int LimCycle = 10_000_000, 
                ):
        """
        Mapout NodeReach on the minimum distance to every possible nodes.
        Using Astar principles.

        Parameters
        ------------
        NidO : int
            starting node ID
        NidD : int
            destination node ID
        DistMin : float, default -1.0
            Minimum distance between nodes if already found, if not the function will use PathFind_AStar to find the minimum distance.
        DistMulLim : float, default 1.2
            distance search limit multiplier for distance filter. i.e. if the minimum distance is 100m, it will propable nodes with propable paths up to 120m
        ReverseEdge : bint, default False
            on bidirectional system, using a reversed Len-LenR for pathfinding.
        LimDist : float, default 10,000.0
            distance limit before giving up the pathfinding process
        LimCycle : int, default 10,000
            number of cycles of priority_queue before giving up the pathfinding process
        DistMul : float, default 1.0
            A* weighting multiplier for the remaining cartesian distance
        EdgeCmin : float, default 0.9
            minimum edge cost multiplier, important if there are costs smaller than 1.0

        Returns
        -----------
        NodeReach*: array with size of self.Nnodes
            an array of mapped in range nodes with minimum reached distances with propable path within the distance limit
            from and to the origin-destination node.
        
        Notes
        -----------
        As a supporting function for multipath.
        """
        cdef NodeReach* nodeMapped = <NodeReach*>malloc(self.Nnodes * sizeof(NodeReach))
        for n in range(self.Nnodes):
            nodeMapped[n].Nid = -1
            nodeMapped[n].Dist = 0.0
        # START
        cdef int cycles
        cycles = 1
        cdef bint keepGoing
        keepGoing = True

        cdef PriorityQueue_NR OpenNodes
        cdef NodeReach startNode
        OpenNodes = PriorityQueue_NR()

        # for Oriding edgeorigin
        startNode.Nid = self.edges[EidO].NidO
        startNode.Eid = EidO
        startNode.Dist = DstO[0]
        startNode.NidO = -1
        startNode.Weight = DstO[0]
        nodeMapped[startNode.Nid] = startNode
        OpenNodes.push(startNode)
        # for destination edgeorigin
        startNode.Nid = self.edges[EidO].NidD
        startNode.Eid = EidO
        startNode.Dist = DstO[1]
        startNode.NidO = -1
        startNode.Weight = DstO[1]
        nodeMapped[startNode.Nid] = startNode
        OpenNodes.push(startNode)

        cdef NodeReach NodeReach_T
        cdef int NidF
        cdef float len
        cdef Edge EdgeC
        cdef NodeReach NodeCheck
        cdef vector[int] pth
        cdef int nlook
        cdef int i

        while keepGoing:
            cycles += 1
            # check paths from OpenNodes
            NodeCheck = OpenNodes.pop_top()

            for i in range(self.EidN):
                Eid = self.nodes[NodeCheck.Nid].Eid[i]
                if Eid == -1:
                    break
                if Eid == NodeCheck.Eid or self._edgesIds[Eid] == -1:
                    continue
                # possible new path
                # new node?
                EdgeC = self.edges[Eid]
                if not ReverseEdge:
                    if EdgeC.NidO == NodeCheck.Nid:
                        len = EdgeC.len
                        NidF = EdgeC.NidD
                    else :
                        len = EdgeC.lenR
                        NidF = EdgeC.NidO
                else:
                    if EdgeC.NidO == NodeCheck.Nid:
                        len = EdgeC.lenR
                        NidF = EdgeC.NidD
                    else :
                        len = EdgeC.len
                        NidF = EdgeC.NidO
                
                len += NodeCheck.Dist + self.nodes[NidF].c

                if len > LimDist: # check if there is still a propable remaining distance to target
                    continue
                
                len += self.nodes[NidF].c

                NodeReach_T.Nid = NidF
                NodeReach_T.Dist = len
                NodeReach_T.Eid = Eid
                NodeReach_T.NidO = NodeCheck.Nid
                NodeReach_T.Weight = len

                # check to visited nodes
                if nodeMapped[NidF].Nid == -1 or nodeMapped[NidF].Dist > len: # if node havent been visited
                    nodeMapped[NidF] = NodeReach_T
                    OpenNodes.push(NodeReach_T)
            if OpenNodes.empty() or cycles > LimCycle:
                break
        return nodeMapped

    def PathFind_Multi(
            self, 
            const int& NidO, const int& NidD, 
            float DistMulLim = 1.1,
            float LimDist = 10_000.0,
            int LimCycle = 1_000_000_000,
            float DistMul = 2.0,
            float EdgeCmin = 0.9,
            int ForceOri = -1,
            int ForceDst = -1,
            int PathLim = 10_000_000,
            ) -> tuple[tuple[float, tuple[int]]] | None:
        """
        Find possible paths within

        Parameters
        ------------
        NidO : int
            starting node ID
        NidD : int
            destination node ID
        DistMulLim : float, default 1.1
            distance search limit multiplier for distance filter. i.e. if the minimum distance is 100m, it will propable nodes with propable paths up to 120m
        LimDist : float, default 10,000.0
            distance limit before giving up the pathfinding process
        LimCycle : int, default 10,000
            number of cycles of priority_queue before giving up the pathfinding process
        DistMul : float, default 1.0
            A* weighting multiplier for the remaining cartesian distance
        EdgeCmin : float, default 0.9
            minimum edge cost multiplier, important if there are costs smaller than 1.0
        
        Returns
        -----------
        set of alternative paths
        tuple of sets of distance and edgeids (213.8, (1, 2, 3, 4))
        
        Notes
        -----------
        Modified alternative paths algorithm, with an operation cost near O(2K+?)
        """
        # checking minimum distance
        cdef float BaseDist = dist3d(self.nodes[NidO], self.nodes[NidD])
        if BaseDist*EdgeCmin > LimDist:
            return None

        cdef float MinimumDistance = self.PathDist_AStar(NidO, NidD, LimDist, LimCycle, DistMul)
        if MinimumDistance == -1.0 or MinimumDistance > LimDist:
            return None
        
        # mapping pathcast
        cdef float LimitDistance = MinimumDistance * DistMulLim
        # cdef NodeReach* PathCast_To = self.C_NodeMap_AStar(NidD, NidO, MinimumDistance*1.5, DistMulLim, 1, LimDist, LimCycle, DistMul, EdgeCmin)
        self.C_NodeMap_AStar(NidD, NidO, LimitDistance, DistMulLim, 1, LimDist, LimCycle, DistMul, EdgeCmin)
        # collecting paths
        cdef vector[vector[int]] MappedPaths
        cdef vector[vector[int]] FoundPaths
        cdef vector[float] FoundDistance
        
        cdef bint keepGoing = True
        cdef NodeReach NodeCheck
        cdef PriorityQueue_NR OpenNodes
        cdef NodeReach startNode
        startNode.Nid = NidO
        startNode.Eid = -1
        startNode.Dist = 0.0
        startNode.NidO = -1
        startNode.Weight = 0.0
        startNode.pathindex = 0
        OpenNodes = PriorityQueue_NR()
        OpenNodes.push(startNode)
        
        cdef Node NodeTarget
        cdef NodeReach NodeReach_T
        cdef float pathlength
        cdef Edge EdgeC
        cdef vector[int] MappedPath_T
        cdef vector[int] MappedPath_T2
        cdef int cycles
        cdef int i
        
        cycles = 0
        MappedPath_T.push_back(-1)
        MappedPaths.push_back(MappedPath_T)
        NodeTarget = self.nodes[NidD]
        cdef float RemainDist

        while keepGoing:
            cycles += 1
            if OpenNodes.empty():
                keepGoing = False
                break
            NodeCheck = OpenNodes.pop_top()
            if NodeCheck.Dist > LimitDistance:
                break
            if NodeCheck.Nid == NidD:
                MappedPaths[NodeCheck.pathindex].erase(MappedPaths[NodeCheck.pathindex].begin()) # erases the first -1 value of path
                if ForceOri != -1:
                    MappedPaths[NodeCheck.pathindex][0] = ForceOri
                if ForceDst != -1:
                    i = <int>MappedPaths[NodeCheck.pathindex].size() - 1
                    MappedPaths[NodeCheck.pathindex][i] = ForceDst
                FoundPaths.push_back(MappedPaths[NodeCheck.pathindex])
                FoundDistance.push_back(NodeCheck.Dist)
                PathLim -= 1
                if PathLim == 0:
                    keepGoing = False
                    break
                continue
            
            MappedPath_T = MappedPaths[NodeCheck.pathindex] # getting mapped path from nodecheck index
            
            for i in range(self.EidN):
                Eid = self.nodes[NodeCheck.Nid].Eid[i]
                if Eid == -1:
                    break
                if Eid == NodeCheck.Eid or self._edgesIds[Eid] == -1:
                    continue
                if Find_IntVector(MappedPaths[NodeCheck.pathindex], Eid): # if Eid already in mapped paths
                    continue
                EdgeC = self.edges[Eid]
                if EdgeC.NidO == NodeCheck.Nid:
                    pathlength = EdgeC.len
                    NidF = EdgeC.NidD
                else:
                    pathlength = EdgeC.lenR
                    NidF = EdgeC.NidO

                pathlength += NodeCheck.Dist + self.nodes[NidF].c
                if self.nodeVisited[NidF].Nid != -1:
                    if (pathlength + self.nodeVisited[NidF].Dist) > LimitDistance:
                        continue
                    else:
                        NodeReach_T.Weight = pathlength + (self.nodeVisited[NidF].Dist-BaseDist) * DistMul
                else:
                    continue
                MappedPath_T2 = MappedPath_T
                MappedPath_T2.push_back(Eid)
                NodeReach_T.Nid = NidF
                NodeReach_T.Dist = pathlength
                NodeReach_T.Eid = Eid
                # NodeReach_T.NidO = NodeCheck.Nid
                
                if MappedPaths[NodeCheck.pathindex] == MappedPath_T:
                    NodeReach_T.pathindex = NodeCheck.pathindex
                    MappedPaths[NodeCheck.pathindex] = MappedPath_T2
                else:
                    NodeReach_T.pathindex = <int>MappedPaths.size()
                    MappedPaths.push_back(MappedPath_T2)
                OpenNodes.push(NodeReach_T)
            
            if cycles > LimCycle:
                break

        return tuple(FoundDistance), tuple(FoundPaths)
    
    def PathFind_Multi_VirtuEntry(
            self, 
            const int& EidO, tuple[float, float, float] PtO, tuple[float, float] DstO,
            const int& EidD, tuple[float, float, float] PtD, tuple[float, float] DstD,
            float DistMulLim = 1.1,
            float LimDist = 10_000.0,
            int LimCycle = 1_000_000_000,
            float DistMul = 1.0,
            float EdgeCmin = 0.9,
            int PathLim = 100_000,
            ) -> tuple[tuple[float, tuple[int]]] | None:
        # point coord, id, edge id, dist to edge
        """
        Find possible paths within

        Parameters
        ------------
        NidO : int
            starting node ID
        NidD : int
            destination node ID
        DistMulLim : float, default 1.1
            distance search limit multiplier for distance filter. i.e. if the minimum distance is 100m, it will propable nodes with propable paths up to 120m
        LimDist : float, default 10,000.0
            distance limit before giving up the pathfinding process
        LimCycle : int, default 10,000
            number of cycles of priority_queue before giving up the pathfinding process
        DistMul : float, default 1.0
            A* weighting multiplier for the remaining cartesian distance
        EdgeCmin : float, default 0.9
            minimum edge cost multiplier, important if there are costs smaller than 1.0
        
        Returns
        -----------
        set of alternative paths
        tuple of sets of distance and edgeids (213.8, (1, 2, 3, 4))
        
        Notes
        -----------
        Modified alternative paths algorithm, with an operation cost near O(2K+?)
        """
        cdef float MinimumDistance = self.PathDist_AStar_VirtuEntry(EidO, PtO, DstO, 
                                                                    EidD, PtD, DstD, 
                                                                    LimDist, LimCycle, DistMul)
        
        if MinimumDistance == -1.0 or MinimumDistance > LimDist:
            return None
        
        # mapping pathcast
        cdef float LimitDistance = MinimumDistance * DistMulLim
        cdef float[3] PointO = (PtO[0], PtO[1], PtO[2])
        cdef float[3] PointD = (PtD[0], PtD[1], PtD[2])
        cdef float[2] DistO = (DstO[0], DstO[1])
        cdef float[2] DistD = (DstD[0], DstD[1])
        cdef float BaseDist = dist3d_ar(PointO, PointD)
        self.C_NodeMap_AStar_VirtuEntry(
                            EidD, PointD, DistD,
                            EidO, PointO, DistO,
                            MinimumDistance, DistMulLim, 1, LimDist, LimCycle, DistMul, EdgeCmin)
        # collecting paths
        cdef vector[vector[int]] MappedPaths
        cdef vector[vector[int]] FoundPaths
        cdef vector[float] FoundDistance
        
        cdef bint keepGoing = True
        cdef NodeReach NodeCheck
        cdef PriorityQueue_NR OpenNodes
        cdef NodeReach startNode
        OpenNodes = PriorityQueue_NR()

        # for Origin EdgeOrigin
        if self.nodeVisited[self.edges[EidO].NidO].Nid != -1:
            startNode.Nid = self.edges[EidO].NidO
            startNode.Eid = EidO
            startNode.Dist = DstO[0]
            startNode.NidO = -1
            startNode.pathindex = 0
            startNode.Weight = DstO[0] + (self.nodeVisited[startNode.Nid].Dist - MinimumDistance) * DistMul
            OpenNodes.push(startNode)
        # for destination edgeorigin
        if self.nodeVisited[self.edges[EidO].NidD].Nid != -1:
            startNode.Nid = self.edges[EidO].NidD
            startNode.Eid = EidO
            startNode.Dist = DstO[1]
            startNode.NidO = -1
            startNode.pathindex = 0
            startNode.Weight = DstO[1] + (self.nodeVisited[startNode.Nid].Dist - MinimumDistance) * DistMul
            OpenNodes.push(startNode)
        # print(f'\tstartB {startNode.Dist} || {self.nodeVisited[startNode.Nid].Dist} || {LimitDistance}')
        cdef NodeReach NodeReach_T
        cdef float pathlength
        cdef Edge EdgeC
        cdef vector[int] MappedPath_T
        cdef vector[int] MappedPath_T2
        cdef int cycles
        cdef int i
        cdef int pathN = 0
        
        cycles = 0
        MappedPath_T.push_back(EidO)
        MappedPaths.push_back(MappedPath_T)
        # cdef float RemainDist
        while keepGoing:
            cycles += 1
            if OpenNodes.empty():
                break
            NodeCheck = OpenNodes.pop_top()
            MappedPath_T = MappedPaths[NodeCheck.pathindex] # getting mapped path from nodecheck index
            
            for i in range(self.EidN):
                Eid = self.nodes[NodeCheck.Nid].Eid[i]
                if Eid == -1:
                    # print(f'\t\tOut Edge {cycles} {Eid}')
                    break
                if Eid == NodeCheck.Eid or self._edgesIds[Eid] == -1:
                    # print(f'\t\tOut similar Edge {cycles} {Eid}')
                    continue

                if Find_IntVector(MappedPaths[NodeCheck.pathindex], Eid): # if Eid already in mapped paths
                    # print(f'\t\tOut Dupe {cycles}')
                    continue

                if Eid == EidD:
                    if NodeCheck.Nid == self.edges[EidD].NidO:
                        pathlength = NodeCheck.Dist + DstD[0]
                    else:
                        pathlength = NodeCheck.Dist + DstD[1]
                    if pathlength < LimitDistance:
                        FoundDistance.push_back(pathlength)
                        MappedPaths[NodeCheck.pathindex].push_back(Eid)
                        FoundPaths.push_back(MappedPaths[NodeCheck.pathindex])
                        pathN += 1
                        if PathLim <= pathN:
                            keepGoing = False
                            break
                    # else: print(f'\t\tOut didnt quite made it {cycles} {pathlength}')
                    continue

                EdgeC = self.edges[Eid]
                if EdgeC.NidO == NodeCheck.Nid:
                    pathlength = EdgeC.len
                    NidF = EdgeC.NidD
                else:
                    pathlength = EdgeC.lenR
                    NidF = EdgeC.NidO

                pathlength += NodeCheck.Dist + self.nodes[NidF].c
                if self.nodeVisited[NidF].Nid != -1:
                    if (pathlength + self.nodeVisited[NidF].Dist) > LimitDistance:
                        # print(f'\t\tOut {cycles} - {Eid} {pathlength} | {self.nodeVisited[NidF].Dist}')
                        continue
                    else:
                        NodeReach_T.Weight = pathlength + (self.nodeVisited[NidF].Dist-MinimumDistance)* DistMul
                else:
                    # print(f'\t\tOut {cycles} - Nid-1')
                    continue
                MappedPath_T2 = MappedPath_T
                MappedPath_T2.push_back(Eid)
                NodeReach_T.Nid = NidF
                NodeReach_T.Dist = pathlength
                NodeReach_T.Eid = Eid
                
                if MappedPaths[NodeCheck.pathindex] == MappedPath_T:
                    NodeReach_T.pathindex = NodeCheck.pathindex
                    MappedPaths[NodeCheck.pathindex] = MappedPath_T2
                else:
                    NodeReach_T.pathindex = <int>MappedPaths.size()
                    MappedPaths.push_back(MappedPath_T2)
                OpenNodes.push(NodeReach_T)
            
            if cycles > LimCycle:
                break
        # print(f'\t{EidO} to {EidD} - c{cycles} || p{pathN}')
        return tuple(FoundDistance), tuple(FoundPaths)


    def PathFind_Multi_MultiDest_VirtuEntry(
            self, 
            const int& EidO, tuple[float, float, float] PtO, tuple[float, float] DstO, int& Oid,
            tuple DestTup,
            float DistMulLim = 1.1,
            float LimDist = 10_000.0,
            int LimCycle = 1_000_000,
            float DistMul = 2.0,
            int PathLim = 100_000,
            ) -> tuple[tuple[float, tuple[int]]] | None:
        # point coord, id, edge id, dist to edge
        """
        Find possible paths within

        Parameters
        ------------
        NidO : int
            starting node ID
        NidD : int
            destination node ID
        DistMulLim : float, default 1.1
            distance search limit multiplier for distance filter. i.e. if the minimum distance is 100m, it will propable nodes with propable paths up to 120m
        LimDist : float, default 10,000.0
            distance limit before giving up the pathfinding process
        LimCycle : int, default 10,000
            number of cycles of priority_queue before giving up the pathfinding process
        DistMul : float, default 1.0
            A* weighting multiplier for the remaining cartesian distance
        EdgeCmin : float, default 0.9
            minimum edge cost multiplier, important if there are costs smaller than 1.0
        
        Returns
        -----------
        set of alternative paths
        tuple of sets of distance and edgeids (213.8, (1, 2, 3, 4))
        
        Notes
        -----------
        Modified alternative paths algorithm, with an operation cost near O(2K+?)
        """

        # initial map
        cdef int EidD
        cdef float LimitDistance
        cdef float[3] PointO = (PtO[0], PtO[1], PtO[2])
        cdef float[3] PointD
        cdef float[2] DistO = (DstO[0], DstO[1])
        cdef float[2] DistD
        cdef float BaseDist
        cdef float MinimumDistance
        cdef NodeReach* nodeMapped = self.C_NodeMap_VirtuEntry(
                                                            EidO, PointO, DistO,
                                                            0, LimDist*DistMulLim, LimCycle*2)

        # collecting paths
        cdef vector[vector[int]] MappedPaths
        cdef vector[vector[int]] FoundPaths
        cdef vector[float] FoundDistance
        cdef vector[float] FoundWeights
        
        cdef bint keepGoing
        cdef NodeReach NodeCheck
        cdef PriorityQueue_NR OpenNodes = PriorityQueue_NR()
        cdef NodeReach startNode

        cdef NodeReach NodeReach_T
        cdef float pathlength
        cdef Edge EdgeC
        cdef vector[int] MappedPath_T
        cdef vector[int] MappedPath_T2
        cdef int cycles
        cdef int i
        cdef int pathN
        cdef int tempC = 0
        # cycles per destination
        for dest in DestTup:
            tempC += 1
            if Oid == dest[4]:
                continue

            MappedPaths.clear()
            OpenNodes.clear()
            MappedPath_T.clear()
            MappedPath_T2.clear()
            keepGoing = True
            pathN = 0

            EidD = dest[0]
            PointD = (dest[1][0], dest[1][1], dest[1][2])
            DistD = (dest[2][0], dest[2][1])
            BaseDist = dist3d_ar(PointO, PointD)
            if BaseDist > LimDist:
                continue
            cycles = 0
            MappedPath_T.push_back(EidD)
            MappedPaths.push_back(MappedPath_T)

            if EidD == EidO: # checks if same edge
                MinimumDistance = abs(DistD[0] - DistO[0])
                FoundPaths.push_back(MappedPath_T)
                if MinimumDistance == 0.0:
                    MinimumDistance = <float>0.1
                # print(f'\t\tSameLine {MinimumDistance}')
                FoundDistance.push_back(MinimumDistance)
                FoundWeights.push_back(dest[3])
                pathN += 1
            else:
                MinimumDistance = self.PathDist_AStar_VirtuEntry(EidO, PtO, DstO, 
                                                                EidD, dest[1], dest[2], 
                                                                LimDist, LimCycle, DistMul)
            # print(dest, MinimumDistance)
            if MinimumDistance == -1.0 or MinimumDistance > LimDist:
                continue
            LimitDistance = MinimumDistance * DistMulLim
            # for Origin EdgeOrigin
            if nodeMapped[self.edges[EidD].NidO].Nid != -1:
                startNode.Nid = self.edges[EidD].NidO
                startNode.Eid = EidD
                startNode.Dist = DistD[0]
                startNode.NidO = -1
                startNode.pathindex = 0
                startNode.Weight = DistD[0] + (nodeMapped[startNode.Nid].Dist - MinimumDistance) * DistMul
                OpenNodes.push(startNode)
            # print(f'\tstartA {startNode.Dist} || {nodeMapped[startNode.Nid].Dist} || {LimitDistance}')
            # for destination edgeorigin
            if nodeMapped[self.edges[EidD].NidD].Nid != -1:
                startNode.Nid = self.edges[EidD].NidD
                startNode.Eid = EidD
                startNode.Dist = DistD[1]
                startNode.NidO = -1
                startNode.pathindex = 0
                startNode.Weight = DistD[1] + (nodeMapped[startNode.Nid].Dist - MinimumDistance) * DistMul
                OpenNodes.push(startNode)
            # print(f'\tstartB {startNode.Dist} || {nodeMapped[startNode.Nid].Dist} || {LimitDistance}')
            # cdef float RemainDist
            while keepGoing:
                cycles += 1
                if OpenNodes.empty():
                    break
                NodeCheck = OpenNodes.pop_top()
                MappedPath_T = MappedPaths[NodeCheck.pathindex] # getting mapped path from nodecheck index
                # print(f'\tC - {cycles} || {NodeCheck}')
                for i in range(self.EidN):
                    Eid = self.nodes[NodeCheck.Nid].Eid[i]
                    if Eid == -1:
                        # print(f'\t\t{cycles} - Out Edge{Eid}')
                        break
                    if Eid == NodeCheck.Eid or self._edgesIds[Eid] == -1:
                        # print(f'\t\t{cycles} - Out similar Edge{Eid}')
                        continue
                    if Find_IntVector(MappedPaths[NodeCheck.pathindex], Eid): # if Eid already in mapped paths
                        continue

                    if Eid == EidO:
                        if NodeCheck.Nid == self.edges[EidO].NidO:
                            pathlength = NodeCheck.Dist + DistO[0]
                        else:
                            pathlength = NodeCheck.Dist + DistO[1]
                        if pathlength < LimitDistance:
                            FoundDistance.push_back(pathlength)
                            MappedPaths[NodeCheck.pathindex].push_back(Eid)
                            FoundPaths.push_back(MappedPaths[NodeCheck.pathindex])
                            FoundWeights.push_back(dest[3])
                            pathN += 1
                            if PathLim <= pathN:
                                keepGoing = False
                                break
                        continue

                    EdgeC = self.edges[Eid]
                    if EdgeC.NidO == NodeCheck.Nid:
                        pathlength = EdgeC.lenR
                        NidF = EdgeC.NidD
                    else:
                        pathlength = EdgeC.len
                        NidF = EdgeC.NidO

                    pathlength += NodeCheck.Dist + self.nodes[NidF].c
                    if nodeMapped[NidF].Nid != -1:
                        if (pathlength + nodeMapped[NidF].Dist) > LimitDistance:
                            # print(f'\t\t{cycles} - Out{Eid} {pathlength} | {nodeMapped[NidF].Dist}')
                            continue
                        else:
                            NodeReach_T.Weight = pathlength + (nodeMapped[NidF].Dist-MinimumDistance) * DistMul
                    else:
                        # print(f'\t\t{cycles} - Out  Nid-1 {NidF}')
                        continue
                    MappedPath_T2 = MappedPath_T
                    MappedPath_T2.push_back(Eid)
                    NodeReach_T.Nid = NidF
                    NodeReach_T.Dist = pathlength
                    NodeReach_T.Eid = Eid
                    
                    if MappedPaths[NodeCheck.pathindex] == MappedPath_T:
                        NodeReach_T.pathindex = NodeCheck.pathindex
                        MappedPaths[NodeCheck.pathindex] = MappedPath_T2
                    else:
                        NodeReach_T.pathindex = <int>MappedPaths.size()
                        MappedPaths.push_back(MappedPath_T2)
                    OpenNodes.push(NodeReach_T)
                
                if cycles > LimCycle:
                    break
            # print(f'\t{EidO} to {EidD} - c{cycles} || p{pathN}')
        free(nodeMapped)
        return tuple(FoundDistance), tuple(FoundPaths), tuple(FoundWeights)
