import math
import random
from collections import defaultdict
# import matplotlib.pyplot as plt
# import networkx as nx


class Graph:
    '''
    Data structure to store graphs (based on adjacency lists)
    Data Members
    ------------
    num_vertices : int, number of vertices
    num_edges : int, number of edges
    adjacency : dict, {tail: {head: weight}} where   (head, tail, weight)
        is an edge in the graph
    incident_vertices : {head: {tails}} where (head, tail, weight)
        is an edge in the graph
    '''

    def __init__(self):

        # Set number of edges and vertices to 0 since the initial graph is
        # empty (adjacency list is empty as well) and everything is added
        # only after initialization
        self.num_vertices = 0
        self.num_edges = 0

        self.adjacency = {}
        self.block_size = None
        self.block_cnt = None
        self.first_visit = None
        self.euler_tour = None
        self.height = None
        self.log_2 = None
        self.st = None
        self.blocks = None
        self.block_mask = None


    def add_vertex(self, vertex):
        '''
        Adds a vertex to the graph
        Parameters
        ----------
        vertex : hashable object
        '''
        if vertex not in self.adjacency:
            self.adjacency[vertex] = {}
            self.num_vertices += 1

    def add_edge(self, head, tail, weight):
        '''
        Adds an edge to the graph
        Parameters
        ----------
        head : vertex
        tail : vertex
        weight : float, weight of the egde from head to tail
        '''
        # Add the vertices to the graph (if they haven't already been added)
        self.add_vertex(head)
        self.add_vertex(tail)

        # Self edge => invalid
        if head == tail:
            return

        # Since graph is undirected, add both edge and reverse edge
        self.adjacency[head][tail] = weight

        self.adjacency[tail][head] = weight

    def distinct_weight(self):
        edges = self.get_edges()
        for edge in edges:
            head, tail, weight = edge
            edges.remove((tail, head, weight))
        for i in range(len(edges)):
            edges[i] = list(edges[i])

        edges.sort(key=lambda e: e[2])
        for i in range(len(edges)-1):
            if edges[i][2] >= edges[i+1][2]:
                edges[i+1][2] = edges[i][2]+1
        for edge in edges:
            head, tail, weight = edge
            self.adjacency[head][tail] = weight
            self.adjacency[tail][head] = weight

    def __str__(self):
        '''
        Returns string representation of the graph
        '''
        string = ''
        for tail in self.adjacency:
            for head in self.adjacency[tail]:
                weight = self.adjacency[head][tail]
                #string += "%d -> %d == %d\n" % (head, tail, weight)
                string += str(head)+' -> '+str(tail)+' == '+str(weight)+'\n'
        return string

    def get_edges(self):
        '''
        Returna all edges in the graph
        '''
        output = []
        for tail in self.adjacency:
            for head in self.adjacency[tail]:
                output.append((tail, head, self.adjacency[head][tail]))
        return output

    def get_vertices(self):
        '''
        Returns all vertices in the graph
        '''
        return self.adjacency.keys()

    def adjacent(self, tail, head):
        '''
        Returns True if there is an edge between head and tail,
            False otherwise
        Parameters
        ---------
        tail : vertex
        head : vertex
        '''
        if tail in self.adjacency:
            if head in self.adjacency[tail]:
                return True
        return False

    def neighbours(self, vertex):
        '''
        Returns a list of all tails such that there is an
            edge between vertex and tail
        Parameters
        ---------
        vertex : vertex
        '''
        if vertex not in self.adjacency:
            return []
        else:
            return self.adjacency[vertex].keys()

    def remove_edge(self, tail, head):
        '''
        Removes an edge from the graphs
        Parameters
        ---------
        tail : vertex
        head : vertex
        '''
        if tail in self.adjacency:
            if head in self.adjacency[tail]:
                del self.adjacency[tail][head]

        if head in self.adjacency:
            if tail in self.adjacency[head]:
                del self.adjacency[head][tail]

    def remove_vertex(self, vertex):
        '''
        Removes a vertex (and edges incident on the vertex) from the graph
        Parameters
        ---------
        vertex : vertex
        '''
        if vertex not in self.adjacency:
            return

        # Delete edges incident on the vertex
        for tail in self.adjacency[vertex]:
            del self.adjacency[tail][vertex]
            del self.adjacency[vertex][tail]

        del self.adjacency[vertex]

    @staticmethod
    def build(vertices=[], edges=[]):
        '''
        Builds a graph from the given set of vertices and edges
        Parameters
        ----------
        vertices : list of vertices where each element is a vertex
        edges : list of edges where each edge is a list [head, tail, weight]
        '''
        g = Graph()
        for vertex in vertices:
            g.add_vertex(vertex)
        for edge in edges:
            g.add_edge(*edge)
        return g

    def myGraphViz(self):
        '''
        Visualizes the graph using networkx
        '''
        G = nx.Graph()
        for head in self.adjacency:
            for tail in self.adjacency[vertex]:
                head, tail, weight = self.adjacency[head][tail]
                G.add_edge(head, tail, weight=weight)
        nx.draw(G, with_labels=True)
        plt.show()

    class UnionFind(object):
        '''
        '''

        def __init__(self):
            self.parent = {}
            self.rank = {}

        def __len__(self):
            return len(self.parent)

        def make_set(self, item):
            if item in self.parent:
                return self.find(item)

            self.parent[item] = item
            self.rank[item] = 0
            return item

        def find(self, item):
            if item not in self.parent:
                return self.make_set(item)
            if item != self.parent[item]:
                self.parent[item] = self.find(self.parent[item])
            return self.parent[item]

        def union(self, item1, item2):
            root1 = self.find(item1)
            root2 = self.find(item2)

            if root1 == root2:
                return root1

            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
                return root1

            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
                return root2

            if self.rank[root1] == self.rank[root2]:
                self.rank[root1] += 1
                self.parent[root2] = root1
                return root1

    def prims_mst(graph):
        '''
        Implementation of Prim's algorithm
        Time Complexity: 
        '''
        mst_vertices = set()
        key = defaultdict(lambda: math.inf)
        parent = defaultdict(lambda: None)

        graph_vertices = list(graph.get_vertices())

        start_vertex = random.choice(graph_vertices)
        key[start_vertex] = 0
        parent[start_vertex] = -1

        while len(mst_vertices) != graph.num_vertices:

            min = math.inf
            for vertex in graph_vertices:
                if key[vertex] < min and vertex not in mst_vertices:
                    min = key[vertex]
                    min_key_vertex = vertex

            mst_vertices.add(min_key_vertex)
            head = min_key_vertex

            for tail in graph.adjacency[head]:
                if tail not in mst_vertices and graph.adjacency[head][tail] < key[tail]:
                    key[tail] = graph.adjacency[head][tail]
                    parent[tail] = head

        mst_edges = []

        for tail in graph_vertices:
            if parent[tail] != -1:
                head = parent[tail]
                edge = [head, tail, graph.adjacency[head][tail]]
                mst_edges.append(edge)

        mst = Graph.build(edges=mst_edges)
        return mst

    def kruskal_mst(graph):
        '''
        Implementation of Kruskal's algorithm
        Time Complexity: 
        '''
        mst_edges = []
        edges = graph.get_edges()
        num_vertices = len(graph.get_vertices())

        edges = graph.get_edges()
        for edge in edges:
            head, tail, weight = edge
            edges.remove((tail, head, weight))
        edges.sort(key=lambda e: e[2])

        union_find = Graph.UnionFind()

        index = 0
            edge = edges[index]
            [tail, head, value] = edge
            index += 1

            if union_find.find(head) == union_find.find(tail):
                continue
            else:
                union_find.union(head, tail)
            mst_edges.append(edge)

        mst = Graph.build(edges=mst_edges)
        return mst

    def boruvka_step(graph, num_components, union_find, mst_edges):
        '''
        Implementation of Boruvka step
        Time Complexity: 
        '''
        cheapest_edge = {}
        for vertex in graph.get_vertices():
            cheapest_edge[vertex] = -1

        edges = graph.get_edges()
        for edge in edges:
            head, tail, weight = edge
            edges.remove((tail, head, weight))
        for edge in edges:
            head, tail, weight = edge
            set1 = union_find.find(head)
            set2 = union_find.find(tail)
            if set1 != set2:
                if cheapest_edge[set1] == -1 or cheapest_edge[set1][2] > weight:
                    cheapest_edge[set1] = [head, tail, weight]

                if cheapest_edge[set2] == -1 or cheapest_edge[set2][2] > weight:
                    cheapest_edge[set2] = [head, tail, weight]
        for vertex in cheapest_edge:
            if cheapest_edge[vertex] != -1:
                head, tail, weight = cheapest_edge[vertex]
                if union_find.find(head) != union_find.find(tail):
                    union_find.union(head, tail)
                    mst_edges.append(cheapest_edge[vertex])
                    num_components = num_components - 1
        return num_components, union_find, mst_edges

    def boruvka_mst(graph):
        '''
        Implementation of Boruvka's algorithm
        Time Complexity: 
        '''
        num_components = graph.num_vertices

        union_find = Graph.UnionFind()
        mst_edges = []
        while num_components > 1:
            num_components, union_find, mst_edges = Graph.boruvka_step(graph, num_components, union_find, mst_edges)

        mst = Graph.build(edges=mst_edges)
        return mst

    class BranchingNode:
        def __init__(self, index, level):
            self.index = index
            self.level = 0
            
    
    def kkt_mst(graph, union_find=None, first_call=False):
        '''
        Implementation of KKT Algorithm
        Time Complexity:
        '''
        if first_call:
            union_find = Graph.UnionFind()
            num_components = graph.num_vertices
            mst_edges = []
            
        if graph.num_edges > 1:
            num_steps_remaining = 2
            while graph.num_edges > 1 and num_steps_remaining > 0:
                num_components, union_find, mst_edges = Graph.boruvka_step(graph, num_components, union_find, mst_edges)
                num_steps_remaining -= 1
            edges = graph.get_edges()
            graph3 = Graph()
            for edge in edges:
                head, tail, weight = edge
                set1 = union_find.find(head)
                set2 = union_find.find(tail)
                if set1 != set2:
                    random_num = random.random()
                    if random_num > 0.5:
                        graph3.add_edge(head, tail, weight)
            num_components, union_find, graph4 = kkt_mst(graph3, union_find, False)
            graph3_copy = copy(graph3)
            edges = graph3.get_edges()
            vertices = graph3.get_vertices()

            component_supervertex = defaultdict(bool)
            
            child = [-1]*graph3.num_vertices
            sibling = [-1]*graph3.num_vertices
            parent = [-1]*graph3.num_vertices
            rightmostchild = [-1]*graph3.num_vertices
                        

            branchingtree=Graph()
            indmap = {}
            ind=0
            for vertex in vertices:
                branchingtree.add_vertex(vertex)
                indmap[vertex]=ind
                ind+=1

            num_components3 = graph.num_vertices
            union_find3 = Graph.UnionFind()
            mst_edges3 = []
            workdone=True
            level = 0
            while num_components3 > 1  and workdone:
                cheapest_edge = {}
                workdone=False
                for vertex in vertices:
                    cheapest_edge[vertex] = -1
                for edge in edges:
                    head, tail, weight = edge
                    edges.remove((tail, head, weight))
                for edge in edges:
                    head, tail, weight = edge
                    set1 = union_find3.find(head)
                    set2 = union_find3.find(tail)
                    if set1 != set2:
                        workdone=True
                        if cheapest_edge[set1] == -1 or cheapest_edge[set1][2] > weight:
                            cheapest_edge[set1] = [head, tail, weight]

                        if cheapest_edge[set2] == -1 or cheapest_edge[set2][2] > weight:
                            cheapest_edge[set2] = [head, tail, weight]
                
                for vertex in cheapest_edge:
                    if cheapest_edge[vertex] != -1:
                        workdone=True
                        head, tail, weight = cheapest_edge[vertex]
                        set1 = union_find3.find(head)
                        set2 = union_find3.find(tail)
                        if set1 != set2:
                            union_find3.union(head, tail)
                            mst_edges3.append(cheapest_edge[vertex])
                            num_components3 = num_components3 - 1
                            supervertex = min(str(set1), str(set2))+max(str(set1), str(set2)) + str(level)
                            if supervertex not in indmap:
                                ind = branchingtree.num_vertices
                                indmap[supervertex] = ind
                                parent.append(-1)
                                parent[indmap[head]] = parent[indmap[tail]] = ind
                                child.append(indmap[head])
                                sibling[indmap[head]] = indmap[tail]
                                rightmost.append(indmap[tail])
                                branchingtree.add_vertex(supervertex)
                                branchingtree.add_edge(head, supervertex, WEIGHT)
                                branchingtree.add_edge(tail, supervertex, WEIGHT)
                            else:
                                ind = indmap[supervertex]
                                if parent[indmap[head]]!=ind:
                                    sibling[rightmostchild[ind]] = indmap[head]
                                    parent[indmap[head]]=ind
                                    branchingtree.add_edge(head, supervertex, WEIGHT)
                                else:
                                    sibling[rightmostchild[ind]] = indmap[tail]
                                    parent[indmap[tail]]=ind
                                    branchingtree.add_edge(tail, supervertex, WEIGHT)
                level+=1
            for vertex in vertices:
                if parent[indmap[vertex]]==-1:
                    branchingtree.add_edge('super_root', vertex, 1)
            branchingtree.precomputeLCA('super root')
            mst = Graph.build(edges=mst_edges)
            
            for edge in edges:
                head, tail, weight = edge
                lca_head_tail = graph3.lca(head, tail)

        else:
            edges = graph.get_edges()
            for edge in edges:
                head, tail, weight = edge
                union_find.union(head, tail)
            return union_find, graph

    def dfs(self, v, p, h):
        # print self.first_visit, v, self.euler_tour
        self.first_visit[v] = len(self.euler_tour)
        self.euler_tour.append(v)
        self.height[v] = h
        for u in self.adjacency[v]:
            # print v, p, h, u
            if u == p:
                continue
            self.dfs(u, v, h + 1)
            self.euler_tour.append(v)

    def min_by_h(self, i, j):
        return i if self.height[self.euler_tour[i]] < self.height[self.euler_tour[j]] else j

    def precompute_lca(self, root):

        # reserves from c++ can be omitted when we use python;
        # capacity is different from size as the latter counts actual items;
        # single-argument vector constructor adds that many actual items with unspecified value

        n = self.num_vertices

        # get euler tour and indices of first occurrences
        self.first_visit = defaultdict(lambda: 0)
        for i in range(n):
            self.first_visit[i] = -1
        self.height = defaultdict(lambda: 0)
        for i in range(n):
            self.height[i] = 0
        self.euler_tour = []
        self.dfs(root, -1, 0)

        # pre-compute all log values
        m = len(self.euler_tour)
        self.log_2 = []
        self.log_2.append(-1)
        for i in range(1, m + 1):
            self.log_2.append(self.log_2[i // 2] + 1)
        self.block_size = max(1, self.log_2[m] // 2)
        self.block_cnt = (m + self.block_size - 1) // self.block_size

        # pre-compute min. of each block and build sparse table;
        # integer vectors automatically initialize using zero
        self.st = []
        for i in range(self.block_cnt):
            self.st.append([0] * (self.log_2[self.block_cnt] + 1))
        b = 0
        j = 0
        for i in range(m):
            if (j == self.block_size):
                j = 0
                b += 1
            if (j == 0 or self.min_by_h(i, self.st[b][0]) == i):
                self.st[b][0] = i
            j += 1
        for l in range(1, self.log_2[self.block_cnt] + 1):
            for i in range(self.block_cnt):
                ni = i + (1 << (l - 1))
                if (ni >= self.block_cnt):
                    self.st[i][l] = self.st[i][l - 1]
                else:
                    # print self.st, i, l
                    self.st[i][l] = self.min_by_h(
                        self.st[i][l - 1], self.st[ni][l - 1])

        # pre-compute mask for each block
        self.block_mask = [0] * self.block_cnt
        b = 0
        j = 0
        for i in range(0, m):
            if (j == self.block_size):
                j = 0
                b += 1
            if (j > 0 and (i >= m or self.min_by_h(i - 1, i) == i - 1)):
                self.block_mask[b] += 1 << (j - 1)
            j += 1

        # pre-compute RMQ for each unique block
        # possibilities = 1 << (self.block_size - 1)
        self.blocks = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        for b in range(0, self.block_cnt):
            mask = self.block_mask[b]
            # print self.blocks, mask, b
            if len(self.blocks[mask]) != 0:
                continue
            # print self.blocks[mask]
            # self.blocks[mask] = []
            for i in range(self.block_size):
                curr_value = [0] * self.block_size
                # print self.blocks[0]
                self.blocks[mask][i] = curr_value
            for l in range(self.block_size):
                self.blocks[mask][l][l] = l
                for r in range(l + 1, self.block_size):
                    self.blocks[mask][l][r] = self.blocks[mask][l][r - 1]
                    if (b * self.block_size + r < m):
                        self.blocks[mask][l][r] = self.min_by_h(b * self.block_size + self.blocks[mask][l][r],
                                                                b * self.block_size + r) - b * self.block_size

    def lca_in_block(self, b, l, r):
        return self.blocks[self.block_mask[b]][l][r] + b * self.block_size

    def lca(self, v, u):
        l = self.first_visit[v]
        r = self.first_visit[u]
        if (l > r):
            v1 = l
            v2 = r
            l = v2
            r = v1
        bl = l // self.block_size
        br = r // self.block_size
        if (bl == br):
            return self.euler_tour[self.lca_in_block(bl, l % self.block_size, r % self.block_size)]
        ans1 = self.lca_in_block(bl, l % self.block_size, self.block_size - 1)
        ans2 = self.lca_in_block(br, 0, r % self.block_size)
        ans = self.min_by_h(ans1, ans2)
        if (bl + 1 < br):
            l = self.log_2[br - bl - 1]
            ans3 = self.st[bl + 1][l]
            ans4 = self.st[br - (1 << l)][l]
            ans = self.min_by_h(ans, self.min_by_h(ans3, ans4))
        return self.euler_tour[ans]
    


g = Graph()
# g = Graph.build([0, 1, 2, 3], [[0, 1, 1], [0, 2, 1],
#                           [0, 3, 1], [1, 2, 1], [2, 3, 1]])
g = Graph.build([0, 1, 2, 3, 4, 5], [[0, 1, 1], [0, 2, 1], [3, 4, 1], [3, 5, 1], [-1,0,1], [-1,3,1]])
# g = Graph.build(['a', 'b', 'c', 'd'], [['a', 'b', 1], ['a', 'c', 1],
#                                        ['a', 'd', 1], ['b', 'c', 1], ['c', 'd', 1]])
g.distinct_weight()


# kg = Graph.kruskal_mst(g)
# print(kg)
# pg = Graph.prims_mst(g)
# print(pg)
# bg = Graph.boruvka_mst(g)
# print(bg)

g.precompute_lca(-1)
print(g.lca(1,3))
