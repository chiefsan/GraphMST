import math
import random

import matplotlib.pyplot as plt
import networkx as nx


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
                string += "%d -> %d == %d\n" % (head, tail, weight)
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
                self.incident_vertices[head].remove(tail)
        if head in self.adjacency:
            if tail in self.adjacency[head]:
                del self.adjacency[head][tail]
                self.incident_vertices[tail].remove(head)

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

        mst_vertices = set()
        key = [math.inf] * graph.num_vertices
        parent = [None] * graph.num_vertices
        graph_vertices = graph.get_vertices()

        # choosing a random vertex to start (should write a generalized one using graph_vertices)
        start_vertex = random.randrange(0, graph.num_vertices)
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

            for tail in graph.incident_vertices[head]:
                if tail not in mst_vertices and graph.adjacency[head][tail] < key[tail]:
                    key[tail] = graph.adjacency[head][tail]
                    parent[tail] = head

        mst_edges = []

        for tail in range(graph.num_vertices):
            if parent[tail] != -1:
                head = parent[tail]
                edge = [head, tail, graph.adjacency[head][tail]]
                mst_edges.append(edge)

        mst = Graph.build(edges=mst_edges)
        return mst

    def Kruskal(self):

        MST = []
        self.adj = sorted(self.adj, key=lambda item: item[2])

        parent = []
        rank = []

        for i in range(self.V):
            parent.append(i)
            rank.append(0)

        j = 0
        e = 0

        while e < self.V - 1:

            u, v, w = self.adj[j]
            j += 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            if x != y:
                e = e + 1
                MST.append([u, v, w])
                self.union(parent, rank, x, y)

        for u, v, w in MST:
            print("%d -> %d == %d" % (u, v, w))

    def kruskal_mst(graph):
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
        while index < num_vertices:
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

    def Boruvka(self):
        if(self.isDistinctWeight):
            parent = []
            rank = []

            cheapEdge = []

            no_of_components = self.V

            for node in range(self.V):
                parent.append(node)
                rank.append(0)
                cheapEdge = [-1] * self.V
            nStep = 1

            while no_of_components > 1 and nStep <= 2:

                for i in range(len(self.adj)):
                    u, v, w = self.adj[i]
                    set1 = self.find(parent, u)
                    set2 = self.find(parent, v)

                    if set1 != set2:

                        if cheapEdge[set1] == -1 or cheapEdge[set1][2] > w:
                            cheapEdge[set1] = [u, v, w]

                        if cheapEdge[set2] == -1 or cheapEdge[set2][2] > w:
                            cheapEdge[set2] = [u, v, w]

                for node in range(self.V):
                    if cheapEdge[node] != -1:
                        u, v, w = cheapEdge[node]
                        set1 = self.find(parent, u)
                        set2 = self.find(parent, v)

                        if set1 != set2:
                            self.union(parent, rank, set1, set2)
                            print("%d -> %d == %d" % (u, v, w))
                            no_of_components = no_of_components - 1

                cheapEdge = [-1] * self.V
                nStep += 1
        else:
            # Tie breaker algorithm to be added
            print("The weight should be distinct")


g = Graph()
g = Graph.build([0, 1, 2, 3], [[0, 1, 1], [0, 2, 1],
                               [0, 3, 1], [1, 2, 1], [2, 3, 1]])
g.distinct_weight()
# print(g.get_edges())
# print(g.adjacency)
kg = Graph.kruskal_mst(g)
# print(str(g))
print(kg)
pg = Graph.prims_mst(g)
print(pg)
