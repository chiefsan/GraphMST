# import networkx as nx
# import matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        self.num_vertices = 0
        self.num_edges = 0
        
        # {tail: {head: value}}
        self.adjacency = {}

        # {head: {tails}}
        self.incident_vertices = {}
    
    def add_vertex(self, vertex):
        if vertex not in self.adjacency:
            self.adjacency[vertex] = {}
            self.num_vertices += 1
        if vertex not in self.incident_vertices:
            self.incident_vertices[vertex] = set()
        
    def add_edge(self,head,tail,weight):
        self.add_vertex(head)
        self.add_vertex(tail)

        # Self edge => invalid
        if head==tail:
            return

        self.adjacency[head][tail] = weight
        self.incident_vertices[tail].add(head)

        self.adjacency[tail][head] = weight
        self.incident_vertices[head].add(tail)
        
    def distinct_weight(self):
        weights = []
        for tail in self.adjacency:
            for head in self.adjacency[vertex]:
                head, tail, weight = self.adjacency[head][tail]
            weights.append(weight)
        
        if( len(set(weights)) == len(weights)):
            return True
        else:
            return False
        
    def __str__(self):
        string = ''
        for tail in self.adjacency:
            for head in self.adjacency[tail]:
                weight = self.adjacency[head][tail]
                string += "%d -> %d == %d\n" % (head, tail, weight)
        return string

    def get_edges(self):
        output = []
        for tail in self.adjacency:
            for head in self.adjacency[tail]:
                output.append((tail, head, self.adjacency[head][tail]))
        return output
    
    def get_vertices(self):
        return self.adjacency.keys()

    def adjacent(self, tail, head):
        if tail in self.adjacency:
            if head in self.adjacency[tail]:
                return True
        return False

    def neighbours(self, vertex):
        if vertex not in self.adjacency:
            return []
        else:
            return self.adjacency[vertex].keys()

    def incident(self, vertex):
        if vertex not in self.incident_vertices:
            return []
        return list(self.incident_vertices[vertex])

    def remove_edge(self, tail, head):
        if tail in self.adjacency:
            if head in self.adjacency[tail]:
                del self.adjacency[tail][head]
                self.incident_vertices[head].remove(tail)
        if head in self.adjacency:
            if tail in self.adjacency[head]:
                del self.adjacency[head][tail]
                self.incident_vertices[tail].remove(head)

    def remove_vertex(self, vertex):
        if vertex not in self.adjacency:
            return

        if vertex in self.values:
            del self.values[vertex]

        for head in self.adjacency[vertex]:
            self.incident_vertices[head].remove(vertex)

        for tail in self.incident_vertices[vertex]:
            del self.adjacency[tail][vertex]

        del self.adjacency[vertex]
        del self.incident_vertices[vertex]

    @staticmethod
    def build(vertices=[], edges=[]):
        g = Graph()
        for vertex in vertices:
            g.add_vertex(vertex)
        for edge in edges:
            g.add_edge(*edge)
        return g

    def myGraphViz(self):
        G = nx.Graph()
        for head in self.adjacency:
            for tail in self.adjacency[vertex]:
                head, tail, weight = self.adjacency[head][tail]
                G.add_edge(head, tail, weight = weight)
        nx.draw(G, with_labels = True)
        plt.show()

    class UnionFind(object):
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
            
    def Kruskal(self): 

        MST =[] 
        self.adj =  sorted(self.adj,key=lambda item: item[2])   

        parent = []
        rank = [] 

        for i in range(self.V):
            parent.append(i)
            rank.append(0)

        j=0
        e=0
        
        while e < self.V -1 : 

            u,v,w =  self.adj[j] 
            j+=1
            x = self.find(parent, u) 
            y = self.find(parent ,v) 

            if x != y: 
                e = e + 1     
                MST.append([u,v,w]) 
                self.union(parent, rank, x, y)             
            
        for u,v,w in MST: 
            print ("%d -> %d == %d" % (u,v,w)) 

    def kruskal_mst(graph):
        mst_edges = []
        edges = graph.get_edges()
        num_vertices = len(graph.get_vertices())

        edges = graph.get_edges()
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
      
            cheapEdge =[] 
      
            no_of_components = self.V 

            for node in range(self.V): 
                parent.append(node) 
                rank.append(0) 
                cheapEdge =[-1] * self.V 
            nStep = 1
      
            while no_of_components > 1 and nStep <=2: 
      
                for i in range(len(self.adj)):  
                    u,v,w =  self.adj[i] 
                    set1 = self.find(parent, u) 
                    set2 = self.find(parent ,v) 
      
                    if set1 != set2:      
                          
                        if cheapEdge[set1] == -1 or cheapEdge[set1][2] > w : 
                            cheapEdge[set1] = [u,v,w]  
      
                        if cheapEdge[set2] == -1 or cheapEdge[set2][2] > w : 
                            cheapEdge[set2] = [u,v,w] 
      

                for node in range(self.V): 
                    if cheapEdge[node] != -1: 
                        u,v,w = cheapEdge[node] 
                        set1 = self.find(parent, u) 
                        set2 = self.find(parent ,v) 
      
                        if set1 != set2 : 
                            self.union(parent, rank, set1, set2)
                            print ("%d -> %d == %d" % (u,v,w))
                            no_of_components = no_of_components - 1
                   
                cheapEdge =[-1] * self.V 
                nStep += 1
        else:
            print("The weight should be distinct")    #Tie breaker algorithm to be added
     
                            
  

g = Graph()
g = Graph.build([1,2,3,4], [[1,2,1], [1,3,1], [1,4,1], [2,3,1], [3,4,1], [2,4,1]])
# print(g.adjacency)
kg = Graph.kruskal_mst(g)
print(str(g))

print(kg)
