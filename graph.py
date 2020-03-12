import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self,no_of_vertices):
        self.V = no_of_vertices
        self.adj = []
        
    def addEdge(self,u,v,w):
        self.adj.append([u,v,w])
        
    def isDistinctWeight(self):
        weight = []
        for i in range(len(self.adj)):
            u,v,w = self.adj[i]
            weight.append(w)
        if( len(set(weight)) == len(weight)):
            return True
        else:
            return False
        
    def myGraph(self):
        for i in range(len(self.adj)):
            u,v,w = self.adj[i]
            print ("%d -> %d == %d" % (u,v,w))

    def myGraphViz(self):
        G = nx.Graph()
        for i in range(len(self.adj)):
            u,v,w = self.adj[i]
            G.add_edge(u,v,weight=w)
        nx.draw(G, with_labels = True)
        plt.show()
        
    def find(self, parent, i): 
        if parent[i] == i: 
            return i
        else:
            return self.find(parent, parent[i]) 
  
    def union(self, parent,rank,vertex1,vertex2):
        
        root1 = self.find(parent,vertex1)
        root2 = self.find(parent,vertex2)
       
        if rank[root1] < rank[root2]: 
            parent[root1] = root2 
        elif rank[root1] > rank[root2]: 
            parent[root2] = root1
        else : 
            parent[root2] = root1
            rank[root1] = rank[root1]+1
            
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
     
                            
  

