class Graph:
    def __init__(self,no_of_vertices):
        self.V = no_of_vertices
        self.adj = []
        
    def addEdge(self,u,v,w):
        self.adj.append([u,v,w])
        
    def isDistinctWeight(self,self.adj):
        weight = []
        for i in range(len(self.adj)):
            u,v,w = self.adj[i]
            weight.append(w)
        if( len(set(weight)) == len(weight)):
            return True
        else
            return False
        
    def Boruvka(self):

        no_of_components = self.V #Initialy the no of components is V
        component = []
        
        for i in range(no_of_components):
            component.append(i)
        cheapestEdge = [-1] * no_of_components
        
        while no_of_components > 1:

            for i in range(len(self.adj)):
                u,v,w = self.adj[i]
                if(component[u] != component[v]):
                    
