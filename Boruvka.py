class Graph:
    def __init__(self,no_of_vertices):
        self.V = no_of_vertices
        self.adj = []
        
    def addEdge(self,u,v,w):
        self.adj.append([u,v,w])
        
    def isDistinctWeight(self,adj):
        weight = []
        for i in range(len(self.adj)):
            u,v,w = self.adj[i]
            weight.append(w)
        if( len(set(weight)) == len(weight)):
            return True
        else:
            return False
    def DFSUtil(self, temp, v, visited): 
  
        # Mark the current vertex as visited 
        visited[v] = True
  
        # Store the vertex to list 
        temp.append(v) 
  
        # Repeat for all vertices adjacent 
        # to this vertex v 
        for i in self.adj[v]: 
            if visited[i] == False: 
                  
                # Update the list 
                temp = self.DFSUtil(temp, i, visited) 
        return temp 

    def connectedComponents(self): 
        visited = [] 
        cc = [] 
        for i in range(self.V): 
            visited.append(False) 
        for v in range(self.V): 
            if visited[v] == False: 
                temp = [] 
                cc.append(self.DFSUtil(temp, v, visited)) 
        return cc 


    def Boruvka(self):

        no_of_components = self.V #Initialy the no of components is V
        component = []
        nStep = 1
        for i in range(no_of_components):
            component.append(i)
        cheapestEdge = [-1] * no_of_components
        
        while no_of_components > 1 and nStep <=2:
	    if(nStep>1:
                
		components = self.connectedComponents()
            for i in range(len(self.adj)):
                u,v,w = self.adj[i]
                if(component[u] != component[v]):
                    if(w<cheapestEdge[u]):
                        cheapest[u] = [u,v,w]
                    if(w<cheapestEdge[v]):
                            cheapest[v] = [u,v,w]

                    for i in range(len(self.V)):
                        if (cheapest[i] != -1):        
                    
g = Graph(4) 
g.addEdge(0, 1, 10) 
g.addEdge(0, 2, 6) 
g.addEdge(0, 3, 5) 
g.addEdge(1, 3, 15) 
g.addEdge(2, 3, 4)
g.connectedComponents()
                    
