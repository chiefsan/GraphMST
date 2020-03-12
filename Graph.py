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

