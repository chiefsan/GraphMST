import Graph

def Boruvka():

    no_of_components = self.V #Initialy the no of components is V
    component = []
    nStep = 1
    for i in range(no_of_components):
        component.append(i)
    cheapestEdge = [-1] * no_of_components
    
    while no_of_components > 1 and nStep <=2:
        if(nStep>1):
            
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
                        continue
                    
g = Graph.Graph(4) 
g.addEdge(0, 1, 10) 
g.addEdge(0, 2, 6) 
g.addEdge(0, 3, 5) 
g.addEdge(1, 3, 15) 
g.addEdge(2, 3, 4)
#g.connectedComponents()
                    
