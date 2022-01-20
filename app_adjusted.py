import networkx
import matplotlib.pyplot as plt
from enum import Enum

class State(Enum):
   NEW = 1
   VERTEX = 2
   EDGE = 3

with open("U_2822.conf", 'r') as f:
   data = f.readlines()

graphs = []
state = State.NEW

for line in data:
   line = line.strip()
   if state is State.EDGE and line == '':
     state = State.NEW
     graph = None
     nr_of_vertices = -1
   elif state is State.NEW:
     graph = networkx.Graph()
     graphs.append(graph)
     state = State.VERTEX
   elif state is State.VERTEX:
     config = line.split()
     nr_of_vertices = int(config[0])
     graph.add_nodes_from(range(1,nr_of_vertices+1))
     state = State.EDGE
   elif state is State.EDGE:
     config = line.split()
     if line == '0':
       pass
     elif int(config[0]) <= nr_of_vertices:
       vertex = int(config[0])
       for i in range(2,2 + int(config[1])):
         graph.add_edge(vertex, int(config[i]))
     else:
       pass
next

##### check
def checkColoring(graph, c, v, i, colored):
    if i == v:

        if validColoring(graph, colored):
            global colored_map
            colored_map = colored
            print(colored_map)
            return True
        return False

    for j in range(1, c + 1):
        colored[i] = j

        if checkColoring(graph, c, v, i + 1, colored):
            return True
        colored[i] = 1
    return False

def validColoring(graph, colored):
    for u in graph:
        for v in graph.neighbors(u):
            if colored[u] == colored[v]:
                return False
    return True

def produceColoring(colored,colors):
    color_map = []
    for i in range(len(colored)):
        index_color = colored[i+1]
        color = colors[index_color]
        color_map.append(color)
    return color_map

G = graphs[4]
n = len(G)
colored_map = {}
colors = {1:'blue',2:'green',3:'red',4:'yellow'}
for node in G:
    colored_map[node]=1
print(checkColoring(G, 4, n, 1, colored_map))
color_map = produceColoring(colored_map,colors)
print(color_map)
networkx.draw(G, node_color=color_map, with_labels=True)
plt.show()
