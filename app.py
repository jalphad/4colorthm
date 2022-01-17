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

networkx.draw(graphs[0])
plt.show()