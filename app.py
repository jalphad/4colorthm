import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum


def import_graphs():
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
            graph = nx.Graph()
            graphs.append(graph)
            state = State.VERTEX
        elif state is State.VERTEX:
            config = line.split()
            nr_of_vertices = int(config[0])
            graph.add_nodes_from(range(1, nr_of_vertices + 1))
            state = State.EDGE
        elif state is State.EDGE:
            config = line.split()
            if line == '0':
                pass
            elif int(config[0]) <= nr_of_vertices:
                vertex = int(config[0])
                for i in range(2, 2 + int(config[1])):
                    graph.add_edge(vertex, int(config[i]))
            else:
                pass

    return graphs


graph_arr = import_graphs()


def get_special_k(graph, colors):
    coloring = {}
    cur_node = list(graph.nodes)[0]
    coloring[cur_node] = colors[0]

    for neighneigh in graph.neighbors(cur_node):
        new_coloring = coloring.copy()
        new_coloring = get_special_k_recur(graph, neighneigh, colors, new_coloring)
        #Turn it into list for NX.draw
        if new_coloring is not None:
            colorMap = []
            for node in graph.nodes:
                colorMap.append(new_coloring[node])
            return colorMap
    return None


def get_special_k_recur(graph, node, colors: list, coloring: dict):
    # If already colored, return current (successful) coloring
    if node in coloring.keys():
        return coloring

    neighbours = list(graph.neighbors(node))
    avail = colors.copy()

    # Remove colors already taken by neighbours
    for neighneigh in neighbours:
        if neighneigh in coloring.keys():
            if coloring[neighneigh] in avail:
                avail.remove(coloring[neighneigh])

    for node_color in avail:
        new_coloring = coloring.copy()
        new_coloring[node] = node_color
        for neighneigh in neighbours:
            # Colour the other neighbours
            new_coloring = get_special_k_recur(graph, neighneigh, colors, new_coloring)
            if new_coloring is None:
                break
        if new_coloring is not None:
            return new_coloring

    return None


colors = ["blue", "red", "green", "yellow"]


networkx.draw(graphs[0])
plt.show()