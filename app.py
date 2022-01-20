import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
import sys


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


def get_special_k(graph, colors):
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

    coloring = {}  # Dictionary with a colour keyed by each node
    coloring = get_special_k_recur(graph, list(graph.nodes)[0], colors, coloring)

    # Check if it was successful
    if coloring is None:
        return None

    # Turn it into list for NX.draw
    color_map = []
    for node in graph.nodes:
        color_map.append(coloring[node])
    return color_map


def ggd_test_service(graph, colouring):
    for e in graph.edges:
        if colouring[e[0]-1] == colouring[e[1]-1]:  # Node numbers start with 1, indexing starts with 0
            return False
    return True


# Checks if coloring is isomorphism, GIVEN that they color the SAME graph (not an isomorphic graph necessarily)
def coloring_is_isomorphism(coloring1: list, coloring2: list):
    # Check same length
    if len(coloring1) != len(coloring2):
        return False

    # Check same amount of colours
    if len(set(coloring1)) != len(set(coloring2)):
        return False

    # Map different colour domains into number identifiers
    d = {ni: indi for indi, ni in enumerate(set(coloring1))}
    coloring1 = map(d.get, coloring1)
    coloring2 = map(d.get, coloring2)

    # Map that switches two colours
    def switcher_hitcher(c, color1, color2):
        if c == color1:
            return color2
        elif c == color2:
            return color1
        else:
            return c

    avail_colors = set(coloring1)

    # Try to switch colours of 2 so that coloring 2 becomes equal to 1
    for clr_i in range(coloring1):
        # If they do not match up, we have to permutatenate coloring 2
        if coloring1[clr_i] != coloring2[clr_i]:
            # Check if the to be replaced color has already been fixed to the permutation of coloring 1 before
            if coloring2[clr_i] in avail_colors:
                # Check if the replacement is already fixed to the specific permutation of coloring 1
                if coloring1[clr_i] in avail_colors:
                    avail_colors.remove(coloring1[clr_i])
                    coloring2 = map(switcher_hitcher, coloring2, coloring1[clr_i], coloring2[clr_i])
                else:
                    return False
            else:
                return False
        # If they do match up fixate the color so it cant be switched
        elif coloring2 in avail_colors:
            avail_colors.remove(coloring2[clr_i])
    return True


graph_arr = import_graphs()
color_set = ["blue", "red", "green", "yellow"]

for i in range(len(graph_arr)):
    print(f"{i + 1}/{len(graph_arr)}")
    sys.stdout.flush()
    k = get_special_k(graph_arr[i], color_set)
    if k is None or not ggd_test_service(graph_arr[i], k):
        print(f'WEEEEUUUUEEEEUUUUU NO COLOR IN MY LIFE: {i}')
        nx.draw(graph_arr[i])
        plt.title(f"i={i}")
        plt.show()

sel = 300
nx.draw(graph_arr[sel], node_color=get_special_k(graph_arr[sel], color_set), with_labels=list(graph_arr[sel].nodes))
plt.show()
