import networkx
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
import sys
from joblib import Parallel, delayed


class Config:
    identifier = None
    graph = None
    size = None
    ring_size = None
    no_of_colorings = None
    ring = None
    inside = None
    pass


def import_graphs():
    class State(Enum):
        NEW = 1
        VERTEX = 2
        EDGE = 3

    # with open("vier-kleuren.txt", 'r') as f: # For the specific example
    with open("U_2822.conf", 'r') as f:
        data = f.readlines()

    graphs = []
    configs = []

    state = State.NEW

    for line in data:
        line = line.strip()
        if state is State.EDGE and line == '':
            cur_config.inside = cur_graph.subgraph(
                filter(lambda v: v > cur_config.ring_size, cur_graph.nodes)
            )
            cur_config.ring = cur_graph.subgraph(
                filter(lambda v: v <= cur_config.ring_size, cur_graph.nodes)
            )

            state = State.NEW
            cur_graph = None
            nr_of_vertices = -1
        elif state is State.NEW:
            cur_graph = nx.Graph()
            graphs.append(cur_graph)

            cur_config = Config()
            cur_config.identifier = int(line.split()[0])
            configs.append(cur_config)

            state = State.VERTEX
        elif state is State.VERTEX:
            config_data = line.split()
            nr_of_vertices = int(config_data[0])
            cur_graph.add_nodes_from(range(1, nr_of_vertices + 1))

            cur_config.graph = cur_graph
            cur_config.size = int(config_data[0])
            cur_config.ring_size = int(config_data[1])
            cur_config.no_of_colorings = int(config_data[2])

            state = State.EDGE
        elif state is State.EDGE:
            config_data = line.split()
            if line == '0':
                pass
            elif int(config_data[0]) <= nr_of_vertices:
                vertex = int(config_data[0])
                for i in range(2, 2 + int(config_data[1])):
                    cur_graph.add_edge(vertex, int(config_data[i]))
            else:
                pass

    return graphs, configs


def get_special_k(graph, colors, color_dic: dict = {}, start_index=0):
    def get_special_k_recur(graph, node, colors: list, coloring: dict):
        # If already colored, return current (successful) coloring
        if node in coloring.keys():
            return coloring

        use_sort = True
        if not use_sort:
            neighbours = list(graph.neighbors(node))
        else:
            neighbours_unsort = list(graph.neighbors(node))
            neighbour_deg = list(graph.degree)
            neighbours = [i for _, i in sorted(zip(neighbour_deg, neighbours_unsort), reverse=True)]
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

    # color_dic: Dictionary with a colour keyed by each node
    color_dic = get_special_k_recur(graph, list(graph.nodes)[start_index], colors, color_dic)

    # Check if it was successful
    if color_dic is None:
        return None

    # Turn it into list for NX.draw
    color_list = []
    for node in graph.nodes:
        color_list.append(color_dic[node])
    return color_list, color_dic # color_list of the colors corresponding to vertices by order, coloring is dictionary


def ggd_test_service(graph, color_list):
    for e in graph.edges:
        if color_list[e[0] - 1] == color_list[e[1] - 1]:  # Node numbers start with 1, indexing starts with 0
            return False
    return True


def special_k_to_the_ggd(g, i: int):
    print(f"{i + 1}/2822")  # Hardcoded for parallelness
    # sys.stdout.flush()
    k, k_vertex = get_special_k(g, color_set)
    if k is None or not ggd_test_service(g, k):
        print(f'WEEEEUUUUEEEEUUUUU NO COLOR IN MY LIFE: {i}')
        nx.draw(g)
        plt.title(f"i={i}")
        plt.show()


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
    coloring1 = list(map(lambda i: d[i], coloring1))
    d = {ni: indi for indi, ni in enumerate(set(coloring2))}
    coloring2 = list(map(lambda i: d[i], coloring2))

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
    for clr_i in range(len(coloring1)):
        # If they do not match up, we have to permutatenate coloring 2
        if coloring1[clr_i] != coloring2[clr_i]:
            # Check if the to be replaced color has already been fixed to the permutation of coloring 1 before
            if coloring2[clr_i] in avail_colors:
                # Check if the replacement is already fixed to the specific permutation of coloring 1
                if coloring1[clr_i] in avail_colors:
                    avail_colors.remove(coloring1[clr_i])
                    coloring2 = list(map(switcher_hitcher, coloring2, coloring1[clr_i], coloring2[clr_i]))
                else:
                    return False
            else:
                return False
        # If they do match up fixate the color so it cant be switched
        elif coloring2[clr_i] in avail_colors:
            avail_colors.remove(coloring2[clr_i])
    return True


graph_arr, config_arr = import_graphs()
color_set = {"blue", "red", "green", "yellow"}



Parallel(n_jobs=8)(delayed(special_k_to_the_ggd)(graph_arr[i], i) for i in range(len(graph_arr)))

# for i in range(len(graph_arr)): deal_and_check(i)

sel = 2000

nx.draw(config_arr[sel].graph)
plt.figure()
nx.draw(config_arr[sel].inside)
plt.figure()
nx.draw(config_arr[sel].ring)
plt.figure()

nx.draw_planar(graph_arr[sel],
               node_color=get_special_k(graph_arr[sel], color_set)[0],
               with_labels=list(graph_arr[sel].nodes))


print(coloring_is_isomorphism(
    get_special_k(graph_arr[sel], {'r', 'b', 'g', 'y'})[0],
    get_special_k(graph_arr[sel], {1, 2, 3, 4})[0]
))

plt.show()
