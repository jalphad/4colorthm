import sys
from dataclasses import is_dataclass
from unittest import result
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
from copy import deepcopy
from itertools import combinations
from joblib import Parallel, delayed
import timeit

# Global constants
COLORS = ("blue", "red", "green", "yellow")
COLOR_PAIRINGS = (
    (("blue", "red"), ("green", "yellow")),
    (("blue", "green"), ("red", "yellow")),
    (("blue", "yellow"), ("red", "green"))
)


class Config:
    identifier = None
    graph = None
    size = None
    ring_size = None
    no_of_colorings = None
    ring = None
    inside = None


class Grouping:

    class Group:
        def __init__(self, id: int, grouping_size: int) -> None:
            self.id = id
            self.sectors = []
            self.connection_count = {}
            for i in range(grouping_size):
                if i != self.id:
                    self.connection_count[i] = 0


        def add(self, sector: int):
            self.sectors.append(sector)


        def update_count(self, group: int, action: str):
            if action == "add":
                self.connection_count[group] += 1
            elif action == "remove":
                self.connection_count[group] -= 1
            else:
                raise Exception("No valid action")


    def __init__(self, kempe_sectors: list, coloring: dict, color_pairing: tuple, size: int) -> None:
        self.size = size
        self.sectors = kempe_sectors
        self.coloring = coloring
        self.color_pairing = color_pairing
        self.groups = []
        for i in range(size):
            self.groups.append(self.Group(i, size))
        self.sectors_to_group = {}
        self.add_sector_to_group(0,0)


    def get_neighboring_groups(self, sector: int):
        neighbors = []
        if (sector-1)%len(self.sectors) in self.sectors_to_group.keys():
            neighbors.append(self.sectors_to_group.get((sector-1)%len(self.sectors)))
        if (sector+1)%len(self.sectors) in self.sectors_to_group.keys():
            neighbors.append(self.sectors_to_group.get((sector+1)%len(self.sectors)))
        return neighbors


    def get_available_groups(self, sector: int):
        in_use = self.get_neighboring_groups(sector)
        return [i for i in range(self.size) if i not in in_use]


    def add_sector_to_group(self, sector: int, group: int):
        self.sectors_to_group[sector] = group
        self.groups[group].add(sector)
        for n in self.get_neighboring_groups(sector):
            self.groups[group].update_count(n, "add")
            self.groups[n].update_count(group, "add")


    def remove_last_sector_from_group(self):
        sector,group = self.sectors_to_group.popitem()
        for n in self.get_neighboring_groups(sector):
            self.groups[group].update_count(n, "remove")
            self.groups[n].update_count(group, "remove")
        self.groups[group].sectors.remove(sector)


    def is_valid(self):
        def is_same_type():
            for group in self.groups:
                if not len(group.sectors) == 0:
                    nodes = []
                    for sector in group.sectors:
                        nodes.extend(self.sectors[sector])
                    if self.coloring[nodes[0]] in self.color_pairing[0]:
                        pair = self.color_pairing[0]
                    else:
                        pair = self.color_pairing[1]
                    for node in nodes:
                        if not self.coloring[node] in pair:
                            return False
            return True


        def has_no_mutual_overlap():
            for removed in self.groups:
                for group in self.groups:
                    if group.id == removed.id:
                        continue
                    if len(group.sectors) <= 1:
                        continue
                    reached = [group.sectors[0]]
                    for i in range(group.sectors[0]+1, group.sectors[-1]+1):
                        if i in removed.sectors:
                            break
                        if i in group.sectors:
                            reached.append(i)
                    if len(reached) == len(group.sectors):
                        continue
                    for i in range(group.sectors[0]-1, group.sectors[1]-len(self.sectors)-1, -1):
                        if i%len(self.sectors) in removed.sectors:
                            break
                        if i%len(self.sectors) in group.sectors:
                            reached.append(i)
                    if len(reached) != len(group.sectors):
                        return False
            return True


        def is_properly_connected():
            for group in self.groups:
                for val in group.connection_count.values():
                    if val == 0 or val == 2:
                        next
                    else:
                        return False
            return True

        return is_same_type() and has_no_mutual_overlap() and is_properly_connected()


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


def verify_all_ring_colorings(config: Config):
    def recurse(config, node: int, coloring: dict):
        if node > config.ring_size:
            result = check_reducible(config, node, coloring)
            return [(coloring.copy(), result)]
            # return True
        graph = config.ring
        result = []

        global_in_use = set(coloring.values())
        global_free = list(filter(lambda c: c not in global_in_use, COLORS))
        colors_available = list(global_in_use) + global_free[0:1]
        colors_in_use = [coloring[i] for i in list(graph.neighbors(node)) if i in coloring.keys()]
        colors_available = list(filter(lambda c: c not in colors_in_use, colors_available))

        while len(colors_available) > 0:
            if len(coloring) > node:
                coloring.popitem()  # CTRL Z on the coloring, let's try the other possibility
            coloring[node] = colors_available[0]
            colors_available = colors_available[1::]
            result += recurse(config, node + 1, coloring)
        return result

    coloring = {}
    return recurse(config, 1, coloring)


def check_reducible(config: Config, node: int, coloring: dict):
    k, _ = get_special_k(config.graph, COLORS, coloring, node)
    if k is None or not ggd_test_service(config.graph, k):
        for pairing in COLOR_PAIRINGS:
            kempe_sectors = compute_kempe_sectors(coloring, pairing)
            # 1 or 2 Kempe sectors will not result in a ring coloring which extends to the interior of the graph
            if len(kempe_sectors) < 3:
                continue
            elif len(kempe_sectors)%2 == 1:
                continue
            elif verify_all_sector_groupings(config, coloring, kempe_sectors, pairing):
                return True
            else:
                continue
        return False
    return True


def compute_kempe_sectors(coloring: dict, pairing: tuple):
    sectors = []
    previous = -1
    for k,v in coloring.items():
        if v in pairing[0]:
            if previous == 0:
                sectors[-1].append(k)
            else:
                sectors.append([k])
            previous = 0
        elif v in pairing[1]:
            if previous == 1:
                sectors[-1].append(k)
            else:
                sectors.append([k])
            previous = 1
        else:
            raise Exception("Color not in any pair!")
    return sectors


def verify_all_sector_groupings(config: Config, coloring: dict, kempe_sectors: list, color_pairing: tuple):
    def recurse(grouping: Grouping, sector: int, groupings: list):
        empty_groups = grouping.size - len(set(grouping.sectors_to_group.values()))
        sectors_left = len(grouping.sectors) - sector
        if empty_groups > sectors_left:
            grouping.remove_last_sector_from_group()
            return
        if sector >= len(grouping.sectors):
            if len(set(grouping.sectors_to_group.values())) == grouping.size and grouping.is_valid():
                groupings.append(deepcopy(grouping))
            grouping.remove_last_sector_from_group()
            return
        available = grouping.get_available_groups(sector)
        global_in_use = {g for g in grouping.sectors_to_group.values()}
        global_free = [g for g in range(grouping.size) if g not in global_in_use]
        if len(global_free) == len(available):
            available = available[0:1]
        else:
            for g in global_free[1::]:
                available.remove(g)
        for group in available:
            grouping.add_sector_to_group(sector, group)
            recurse(grouping, sector+1, groupings)
        grouping.remove_last_sector_from_group()


    valid_groupings = []

    # While I haven't looked at a proof for this it seems that:
    # - an uneven amount of kempe sectors will never have a valid grouping
    # - for n kempe sectors where n is even, will only have valid groupings of size n/2 + 1
    groups = len(kempe_sectors)//2 + 1
    grouping = Grouping(kempe_sectors, coloring, color_pairing, groups)
    recurse(grouping, 1, valid_groupings)

    # for max_groups in range(3, len(kempe_sectors)):
    #     grouping = Grouping(kempe_sectors, coloring, color_pairing, max_groups)
    #     recurse(grouping, 1, valid_groupings)

    if len(valid_groupings) == 0:
        return False

    print(f"{len(valid_groupings)} valid groupings found")
    for grouping in valid_groupings:
        result = do_color_switching(config, coloring, kempe_sectors, grouping, color_pairing)
        if not result:
            return False
    return True


def do_color_switching(config: Config, coloring: dict, kempe_sectors: list, grouping: Grouping, color_pairing: tuple):
    grouping_combinations = []
    for i in range(1,grouping.size+1):
        grouping_combinations += list(combinations(grouping.groups, i))
    
    for combi in grouping_combinations:
        new_coloring = coloring.copy()
        for group in combi:
            nodes = [node for node in kempe_sectors[i] for i in group.sectors]
            if coloring[nodes[0]] in color_pairing[0]:
                color_pair = color_pairing[0]
            else:
                color_pair = color_pairing[1]
            for node in nodes:
                color = coloring[node]
                new_coloring[node] = color_pair[(color_pair.index(color) + 1)%2]
        k, _ = get_special_k(config.graph, COLORS, new_coloring, len(new_coloring))
        if k is not None:
            if ggd_test_service(config.graph, k):
                return True
    return False


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
        avail = list(COLORS)

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
        return None, None

    # Turn it into list for NX.draw
    color_list = list(color_dic.values())
    return color_list, color_dic # color_list of the colors corresponding to vertices by order, coloring is dictionary


def ggd_test_service(graph, color_list):
    for e in graph.edges:
        if color_list[e[0] - 1] == color_list[e[1] - 1]:  # Node numbers start with 1, indexing starts with 0
            return False
    return True


def special_k_to_the_ggd(g, i: int):
    print(f"{i + 1}/2822")  # Hardcoded for parallelness
    print(f"ring_size:{config_arr[i].ring_size}")
    # sys.stdout.flush()
    k, _ = get_special_k(g, COLORS)
    if k is None or not ggd_test_service(g, k):
        print(f'WEEEEUUUUEEEEUUUUU NO COLOR IN MY LIFE: {i}')
        nx.draw(g)
        plt.title(f"i={i}")
        plt.show()


# Checks if coloring is isomorphic up to switching colors, GIVEN that they color the SAME graph (not an isomorphic graph necessarily)
def coloring_is_isomorphism(coloring1: dict, coloring2: dict):
    # Check same length
    if len(coloring1) != len(coloring2):
        return False

    # Check same amount of colours
    if len(set(coloring1.values())) != len(set(coloring2.values())):
        return False

    # Group the vertices by color
    c1grouped = dict()
    c2grouped = dict()
    for k,v in sorted(coloring1.items()):
        if v in c1grouped:
            c1grouped[v].append(k)
        else:
            c1grouped[v] = [k]
    for k,v in sorted(coloring2.items()):
        if v in c2grouped:
            c2grouped[v].append(k)
        else:
            c2grouped[v] = [k]

    return list(c1grouped.values()) == list(c2grouped.values())


# Doing the stuffs
########################################################################################################################

graph_arr, config_arr = import_graphs()     # Get configs from file

print(max([(cfg.size, cfg.identifier) for cfg in config_arr]))
print(max([(cfg.ring_size, cfg.identifier) for cfg in config_arr]))


colorings = []
start = timeit.default_timer()
verify_all_ring_colorings(config_arr[0])
print("Time taken: ", timeit.default_timer() - start)
start = timeit.default_timer()
verify_all_ring_colorings(config_arr[11])
print("Time taken: ", timeit.default_timer() - start)
start = timeit.default_timer()
verify_all_ring_colorings(config_arr[18])
print("Time taken: ", timeit.default_timer() - start)
start = timeit.default_timer()
verify_all_ring_colorings(config_arr[29])
print("Time taken: ", timeit.default_timer() - start)
start = timeit.default_timer()
verify_all_ring_colorings(config_arr[2685])
print("Time taken: ", timeit.default_timer() - start)
start = timeit.default_timer()
verify_all_ring_colorings(config_arr[2820])     # = conf 2821
print("Time taken: ", timeit.default_timer() - start)
# for i in range(len(config_arr)):
#     print(f"{i+1}/2282")
#     sys.stdout.flush()
#     verify_all_ring_colorings(config_arr[i])

#Parallel(n_jobs=8)(delayed(special_k_to_the_ggd)(graph_arr[i], i) for i in range(len(graph_arr)))   # Color all configs
# for i in range(len(graph_arr)): special_k_to_the_ggd(graph_arr[i], i)     # Single thread version

### Small subset for testing purposes
# for i in range(10): special_k_to_the_ggd(graph_arr[i], i)

sel = 3      # Arbitrary selection of a config

# # Draw defined subgraphs
# nx.draw(config_arr[sel].inside)
# plt.figure()
# nx.draw(config_arr[sel].ring)
# plt.figure()
# # Draw entire graph with 4 koloring
# nx.draw(graph_arr[sel],
#                node_color=get_special_k(graph_arr[sel], COLORS)[0],
#                with_labels=list(graph_arr[sel].nodes))
# plt.show()