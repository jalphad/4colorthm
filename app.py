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
import math

# Global settings
PRINT_CLR_CNT = True
PRINT_RESULTS = False
PRINT_FALSE = True

# Global constants
# 1: red
# 2: blue
# 3: green
# 4: yellow
COLORS = (1, 2, 3, 4)
COLOR_PAIRINGS = (
    ((1, 2), (3, 4)),
    ((1, 3), (2, 4)),
    ((1, 4), (2, 3))
)


class Config:
    identifier = None
    graph = None
    size = None
    ring_size = None
    no_of_colorings = None
    ring = None
    inside = None


# Data structure to organise info about the Kempe-"blocks" in a ring, for a given coloring/pairing
# Note that an instance is constructed externally, kempe_sectors are not automatically generated for example
# Can also be instantiated with data that would not constitute a grouping. is_valid determines whether this is actually
# the case.
class Grouping:

    # Group essentially stored as a list, and the amount of connections to each other group is tracked
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
            if self.connection_count[group] < 0:
                raise Exception("Negative amount of connections")

    # Grouping is a list of groups, while simultaneously keeping track of the group of every sector for performance
    def __init__(self, kempe_sectors: list, coloring: dict, color_pairing: tuple, size: int) -> None:
        self.size = size    # Amount of groups
        self.sectors = kempe_sectors    # In order of ring => i & i+1%ringsize are neighbouring sectors
        self.coloring = coloring    # Coloring of the ring
        self.color_pairing = color_pairing
        self.groups = []
        for i in range(size):
            self.groups.append(self.Group(i, size))
        self.sectors_to_group = {}  # Dictionary for faster performance
        self.add_sector_to_group(0, 0)   # "First" sector is added to block 0.

    # Returns groups of neighbouring sectors
    # Note:
    # If sectors don't belong to a group yet, then also no group is returned for that neighbour
    # Neighbouring sectors are never in the same group
    def get_neighboring_groups(self, sector: int):
        neighbors = []
        if (sector-1)%len(self.sectors) in self.sectors_to_group.keys():    # If the sector belongs to a group
            neighbors.append(self.sectors_to_group.get((sector-1)%len(self.sectors)))   # Get the group of neighbour
        if (sector+1)%len(self.sectors) in self.sectors_to_group.keys():
            neighbors.append(self.sectors_to_group.get((sector+1)%len(self.sectors)))
        return neighbors

    # Returns all groups not used by neighbouring sectors
    def get_available_groups(self, sector: int):
        in_use = self.get_neighboring_groups(sector)
        return [i for i in range(self.size) if i not in in_use]

    # Title says it all
    def add_sector_to_group(self, sector: int, group: int):
        self.sectors_to_group[sector] = group   # Tie group to the sector
        self.groups[group].add(sector)  # Add sector to collection of sectors of the group
        # If sector next to it is in a group, then both groups are connected => +1 connection_count
        for n in self.get_neighboring_groups(sector):   # Groups of neighbouring sectors are now connected so:
            self.groups[group].update_count(n, "add")   # Added group has one more connection
            self.groups[n].update_count(group, "add")   # The neighbouring group has one more connection with added group

    # CTRL_Z for add_sector_to_group
    def remove_last_sector_from_group(self):
        sector, group = self.sectors_to_group.popitem()  # Remove "registration of group membership" for sector
        # Idem dito as in add_sector
        for n in self.get_neighboring_groups(sector):   # For the groups of neighbouring sectors
            self.groups[group].update_count(n, "remove")
            self.groups[n].update_count(group, "remove")
        self.groups[group].sectors.remove(sector)

    # Checks the 3 conditions on whether this is actually a grouping / blocks
    def is_valid(self):
        def is_same_type():     # All colors in group must be member of the same color pair
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

        # Removing (nodes of) ANY group in the ring => all groups are in one connected component of the leftover "ring"
        def has_no_mutual_overlap():
            for removed in self.groups:
                for group in self.groups:
                    if group.id == removed.id:  # We're only  checking the groups we didn't remove
                        continue
                    if len(group.sectors) <= 1:     # Edge case
                        continue
                    reached = [group.sectors[0]]    # Start with the "first" sector"
                    # List all sectors of the group that we can reach without going over removed sector
                    for i in range(group.sectors[0]+1, group.sectors[-1]+1):  # In one direction
                        if i in removed.sectors:
                            break
                        if i in group.sectors:
                            reached.append(i)
                    if len(reached) == len(group.sectors):  # If already succeeded, bypass other direction
                        continue
                    for i in range(group.sectors[0]-1, group.sectors[1]-len(self.sectors)-1, -1): # In other direction
                        if i%len(self.sectors) in removed.sectors:
                            break
                        if i%len(self.sectors) in group.sectors:
                            reached.append(i)
                    if len(reached) != len(group.sectors):  # Implies we couldn't reach all sectors => not connected
                        return False
            return True


        def is_properly_connected():    # If blocks are connected => they are connected at two points
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
    # Recursive function over all ring nodes: "1" -> "RING_NODES_AMOUNT".
    # Note that the first nodes of the conf are the ring.
    def recurse(config, node: int, coloring: dict, ambiguous_colors: list):
        # BASE KAAS (base case)
        # If node number is > ring_size, the node is not in the ring, but in the inside.
        if node > config.ring_size:
            # Therefore all nodes in ring are colored: Now check reducibility and return result
            recurse.coloring_cnt += 1
            result = check_reducible(config, node, coloring)
            return [(coloring.copy(), result)]
        # RECURSIVE KEES
        else:
            graph = config.ring
            result = []

            # Check which colors are not taken by neighbours
            colors_in_use = [coloring[i] for i in list(graph.neighbors(node)) if i in coloring.keys()]
            colors_available = list(filter(lambda c: c not in colors_in_use, COLORS))

            # If there are more than 1 color unused, choosing between them is arbitrary and would result in isomorphism
            # If there is an arbitrary color decision (2 or more colors not used in the entire coloring), split recursive
            # cases for the "new" color, and other available colors.

            if len(ambiguous_colors) > 1:  # If there are multiple unused colors
                for c in ambiguous_colors:  # Ignore all of these in the code below
                    colors_available.remove(c)
                # Now execute the case of picking one of the colours arbitrarily separately
                coloring[node] = ambiguous_colors.pop()     # Pick an unused color (now used, so removed from ambig...)
                result += recurse(config, node + 1, coloring, ambiguous_colors) # Use it and continue recursion
                ambiguous_colors.append(coloring[node])     # For the other cases it's not used, push it back on

            while len(colors_available) > 0:    # The normal case: go over all possible colors for this node

                while len(coloring) > node:     # Same coloring structure is used, so we have to undo the effect further
                    # down the recursion before we a new possible color
                    coloring.popitem()  # CTRL Z on the coloring, let's try the other possibility
                coloring[node] = colors_available[0]    # Try a color
                colors_available = colors_available[1::]    # Remove color from our to-do list
                result += recurse(config, node + 1, coloring, ambiguous_colors)     # Append result tuple to long list
            return result

    # Set up
    colors_left = [c for c in COLORS]   # ALl colors are not used in the whole coloring. Starting with any is arbitrary
    coloring = {}    # Start with empty coloring
    recurse.coloring_cnt = 0    # Set the counter for tracking how may ring colorings we actually

    # Call the recursive function
    result = recurse(config, 1, coloring, colors_left)  #  result temporarily for possible displays

    # Display results
    if PRINT_FALSE:
        if False in result[::][1]:
            print(f"Non-reducible coloring exists: {config.identifier}")
    if PRINT_CLR_CNT:
        print(recurse.coloring_cnt)
    if PRINT_RESULTS:
        print(result)
    return result


def check_reducible(config: Config, node: int, coloring: dict):
    k, _ = get_special_k(config.graph, COLORS, coloring, node)
    if k is None or not ggd_test_service(config.graph, k):
        # If not extending to inside, find a color pairing for which all possible groupings/block decomposition works
        for pairing in COLOR_PAIRINGS:
            kempe_sectors = compute_kempe_sectors(coloring, pairing)
            # 1 or 2 Kempe sectors will not result in a ring coloring which extends to the interior of the graph
            # TODO: Proof?
            if len(kempe_sectors) < 3:
                continue
            # It seems that if there are an uneven amount of Kempe sectors, a valid grouping cannot be done
            # TODO: Proof
            elif len(kempe_sectors)%2 == 1:
                continue
            elif verify_all_sector_groupings(config, coloring, kempe_sectors, pairing):
                return True
            else:
                continue
        return False
    else:   # If we can color the inside straight away, bypass the rest and move on
        return True


def compute_kempe_sectors(coloring: dict, pairing: tuple):
    sectors = []
    previous = -1
    # As byproduct of other code / way rings are in conf file, consecutive nodes in for loop are neighbours in ring
    # TODO: For rigorousness and stability maybe sort this first per key?
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
    # Merge if first and last are actually same
    if ((coloring[sectors[0][0]] in pairing[0] and coloring[sectors[-1][0]] in pairing[0]) or
        (coloring[sectors[0][0]] in pairing[1] and coloring[sectors[-1][0]] in pairing[1])):
        if sectors[0] != sectors[-1]:
            sectors[0] += sectors[-1]
            sectors.pop(-1)

    return sectors


def verify_all_sector_groupings(config: Config, coloring: dict, kempe_sectors: list, color_pairing: tuple):
    def recurse(grouping: Grouping, sector: int, groupings: list, groupings_size: int):
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
            recurse(grouping, sector+1, groupings, groupings_size)
            if groupings_size == len(groupings):
                grouping.remove_last_sector_from_group()
                return
        grouping.remove_last_sector_from_group()


    valid_groupings = []

    # While I haven't looked at a proof for this it seems that:
    # - an uneven amount of kempe sectors will never have a valid grouping
    # - for n kempe sectors where n is even, will only have valid groupings of size n/2 + 1
    # - the amount of valid groupings found is (n choose (n/2 + 1))/(n/2)
    ks_size = len(kempe_sectors)
    groups = ks_size//2 + 1
    vgsize = math.comb(ks_size, groups) // (ks_size // 2)
    grouping = Grouping(kempe_sectors, coloring, color_pairing, groups)
    recurse(grouping, 1, valid_groupings, vgsize)

    # for max_groups in range(3, len(kempe_sectors)):
    #     grouping = Grouping(kempe_sectors, coloring, color_pairing, max_groups)
    #     recurse(grouping, 1, valid_groupings)

    if len(valid_groupings) == 0:
        raise Exception("There should be valid groupings")

    # print(f"{len(valid_groupings)} valid groupings found")
    for grouping in valid_groupings:
        result = do_color_switching(config, coloring, kempe_sectors, grouping, color_pairing)
    # From my current understanding, I don't have to be able to color switch and then color all block decompositions
    # If I can find a valid block decomposition for which I can then do color switching which extends to the configuration inside we're done
    # As an example try:
    # - the first configuration
    # - with ring coloring [y g b r g r]
    # - use color pairing (r y)(b g) since the others won't result in any possible coloring
        if result:
            return True
    return False
    # So from my current understanding this is not correct (see above)
    #     if not result:
    #         return False
    # return True


# Why are sectors not implicitly passed using the Grouping parameter, but separately? @Joren
# Try all possible allowed changes in the coloring, until we find end up with an extendable one
def do_color_switching(config: Config, coloring: dict, kempe_sectors: list, grouping: Grouping, color_pairing: tuple):
    grouping_combinations = []
    for i in range(1,grouping.size+1):  # Combinatorics: all possible combinations of allowed color switch actions
        grouping_combinations += list(combinations(grouping.groups, i))
    
    for combi in grouping_combinations:     # Try every combinations
        new_coloring = coloring.copy()  # Start over again with the original situation
        for group in combi: # For all groups that are selected to be switched
            nodes = [node for i in group.sectors for node in kempe_sectors[i]]  # List all individual nodes
            if coloring[nodes[0]] in color_pairing[0]:  # Check first node to see colour pairing of the sector/group
                color_pair = color_pairing[0]
            else:
                color_pair = color_pairing[1]
            for node in nodes:  # Now use that color pairing to do switcheroooo for all nodes in group
                color = coloring[node]
                new_coloring[node] = color_pair[(color_pair.index(color) + 1)%2]
        k, _ = get_special_k(config.graph, COLORS, new_coloring, len(new_coloring))
        if k is not None:   # If we found a coloring over the whole graph (w/ inside)
            if ggd_test_service(config.graph, k):   # Verify that this extendable coloring is valid
                return True # Hurray, you are D-reducible
    return False # No color switch could make our relationship work, we can no longer keep living like this


# Different version
def do_color_switching_2(config: Config, coloring: dict, grouping: Grouping):
    def recurse(config: Config, coloring: dict, grouping: Grouping, cur_group_index:int):
        if cur_group_index >= grouping.size:    # Base case
            k, _ = get_special_k(config.graph, COLORS, coloring, len(coloring))
            if k is not None:  # If we found a coloring over the whole graph (w/ inside)
                if ggd_test_service(config.graph, k):  # Verify that this extendable coloring is valid
                    return True  # Hurray, you are D-reducible
            else:
                return False
        else:
            if recurse(config, coloring, grouping, cur_group_index+1):
                return True
            else:
                cur_group = grouping.groups[cur_group_index]
                nodes = [node for i in cur_group.sectors for node in grouping.sectors[i]]  # List all individual nodes
                if coloring[nodes[0]] in grouping.color_pairing[
                    0]:  # Check first node to see colour pairing of the sector/group
                    color_pair = grouping.color_pairing[0]
                else:
                    color_pair = grouping.color_pairing[1]
                for node in nodes:  # Now use that color pairing to do switcheroooo for all nodes in group
                    color = coloring[node]
                    coloring[node] = color_pair[(color_pair.index(color) + 1) % 2]
                return recurse(config, coloring, grouping, cur_group_index)
    return recurse(config, coloring, grouping, 0)


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
        print(f'ALARM: NOT COLORABLE: {i}')
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


def find_coloring_isomorphism(x):
    ans = verify_all_ring_colorings(config_arr[x])
    for a in ans:
        for b in ans:
            if a != b:
                print(f"{ans.index(a)} x {ans.index(b)}: {coloring_is_isomorphism(a[0], b[0])}")


def timed_reducibility_check(x):
    start = timeit.default_timer()
    verify_all_ring_colorings(config_arr[x])
    print("Time taken: ", timeit.default_timer() - start)


def draw_defined_subgraphs(sel):
    nx.draw(config_arr[sel].inside)
    plt.figure()
    nx.draw(config_arr[sel].ring)
    plt.figure()
    # Draw entire graph with 4 koloring
    nx.draw(graph_arr[sel],
                   node_color=get_special_k(graph_arr[sel], COLORS)[0],
                   with_labels=list(graph_arr[sel].nodes))
    plt.show()


def color_graphs_all(multi_thread=True):
    if multi_thread:
        Parallel(n_jobs=8)(delayed(special_k_to_the_ggd)(graph_arr[i], i) for i in range(len(graph_arr)))  # Color all configs
    else:
        for i in range(len(graph_arr)): special_k_to_the_ggd(graph_arr[i], i)     # Single thread version


def d_reduce_all(configs: list, multi_thread=True):
    if multi_thread:
        Parallel(n_jobs=8)(delayed(verify_all_ring_colorings)(cfg) for cfg in configs)  # Color all configs
    else:
        for cfg in configs: verify_all_ring_colorings(cfg)    # Single thread version


# Doing the stuffs
########################################################################################################################

graph_arr, config_arr = import_graphs()     # Get configs from file

timed_reducibility_check(0)
timed_reducibility_check(11)
timed_reducibility_check(18)
timed_reducibility_check(29)
timed_reducibility_check(2685)
timed_reducibility_check(2820)