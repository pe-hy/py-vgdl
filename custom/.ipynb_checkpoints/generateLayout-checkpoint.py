import random
import itertools
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from itertools import chain


def all_less_then(comb,n):
    for i in comb:
        if i<n:
            return False
    return True

def get_divisions(n, k, min_room_size):
    divisions = []
    combs = itertools.product(range(1, n), repeat=k)
    dic={}
    for comb in combs:
        if sum(comb) == n and all_less_then(comb,min_room_size):
            dic["".join(map(str,sorted(comb)))]=comb
    return list(dic.values())


def get_points(part):
    points = []
    y = part[0]
    x = part[1]
    for i in range(part[2]):
        for j in range(part[3]):
            points.append((y+i,x+j))
    return points


def get_choosen_rooms(divisions,min_room_size,num_rooms):
    horizontal_split = list(random.choice(divisions))
    vertical_splits = [list(random.choice(divisions)) for i in horizontal_split]
    random.shuffle(horizontal_split)
    for split in vertical_splits:
        random.shuffle(split)

    horizontal_data = []
    x_accum = 1
    for part in horizontal_split:
        horizontal_data.append((x_accum,part))
        x_accum += part+1

    vertical_data = []
    for split in vertical_splits:
        y_accum = 1
        data = []
        for part in split:
            data.append((y_accum,part))
            y_accum += part+1
        vertical_data.append(data)

    rooms_ixs = []

    while True:
        x = random.choice(range(len(horizontal_split)))
        y = random.choice(range(len(vertical_splits[x])))
        ix = (x,y)
        if ix not in rooms_ixs:
            rooms_ixs.append(ix)
        if len(rooms_ixs) == num_rooms:
            break

    chosen_rooms = []
    for ix_x,ix_y in rooms_ixs:
        x,w = horizontal_data[ix_x]
        y,h = vertical_data[ix_x][ix_y]
        chosen_rooms.append((y,x,h,w))

    walkable_points = []
    for w in chosen_rooms:
        walkable_points += get_points(w)
    walkable_points = set(walkable_points)
    return walkable_points, chosen_rooms

def create_string(walkable,grid_size):
    string = ""
    for i in range(grid_size):
        for j in range(grid_size):
            if (i,j) in walkable:
                string += " "
            else: string += "#"
        string += "\n"
    return string


def toG(y,x):
    return y-1,x-1
def toL(p):
    y,x = p
    return y+1,x+1
def get_room_center(room):
    y,x,h,w=room
    cy = y + (h//2)
    cx = x + (w//2)
    return (cy,cx)

def get_distance(r1,r2):
    A = np.array(get_points(r1))
    B = np.array(get_points(r2))
    distances = cdist(A, B)
    min_distance = np.min(distances)
    return min_distance


def get_neighs(p):
    x,y = p
    neighbors = [(x + dx, y + dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
    return neighbors

def get_path_border(path):
    border = []
    for p in path:
        border += get_neighs(p)
    surrounding_nodes = list(set(border)-set(path))
    return surrounding_nodes

def get_connecting_edges(surrounding_nodes):
    ret = []
    for node in surrounding_nodes:
        x,y = node
        neighbors = get_neighs((x,y))
        for neighbor in neighbors:
            if neighbor in surrounding_nodes:
                ret.append((node, neighbor))
    return ret

def get_path_border(path):
    border = []
    for p in path:
        border += get_neighs(p)
    surrounding_nodes = list(set(border)-set(path))
    return get_connecting_edges(surrounding_nodes)


def get_room_border(graph, room):
    y, x, h, w = room
    y,x = toG(y,x)
    surrounding_nodes = [(y - 1 + i, x - 1 + j) for i in range(h + 2) for j in range(w + 2) if i in [0, h + 1] or j in [0, w + 1]]
    return get_connecting_edges(surrounding_nodes)


def set_weights(G,taken,weigth):
    for node in taken:
        for k,v in G[node].items(): #edges connected to the node
            if G.nodes[k]['taken']==True: #ignore if the node on the other side is already taken
                  continue
            else:
                v['weigth']+=weigth
    return G

def out_of_bounds(indices,grid_size):
    for i in indices:
        if i<0 or i>grid_size-3:
            return True
    return False

def set_weight_edges(G,edges,weigth,grid_size):
    for e in edges:
        indices=set(list(e[0])+list(e[1]))
        if out_of_bounds(indices,grid_size):
            continue
        if G.nodes[e[0]]['taken']==True or G.nodes[e[1]]['taken']==True:
            continue
        G.edges[e]['weigth']=weigth
    return G 


def get_grid_graph(grid_size,walkable_points,chosen_rooms):
    G=nx.grid_graph(dim=[grid_size-2,grid_size-2])

    for e in G.edges(data=True):
        e[2]['weigth']=1
    for n,d in G.nodes(data=True):
        d['taken']=False
    for y,x in walkable_points:
        G.nodes[(y-1,x-1)]['taken']=True
    edges_around_rooms = []
    for room in chosen_rooms:
        edges_around_rooms+=get_room_border(G,room)
    G = set_weight_edges(G,edges_around_rooms,100,grid_size)    
    rooms_points =  map(lambda x:x[0],filter(lambda d: d[1]['taken'] == True, G.nodes(data=True)))
    G = set_weights(G,rooms_points,100)
    return G

def get_paths(G,sorted_pairs,chosen_rooms,grid_size):
    paths = {}
    for pair,_ in sorted_pairs:
        r1 = chosen_rooms[pair[0]]
        r2 = chosen_rooms[pair[1]]
        c1=get_room_center(r1)
        c2=get_room_center(r2)
        path=nx.shortest_path(G, source=c1, target=c2, weight='weigth')
        paths[pair]=path
        for y,x in path:
            G.nodes[(y,x)]['taken']=True
        G = set_weight_edges(G,get_path_border(path),100,grid_size)
        G=set_weights(G,path,50)
    return paths


def get_room_points(chosen_rooms):
    transformed_points = []
    for room in chosen_rooms:
        points = get_points(room)
        transformed_points.append(list(map(lambda x:toG(x[0], x[1]), points)))
    return transformed_points
    

def checkIntersection(setA, setB):
    intersection = setA.intersection(set(setB).union(set(get_path_border(setB))))
    if len(intersection) > 0:
        return True
    return False


def get_rooms_graph(chosen_rooms,paths_lst,rooms_points):
    
    edge_id = 0
    room_G = nx.Graph()
    for i, room in enumerate(chosen_rooms):
        room_G.add_node(i, room=room, points=rooms_points[i])
    # 2
    for corridor_key, corridor_points in paths_lst:
        start_room_index = corridor_key[0]
        end_room_index = corridor_key[1]
        # 3
        corridor_points = set(corridor_points)
        # 4, 5, 6
        start_room_points = set(rooms_points[start_room_index])
        end_room_points = set(rooms_points[end_room_index])
        corridor_points -= start_room_points
        corridor_points -= end_room_points

        # 7
        crossed_corridors = []
        crossed_rooms = []
        for i, room in enumerate(rooms_points):
            room_points = set(room)
            if checkIntersection(room_points, corridor_points):
                crossed_rooms.append(i)
        for j, path in enumerate(paths_lst):
            if edge_id == j:
                continue
            path_points = set(path[1])
            if checkIntersection(path_points, corridor_points):
                crossed_corridors.append(j)
        room_G.add_edge(start_room_index, end_room_index, corridor_key=edge_id, crossed_corridors=crossed_corridors, crossed_rooms=crossed_rooms)
        edge_id += 1
    return room_G


def _expand(G, explored_nodes, explored_edges):
    """
    Expand existing solution by a process akin to BFS.

    Arguments:
    ----------
    G: networkx.Graph() instance
        full graph

    explored_nodes: set of ints
        nodes visited

    explored_edges: set of 2-tuples
        edges visited

    Returns:
    --------
    solutions: list, where each entry in turns contains two sets corresponding to explored_nodes and explored_edges
        all possible expansions of explored_nodes and explored_edges

    """
    frontier_nodes = list()
    frontier_edges = list()
    for v in explored_nodes:
        for u in nx.neighbors(G,v):
            if not (u in explored_nodes):
                frontier_nodes.append(u)
                frontier_edges.append([(u,v), (v,u)])

    return zip([explored_nodes | frozenset([v]) for v in frontier_nodes], [explored_edges | frozenset(e) for e in frontier_edges])

def find_all_spanning_trees(G, root=0):
    """
    Find all spanning trees of a Graph.

    Arguments:
    ----------
    G: networkx.Graph() instance
        full graph

    Returns:
    ST: list of networkx.Graph() instances
        list of all spanning trees

    """

    # initialise solution
    explored_nodes = frozenset([root])
    explored_edges = frozenset([])
    solutions = [(explored_nodes, explored_edges)]
    # we need to expand solutions number_of_nodes-1 times
    for ii in range(G.number_of_nodes()-1):
        # get all new solutions
        solutions = [_expand(G, nodes, edges) for (nodes, edges) in solutions]
        # flatten nested structure and get unique expansions
        solutions = set([item for sublist in solutions for item in sublist])

    return [nx.from_edgelist(edges) for (nodes, edges) in solutions]

def get_valid_skeletons(G_room, room_indices):
    group = G_room.subgraph(room_indices)
    #nx.draw(group, with_labels=True)
    all_spanning_trees = find_all_spanning_trees(group, root=room_indices[0])
    valid_skeletons = []
    for spanning_tree in all_spanning_trees:
        edges = list(spanning_tree.edges())
        intersections = []
        corridor_keys = []
        valid_skeleton = True
        for edge_id in edges:
            edge = G_room[edge_id[0]][edge_id[1]]
            if all(n in room_indices for n in edge["crossed_rooms"]):
                intersections += edge["crossed_corridors"]
                corridor_keys.append(edge["corridor_key"])
            else:
                valid_skeleton = False
        if valid_skeleton:
            valid_skeletons.append({"edges": set(corridor_keys), "intersections": set(intersections)})
    return valid_skeletons


def get_group_skeletons(available,group_size,room_G):
    combs = list(itertools.combinations(available,group_size))
    for c in combs:
        skels = get_valid_skeletons(room_G, list(c))
        if len(skels)!=0:
            yield (set(c),skels)
    return (False,False)


def iterate_G(available,group_sizes,i,room_G):
    group_size = group_sizes[0]
    group_sizes = group_sizes[1:]
    iterator = get_group_skeletons(available,group_size,room_G)
    for ixs,skels in iterator:
        if ixs == False:
            return []
        if len(group_sizes)!=0:
            available_ = available-ixs
            iterator_ = iterate_G(available_,group_sizes,i+1,room_G)
            finished = False
            for groups in iterator_:
                if len(groups) != 0:
                    groups.append({'vertices':ixs,'skeletons':skels})
                    yield groups
                else:
                    yield []
        else:
            yield [{'vertices':ixs,'skeletons':skels}] 
            

def get_possible_split(groups,intersections=set(),i=0):
    group = groups[0]
    if len(groups)==1:
        groups = []
    else:
        groups = groups[1:]
    for s1 in group['skeletons']:
        if len(s1['edges'].intersection(intersections))==0: # good
            if len(groups)!=0:
                it = get_possible_split(groups,intersections.union(s1['intersections']),i=i+1)
                for s2 in it:
                    s2.append(s1)
                    yield s2
            else:
                yield [s1]
        else:
            yield []
    return []


def get_vertices_from_skels(vertices_edges_dict, skels):
    skel_vertices = {}
    #if type(skels) == list:
    #    for skel in skels:
    #        skel_edges = skel['edges']
    #        for edge_id in skel_edges:
    #            skel_vertices[edge_id] = vertices_edges_dict[edge_id]
    #else:
    skel_edges = skels['edges']
    for edge_id in skel_edges:
        skel_vertices[edge_id] = vertices_edges_dict[edge_id]
    return skel_vertices


def get_vertices_from_edge_ids(edge_ids,vertices_edges_dict_inv):
    skel_vertices = []
    for entry in vertices_edges_dict_inv:
        for edge_id in edge_ids:
            if edge_id == entry:
                skel_vertices.append(vertices_edges_dict_inv[edge_id])
    return skel_vertices

def get_edge_ids_from_vertices(vertex_tuples, vertices_edges_dict):
    edge_ids = []
    for vertex_tuple in vertex_tuples:
        for edge_id, edge_vertices in vertices_edges_dict.items():
            if vertex_tuple == edge_vertices:
                edge_ids.append(edge_id)
    return edge_ids


def get_points_from_group(group_id, skels,vertices_edges_dict_inv,room_G,paths_lst):
    edge_ids = []
    room_ids = set()
    for i, skel in enumerate(skels):
        if i == group_id:
            edge_ids.append(list(skel["edges"]))
            edge_ids = list(chain.from_iterable(edge_ids))
            room_ids = set(list(chain.from_iterable(get_vertices_from_edge_ids(edge_ids,vertices_edges_dict_inv))))
    room_points = []
    transformed_points_dict = nx.get_node_attributes(room_G, "points")
    for room in transformed_points_dict:
        for id in room_ids:
            if room == id:
                room_points.append(transformed_points_dict[id])
    corridor_points = []
    group_edges = skels[group_id]["edges"]
    for edge in group_edges:
        corridor_points.append(paths_lst[edge][1])
    corridor_points = [tuple for sublist in corridor_points for tuple in sublist]
    room_points = [tuple for sublist in room_points for tuple in sublist]
    all_points = set(corridor_points).union(set(room_points))
    return all_points

def get_points_from_room(room_id,room_G):
    room_points = []
    transformed_points_dict = nx.get_node_attributes(room_G, "points")
    for room in transformed_points_dict:
            if room_id == room:
                room_points.append(transformed_points_dict[room_id])
    room_points = [tuple for sublist in room_points for tuple in sublist]
    return room_points

def get_crossing_edges(skels, groups_to_connect,vertices_edges_dict_inv,vertices_edges_dict):
    i, j = groups_to_connect

    setA = get_vertices_from_skels(vertices_edges_dict_inv, skels[i])
    setB = get_vertices_from_skels(vertices_edges_dict_inv, skels[j])

    setA_vertices = set([vertex for edge in setA.values() for vertex in edge])
    setB_vertices = set([vertex for edge in setB.values() for vertex in edge])

    crossing_edges = list(itertools.product(setA_vertices, setB_vertices))
    crossing_edges += (list(itertools.product(setB_vertices, setA_vertices)))
    crossing_edges = [edge for edge in crossing_edges if edge in vertices_edges_dict.keys()]
    return crossing_edges

def get_edges_from_skels(crossing_edges, groups_to_connect, skels,vertices_edges_dict_inv,room_G,paths_lst,vertices_edges_dict,crossed_edges):
    i,j = groups_to_connect
    group_points = [get_points_from_group(i, skels,vertices_edges_dict_inv,room_G,paths_lst) for i, _ in enumerate(skels)]

    for c_e in crossing_edges:
        e = vertices_edges_dict[c_e]
        room_ids = c_e

        rooms_points = [get_points_from_room(i,room_G) for i in room_ids]
        rooms_points = [tuple for sublist in rooms_points for tuple in sublist]

        edge_points_only = paths_lst[e][1]

        rooms_points = set(rooms_points)
        edge_points_only = set(edge_points_only)
        edge_points = edge_points_only.difference(rooms_points)
        #vytáhnout body pro obě roomky v room_ids
        #vytáhnout body pro e
        #odseknout od "e" body obou roomek
        #edge_points je set koridoru odseknutého od roomek
        skip = False
        for g in range(len(skels)):
            if g not in (i, j):
                if len(edge_points.intersection(group_points[g])) != 0:
                    skip = True
                    break
        if skip:
            continue
        intA = edge_points.intersection(group_points[i])
        intB = edge_points.intersection(group_points[j])
        if len(intA) != 0 and len(intB) != 0:
            continue
        else:
            return e

def get_layout_str(walkable_points,skels,crossing_edges,paths_lst,grid_size):
    edges = []
    for s in skels:
        edges += list(s['edges'])
    edges += crossing_edges
    walkable_points_all = list(walkable_points)
    for ix in edges:
        walkable_points_all += list(map(toL,paths_lst[ix][1]))
    walkable_points_all = set(walkable_points_all)
    string = create_string(walkable_points_all,grid_size)
    return string

def get_crossing_edges_list(skels,room_G,groups_to_connect,paths_lst):
    vertices_edges_dict = nx.get_edge_attributes(room_G,'corridor_key')
    vertices_edges_dict_inv = {v: k for k, v in vertices_edges_dict.items()}
    crossing_edges = []
    for grs in groups_to_connect:
        possible_cr_edges = get_crossing_edges(skels, grs,vertices_edges_dict_inv,vertices_edges_dict)
        crossing_edge = get_edges_from_skels(possible_cr_edges, grs, skels,vertices_edges_dict_inv,room_G,paths_lst,vertices_edges_dict,crossing_edges)
        crossing_edges.append(crossing_edge)
    return crossing_edges