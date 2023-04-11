import random as rnd
from collections import defaultdict

import networkx as nx

from custom.EnvironmentGeneration.MoleculeGeneration import get_blocks, flatten, split_node, merge_block


def merge_parents(G, parents, node):
    data = []
    for p in parents:
        data.append(G.nodes(data=True)[p]['data'])
        G.remove_node(p)
    G.add_node(p, data=flatten(data))
    succs = list(G.successors(node))
    G = split_node(G, node, p)
    assert len(succs) < 2
    if len(succs) == 1:
        G.add_edge(node, succs[0])
    return G


def get_block_graph(merged_blocks, mol_ids):
    G = nx.DiGraph()
    edges = []
    for k, v in merged_blocks.items():
        if type(v) == dict:
            for k_, v_ in v.items():
                edges.append(((k_, k), mol_ids[k_]))
        G.add_node(k, data=v)
    for e, v in edges:
        G.add_edge(*e, obstacle_id=v)

    nodes_to_split = []

    for v in G.nodes():
        if len(G.in_edges(v)) > 1:
            nodes_to_split.append(v)
    for v in sorted(nodes_to_split):
        parents = list(G.predecessors(v))
        G = merge_parents(G, parents, v)
    return G, nodes_to_split


def get_merged_blocks(blocks):
    atoms_in_blocks = defaultdict(list)
    for k, v in blocks.items():
        locked = blocks[k][1]
        if locked == None:
            atoms_in_blocks[k].append(blocks[k][0])
        else:
            atoms_in_blocks[k].append(locked)
            atoms_in_blocks[list(locked.keys())[0]].append(blocks[k][0])

    merged_blocks = {}
    for k, v in atoms_in_blocks.items():
        merged_blocks[k] = merge_block(v)
    return merged_blocks


def split_to_rooms(lst):
    max_ = min(len(lst) - 1, 3)
    min_ = min((len(lst) - 1) // 3, 3)
    num_splits = rnd.randint(min_, max_)
    possible = list(range(1, len(lst)))
    splits = sorted(rnd.sample(possible, num_splits))
    chunks = []
    start = 0
    end = 0
    for end in splits:
        chunks.append(lst[start:end])
        start = end
    chunks.append(lst[end:])
    return chunks


def get_molecule_ids(instance):
    return [i['molecule_ix'] for i in instance]


def get_instance_graph(instance):
    pointers, blocks = get_blocks(instance)
    mol_ids = get_molecule_ids(instance)
    merged_blocks = get_merged_blocks(blocks)
    G, nodes_to_split = get_block_graph(merged_blocks, mol_ids)
    for node in G.nodes(data=True):
        data = node[1]['data']
        if type(data) != list:
            data = list(data.values())[0]
        data = split_to_rooms(data)
        G.nodes[node[0]]['data'] = data
    return G
