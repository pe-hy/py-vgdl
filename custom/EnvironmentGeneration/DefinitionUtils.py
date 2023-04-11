import pickle
import itertools
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

def find_and_ids(mol_graph):
    and_ids = []
    for node, data in mol_graph.nodes(data=True):
        if data["type"] == "and":
            and_ids.append(node)

    return and_ids

def find_syms(mol_graph):
    sym_dict = {}
    and_ids = find_and_ids(mol_graph)
    for idx in and_ids:
        sym_list = [v for u, v in mol_graph.out_edges(idx)]
        sym_dict[idx] = sym_list
    return sym_dict

def find_ands_with_only_sym(mol_graph, and_ids, sym_list):
    lst_lower = []
    lst_upper = []
    for idx in and_ids:
        current_and = sym_list[idx]
        lst = []
        for v in current_and:
            v_type = mol_graph.nodes[v]["type"]
            lst.append(v_type)
        if "sub" in lst:
            lst_upper.append(idx)
        else:
            lst_lower.append(idx)
    return {"lower": lst_lower, "upper": lst_upper}

def generate_comb_ids(lst,last_id,dic):
    last_id += 1
    for el in lst:
        k = tuple(sorted(el))
        dic[k]=last_id
    return dic,last_id


def generate_combinations(lst):
    for i in range(1, (len(lst)//2)+1):
        for subset in itertools.combinations(range(len(lst)), i):
            # get complement of subset
            complement = [x for x in range(len(lst)) if x not in subset]
            yield [lst[x] for x in subset], [lst[x] for x in complement]

def make_2tuple(tup):
    if len(tup)>2:
        tup = ((tup[0],),make_2tuple(tup[1:]))
    return tup

def process_tuple(tup,dic,last_id,lst):
    if tup==(7, 18):
        print(last_id)
    tup_key = make_2tuple(tup)
    if tup_key in dic.keys():
        lst.append(tup_key)
        id = dic[tup_key]
        return dic,id, last_id, lst
    if len(tup)==2:
        if tup[0]>20000 or tup[1]>20000:
            print(tup)
        new_id = last_id + 1
        dic[tup] = new_id
        lst.append(tup)
        return dic,new_id, new_id, lst
    else:
        subset_syms = []
        for t1,t2 in generate_combinations(tup):
            t1 = tuple(sorted(t1))
            t2 = tuple(sorted(t2))
            dic, sym_id1, last_id, lst = process_tuple(t1,dic,last_id,lst)
            dic, sym_id2, last_id, lst = process_tuple(t2,dic,last_id,lst)
            subset_syms.append((t1,t2))
            lst.append(tuple(sorted((t1,t2))))
        #print(last_id)
        dic,last_id = generate_comb_ids(subset_syms,last_id,dic)
        return dic, last_id, last_id, lst

def get_and_interaction(mol_graph, and_id, sym_list, dic, last_id):
    tmp = sym_list[and_id]
    lst = []
    for i in tmp:
        label = mol_graph.nodes[i]["label"]
        lst.append(int(label))
    tup = tuple(sorted(lst))
    #tup = (1,1,5,7,8)
    dic, sym_id, last_id, lst = process_tuple(tup,dic,last_id,[])
    return dic, sym_id, last_id, tup, lst


def get_mol_info(G, dic, last_id_lower, last_id_upper):
    #lower
    G = G.copy()
    sym_list = find_syms(G)
    lsts = []
    and_ids = find_and_ids(G)
    lower_upper_dic = find_ands_with_only_sym(G, and_ids, sym_list)
    and_ids_lower = lower_upper_dic["lower"]
    for id in and_ids_lower:
        dic,sym_id,last_id_lower, tup, lst = get_and_interaction(G, id, sym_list, dic, last_id_lower)
        lsts.extend(lst)
        for node in G.nodes():
            if id in G.successors(node):
                G.add_node(node, label = sym_id)
    #upper
    #print("-"*20)
    and_ids_upper = lower_upper_dic["upper"]
    assert len(and_ids_upper)<=1
    if len(and_ids_upper)==0:
        assert len(and_ids_lower)==1
    for id in and_ids_upper:
        dic,sym_id,last_id_upper, tup, lst = get_and_interaction(G, id, sym_list, dic, last_id_upper)
        lsts.extend(lst)
    #print(last_id_lower,last_id_upper)
    #print("*"*20)
    return dic, lsts, last_id_lower, last_id_upper,sym_id

def get_key(k, dic):
    k0,k1 = k
    if not isinstance(k0,int) and len(k0)>1:
        k0 = dic[k0]
    if not isinstance(k1,int) and len(k1)>1:
        k1 = dic[k1]
    if isinstance(k0,tuple):
        k0 = k0[0]
    if isinstance(k1,tuple):
        k1 = k1[0]
    return (k0,k1)

def rewrite_dict(dic):
    new_dict = {}
    for k,v in dic.items():
        new_k = get_key(k,dic)
        new_dict[k] = (new_k,v)
    return new_dict

def get_left_sides(k,dic_all):
    left_sides = []
    for k1,v in dic_all.items():
        if v==k:
            left_sides.append(k1)
    return left_sides

def get_child_rules(k,dic_all,child_rules):
    if k[0]>=10000: #non-atom
        left_sides = get_left_sides(k[0],dic_all)
        for i in left_sides:
            child_rules.append((i,k[0]))
            child_rules=get_child_rules(i,dic_all,child_rules)
    if k[1]>=10000: #non-atom
        left_sides = get_left_sides(k[1],dic_all)
        for i in left_sides:
            child_rules.append((i,k[1]))
            child_rules=get_child_rules(i,dic_all,child_rules)
    return child_rules

def get_complete_mol_keys(mol_keys_dic,dic_all):
    new_rules = []
    new_mol_keys_dic = {k:v for k,v in mol_keys_dic.items()}
    for k,v in mol_keys_dic.items():
        child_rules = get_child_rules(k,dic_all,[])
        new_rules.extend(child_rules)
    for i in new_rules:
        new_mol_keys_dic[i[0]] = i[1]
    return new_mol_keys_dic

