import networkx as nx
import random as rnd
import collections
from collections import defaultdict


def generateOr(G,n_sym,last_ix,pointer,level,parameter,parameter_used):
    num_ands = rnd.randint(2,3)
    G.add_node(last_ix+1,type='or',label='or')
    G.add_edge(pointer,last_ix+1)
    last_ix += 1
    pointer = last_ix
    for i in range(num_ands):
        last_ix,G = generateAnd(G,n_sym,last_ix,pointer,level,parameter,parameter_used,disj_used=True)
        parameter_used = True
    return last_ix,G


def generateAnd(G,n_sym,last_ix,pointer,level,parameter=None,parameter_used=True,disj_used=False):
    # set prob of disj to some higher number if disjunctive rules should be used
    probability_of_disj = 0
    disj = rnd.random() < probability_of_disj and not disj_used
    # generate Or node
    if disj:
        last_ix,G=generateOr(G,n_sym,last_ix,pointer,level,parameter,parameter_used)     
    # generate And node
    else:
        G.add_node(last_ix+1,type='and',label='and')
        G.add_edge(pointer,last_ix+1)
        last_ix += 1
        pointer = last_ix
        num_syms = rnd.randint(2,3)
        # generate And descendants
        for i in range(num_syms):
            # first leaf should be parameter if the molecule is parametrized
            if not parameter_used:
                G.add_node(last_ix+1,type='par',label=str(parameter))                    
                G.add_edge(pointer,last_ix+1) 
            # normal symbol
            else:
                sym = rnd.randint(0,n_sym)
                in_par = (sym  // 5) == parameter
                # the symbol may be sampled from the range of the parameter
                if in_par:
                    G.add_node(last_ix+1,type='par',label=str(parameter))
                else: G.add_node(last_ix+1,type='sym',label=str(sym))                    
                G.add_edge(pointer,last_ix+1) 
            last_ix += 1
            parameter_used = True
    return last_ix, G
        

def generateMoleculePart(mol_id,num_of_atoms):
    # set prob_of_parametrized to higher value if you want parametrized molecule
    prob_of_parametrized = 0
    parametrized = rnd.random() < prob_of_parametrized
    G = nx.DiGraph()
    last_ix = mol_id
    pointer = last_ix
    G.add_node(last_ix,type='root',label = str(mol_id))
    parameter = None
    if parametrized:
        parameter = rnd.randint(0,4)
        ix,graph = generateAnd(G,num_of_atoms,last_ix,pointer,1,parameter,False)
    else:
        ix,graph = generateAnd(G,num_of_atoms,last_ix,pointer,1)
    return last_ix,graph, parameter



def adjoinSubtree(G,sG,leaf,root,num_replaced):
    G = G.copy()
    
    if num_replaced > 0:
        leaf_name = num_replaced*'g'+str(leaf)
        par = list(G.predecessors(leaf_name))[0]
        G.remove_node(leaf_name)
    else:
        par = list(G.predecessors(leaf))[0]
        G.remove_node(leaf)
    G = nx.union(G,sG,rename=('g','h'))
    par_name,root_name = 'g'+str(par),'h'+str(root)
    G.add_edge(par_name,root_name)
    G.nodes[root_name]['type']='sub'
    return G

def makeParametrized(G,parameters):
    parameters = list(set(parameters))
    leafs = [x for x in G.nodes(data=True) if G.out_degree(x[0])==0 and G.in_degree(x[0])]
    for l in leafs:
        p_range = int(l[1]['label'])//5
        if l[1]['type']=='sym' and p_range in parameters:
            p_ix = parameters.index(p_range)
            G.nodes[l[0]]['type']='par'
            G.nodes[l[0]]['label']=parameters[p_ix]
    return G


def generateMolecule(advanced_mols,basic_mols):
    params = []
    ix,G,p=rnd.choice(advanced_mols)
    leafs = [x for x in G.nodes(data=True) if G.out_degree(x[0])==0 and G.in_degree(x[0])]
    params.append(p)
    num_replaced = 0
    for l in leafs:
        replace = rnd.random()<0.4
        if replace:
            ix,sG,p=rnd.choice(basic_mols)
            params.append(p)
            G=adjoinSubtree(G,sG,l[0],ix,num_replaced)
            num_replaced+=1
    return params,makeParametrized(G,params)


def get_edges_to_and(G):
    edges = [x for x in G.edges() if G.nodes[x[0]]['type']=='and']
    rnd.shuffle(edges)
    return edges

def append_subgoal(subgoals,mols):
    k = rnd.choice(range(len(mols)))
    params,G = mols[k]
    if len(subgoals)==0:       
        subgoals.append({'obstacle_G_ix': None,
                         'edge':None,
                         'params':params,
                         'molecule':G,
                         'molecule_ix':k
                        })
    else:
        obstacle_G_ix = rnd.randint(0,len(subgoals)-1)
        obstacle_G = subgoals[obstacle_G_ix]['molecule']
        edges = get_edges_to_and(obstacle_G)
        subgoals.append({'obstacle_G_ix': obstacle_G_ix,
                         'edge':edges[0],
                         'params':params,
                         'molecule':G,
                         'molecule_ix':k
                        })
    return subgoals
        

def generateInstanceData(mols): 
    subgoals = []
    num_obs = rnd.randint(1,3)
    for i in range(num_obs):
        subgoals = append_subgoal(subgoals,mols)
    return subgoals


def get_spliting_edges(instance):
    edges = defaultdict(list)
    for i,m in enumerate(instance):
        if m['edge']==None:
            continue
        else:
            edges[m['obstacle_G_ix']].append(m['edge'])
    return edges


def collect_symbols(molecule):
    symbols = []
    G=molecule
    for v in G.nodes(data=True):
        if v[1]['type']=='sym':
            symbols.append(v[1]['label'])
    return [symbols]


    


def split_molecule(molecule,edges):
    G = molecule['molecule'].copy()
    def collect_symbols_comp(comp):
        comp_syms = []
        for v in comp:
            vData = G.nodes(data=True)[v]
            if vData['type']=='sym':
                comp_syms.append(vData['label'])
        return comp_syms
    for e in edges:
        G.remove_edge(e[0], e[1])
    comps = nx.weakly_connected_components(G)
    comps = list(map(collect_symbols_comp,comps ))
    return comps
        
def get_comps(index,molecule,edges):
    if index not in edges:
        return collect_symbols(molecule['molecule'])
    else:
        return split_molecule(molecule,edges[index])

def get_splited_molecules(instance):
    edges = get_spliting_edges(instance)
    splited_molecules = {}
    for i,m in enumerate(instance):
        splited_molecules[i]={
            'comps':get_comps(i,m,edges),
            'doorTo':m['obstacle_G_ix']
        }
    return splited_molecules

def get_pointing_rooms(ix,pointers):
    pointing_rooms = []
    for k,v in pointers.items():
        if v==ix:
            pointing_rooms.append(k)
    return pointing_rooms

def distribute_atoms(comps,pointing_rooms):
    assert (len(comps)-1)==len(pointing_rooms)
    if len(pointing_rooms)==0:
        return comps,None
    rnd.shuffle(comps)
    distribution = defaultdict(list)
    comps_left = [comps[-1]]
    for i,r in enumerate(pointing_rooms):
        add = rnd.random() < 2
        comp = comps[i]
        if add:
            distribution[r].append(comp)
        else:
            comps_left.append(comp)
    return comps_left,distribution
            
def get_blocks(instance):
    dic = get_splited_molecules(instance)
    num_blocks = len(dic)
    new_dic = {}
    pointers_dic = {}
    for i in range(num_blocks):
        pointers_dic[i] = dic[i]['doorTo']
    for i in range(num_blocks):
        ix = num_blocks-1-i
        pointing_rooms = get_pointing_rooms(ix,pointers_dic)
        comps = dic[ix]['comps']
        comps_left,dist = distribute_atoms(comps,pointing_rooms)
        new_dic[ix] = (comps_left,dist)
    return pointers_dic,new_dic

def flatten(lst):
    flattened_list = []
    for element in lst:
        if type(element) == list:
            flattened_list.extend(flatten(element))
        else:
            flattened_list.append(element)
    return flattened_list

def merge_block(v):
    dic = {}
    if type(v[0])==defaultdict:
        ks = list(v[0].keys())
        k = ks[0]
        flat_block = flatten(v[0][k]+v[1:])
        dic[k] = flat_block
        for i in range(len(ks)-1):
            dic[ks[i+1]] = v[0][ks[i+1]] 
    else:
        dic = flatten(v)
    return dic


def split_node(G,node,parent):
    splits = list(G.nodes(data=True)[node]['data'].values())
    G.remove_node(node)
    for i in range(len(splits)):
        if i==0:
            name = node
        else:
            name = str(node)+"_"*i
        G.add_node(name,data=splits[i])
        G.add_edge(parent,name)
    return G

def merge_parents(G,parents,node):
    data = []
    for p in parents:
        data.append(G.nodes(data=True)[p]['data'])
        G.remove_node(p)
    G.add_node(p,data=flatten(data))
    succs = list(G.successors(node))
    G = split_node(G,node,p)
    assert len(succs)<2
    if len(succs)==1: 
        G.add_edge(node,succs[0])
    return G

def get_block_graph(merged_blocks):
    G = nx.DiGraph()
    edges = []
    for k,v in merged_blocks.items():
        if type(v)==dict:
            for k_,v_ in v.items():
                edges.append((k_,k))
        G.add_node(k,data=v)
    for e in edges:
        G.add_edge(*e)

    nodes_to_split = []

    for v in G.nodes():
        if len(G.in_edges(v))>1:
            nodes_to_split.append(v)
    for v in sorted(nodes_to_split):
        parents = list(G.predecessors(v))
        G = merge_parents(G,parents,v)
    return G, nodes_to_split

def get_merged_blocks(blocks):
    atoms_in_blocks = defaultdict(list)
    for k,v in blocks.items():
        locked = blocks[k][1]
        if locked == None:
            atoms_in_blocks[k].append(blocks[k][0])
        else:
            atoms_in_blocks[k].append(locked)
            atoms_in_blocks[list(locked.keys())[0]].append(blocks[k][0])

    merged_blocks = {}
    for k,v in atoms_in_blocks.items():
        merged_blocks[k]=merge_block(v)
    return merged_blocks