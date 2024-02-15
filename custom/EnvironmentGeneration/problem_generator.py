from utils.MoleculeGeneration import *
from utils.CreateInstanceGraph import get_instance_graph
from utils.DefinitionUtils import *
import pickle

# GraphGeneratorMolecules
num_of_atoms = 20
basic_mols = [generateMoleculePart(100 * i, num_of_atoms) for i in range(20)]
advanced_mols = [generateMoleculePart(10000 * i, num_of_atoms) for i in range(20)]

mols = {i: generateMolecule(advanced_mols, basic_mols) for i in range(50)}
instancesData = [generateInstanceData(mols) for i in range(500)]

graph_instances_lst = []
for i, instance in enumerate(instancesData):
    try:
        g = get_instance_graph(instance)
        graph_instances_lst.append({"layout_spec": g, "molecules": instance})
    except:
        pass

with open('pickles/graph_instances_lst.pkl', 'wb') as f:
    pickle.dump(graph_instances_lst, f)

with open("pickles/mols.pkl", "wb") as f:
    pickle.dump(mols, f)

print("generated problems:", len(graph_instances_lst), "saved to: pickles/mols.pkl, pickles/graph_instances.pkl")

dic_all = {}
mol_keys_dic = {}
mol_final_id_dic = {}
last_id_lower, last_id_upper = 10000, 100000
for mol in mols.items():
    id, m = mol
    l, this_G = m
    dic_all, lsts, last_id_lower, last_id_upper, sym_id = get_mol_info(this_G, dic_all, last_id_lower + 1,
                                                                       last_id_upper + 1)
    mol_final_id_dic[id] = sym_id
    mol_keys_dic[id] = lsts

rewriten_dic_all = rewrite_dict(dic_all)
clean_dic = {k: v for _, (k, v) in rewriten_dic_all.items()}

rewritten_mol_keys_dic = {}
for k, v in mol_keys_dic.items():
    mol_dic = {}
    for i in v:
        pre, post = rewriten_dic_all[i]
        mol_dic[pre] = post
    rewritten_mol_keys_dic[k] = mol_dic

complete_mol_keys_dic = {}
for k, v in rewritten_mol_keys_dic.items():
    complete_mol_keys_dic[k] = {'final_sym': mol_final_id_dic[k],
                                'molecule_rules': get_complete_mol_keys(rewritten_mol_keys_dic[k], clean_dic)}

test_dic = {}
for k, v in complete_mol_keys_dic.items():
    v = v['molecule_rules']
    for k1, v1 in v.items():
        if k1 in test_dic:
            assert test_dic[k1] == v1
        test_dic[k1] = v1
assert test_dic == clean_dic

with open("pickles/definition_dic.pkl", "wb") as file:
    pickle.dump({'all_dic': clean_dic, 'mol_keys_dic': complete_mol_keys_dic}, file)

print("Saved pickles/definition_dic.pkl")

OBSTACLE_INCR = 1000

obstacle_dic = {k: v["final_sym"] + OBSTACLE_INCR for k, v in complete_mol_keys_dic.items()}

with open('pickles/obstacle_dic.pkl', 'wb') as file:
    pickle.dump(obstacle_dic, file)

all_def_dic = clean_dic
mol_def_dic = complete_mol_keys_dic

final_sym_lst = []
for k, v in mol_def_dic.items():
    final_sym = v["final_sym"]
    final_sym_lst.append(final_sym)

number_set = set()
for key, value in all_def_dic.items():
    number_set.add(key[0])
    number_set.add(key[1])
    number_set.add(value)

symbols = [chr(i) for i in range(945, 970)] + [chr(i) for i in range(1000, 1200)]
symbols_for_obstacles = [i + OBSTACLE_INCR for i in final_sym_lst]
symbols_sorted = sorted(symbols)

number_set.update(symbols_for_obstacles)
symbol_mappings = {}

for i, number in enumerate(sorted(number_set)):
    symbol_mappings[number] = symbols_sorted[i]

with open('pickles/symbol_mappings.pkl', 'wb') as file:
    pickle.dump(symbol_mappings, file)

print("Saved pickles/symbol_mappings.pkl")
