from custom.EnvironmentGeneration.utils.DefinitionGeneratorUtils import *
from os import listdir
from os.path import isfile, join
import pickle
import os


# DefinitionGenerator
def generate_game_definition(graph_instances_lst, definition_dic, obstacle_dic, MOL_IDX):
    mol_keys_dic = definition_dic["mol_keys_dic"]
    molecules_G = graph_instances_lst[MOL_IDX]
    get_final_mol(molecules_G)

    mol_ixs = []
    for m in molecules_G["molecules"]:
        mol_ixs.append(m["molecule_ix"])

    rules = {}
    for ix in mol_ixs:
        rules[ix] = (mol_keys_dic[ix]["final_sym"], mol_keys_dic[ix]["molecule_rules"])

    only_rules = extract_only_rules(rules)

    mol_obs_rules = molecule_obstacle_rules(rules, obstacle_dic)

    stepback_rules = create_combinations(only_rules)

    number_set = set()
    for mol in rules.values():
        _, mol = mol
        for key, value in mol.items():
            number_set.add(key[0])
            number_set.add(key[1])
            number_set.add(value)

    with open('pickles/symbol_mappings.pkl', 'rb') as file:
        symbol_mappings = pickle.load(file)

    rules_mapped = {}
    for key, value in rules.items():
        new_value = (symbol_mappings.get(value[0], value[0]), {})
        for k, v in value[1].items():
            new_key = tuple([symbol_mappings.get(i, i) for i in k])
            new_value[1][new_key] = symbol_mappings.get(v, v)
        rules_mapped[key] = new_value

    atom_sprites_folder = "../../vgdl/sprites/atom_sprites"
    obstacle_sprites_folder = "../../vgdl/sprites/obstacle_sprites/"
    atom_sprites = [f for f in listdir(atom_sprites_folder) if isfile(join(atom_sprites_folder, f))]
    obstacle_sprites = [f for f in listdir(obstacle_sprites_folder) if isfile(join(obstacle_sprites_folder, f))]
    atom_sprites = sorted(atom_sprites)
    obstacle_sprites = sorted(obstacle_sprites)
    obstacle_symbols = []
    for ix in mol_ixs:
        obstacle_symbols.append(obstacle_dic[ix])
    atom_mols_symbols = list(number_set)
    atom_sprite_mapped = dict(zip(atom_mols_symbols, atom_sprites))
    obstacle_sprite_mapped = dict(zip(obstacle_symbols, obstacle_sprites))

    instances_folder = "generated"
    if not os.path.exists(instances_folder):
        os.makedirs(instances_folder)

    generated_definition = os.path.join(instances_folder, "generated_definition.txt")

    atom_obstacle_mappings = {}
    atom_mols_symbols_obstacles = set(obstacle_symbols).union(atom_mols_symbols)
    for idx in atom_mols_symbols_obstacles:
        atom_obstacle_mappings[idx] = symbol_mappings[idx]

    final_mol_sym, _ = rules[get_final_mol(molecules_G)]

    text = create_termination_set(atom_obstacle_mappings, obstacle_sprite_mapped, atom_sprite_mapped, only_rules,
                                  stepback_rules, mol_obs_rules, final_mol_sym, generated_definition)

    write_lines_to_file(generated_definition, text)
