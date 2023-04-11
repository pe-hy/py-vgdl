def extract_only_rules(rules):
    only_rules = {}
    for k, v in rules.items():
        i, j = v
        for s1, s2 in j.items():
            only_rules[s1] = s2
    return only_rules


def get_final_mol(mol_G):
    for m in mol_G["molecules"]:
        if m["obstacle_G_ix"] is None:
            return m["molecule_ix"]


def molecule_obstacle_rules(rules, obstacle_dic):  # which molecules destroy obstacle
    mol_obs_rules = {}
    for idx, tup in rules.items():
        idx_o = obstacle_dic[idx]
        final_sym, rule = tup
        mol_obs_rules[idx_o] = final_sym
    return mol_obs_rules


def create_combinations(input_dict):
    result = {}
    for key, value in input_dict.items():
        a, b = key
        if (b, a) not in input_dict:
            result[(b, a)] = value
        if (a, value) not in input_dict:
            result[(a, value)] = b
        if (b, value) not in input_dict:
            result[(b, value)] = a
        if (value, a) not in input_dict:
            result[(value, a)] = b
        if (value, b) not in input_dict:
            result[(value, b)] = a
    return result


def write_lines_to_file(def_file, text):
    with open(def_file, 'w', encoding="utf-8") as file:
        for line in text:
            file.write(line + '\n')


def create_header():
    header = []
    header.append('BasicGame block_size=10')
    header.append('  SpriteSet')
    header.append('    background > Immovable randomtiling=0.9 img=oryx/floor3 hidden=True')
    header.append('    avatar > MovingAvatar img=oryx/knight1')
    header.append('    wall > Immovable autotiling=true img=oryx/wall3')
    return header


def create_sprite_set(atom_sprite_mapped, obstacle_sprite_mapped):
    text = create_header()
    text.append("    movable > ")
    for atom, sprite in atom_sprite_mapped.items():
        text.append("       " + str(atom) + "  " + "> " + "Passive" + " img=atom_sprites" + "/" + sprite)
    text.append("    obstacle > ")
    for obstacle, sprite in obstacle_sprite_mapped.items():
        text.append("       " + str(obstacle) + "  " + "> " + "Immovable" + " img=obstacle_sprites" + "/" + sprite)

    return text


def create_level_mapping(atom_obstacle_mappings, atom_sprite_mapped, obstacle_sprite_mapped):
    text = create_sprite_set(atom_sprite_mapped, obstacle_sprite_mapped)
    text.append("")
    text.append("  LevelMapping")
    text.append("    " + "A" + " > " + "background avatar")
    text.append("    " + "w" + " > " + "wall")
    text.append("    " + "." + " > " + "background")
    for idx, symbol in atom_obstacle_mappings.items():
        text.append("    " + symbol + " > background " + str(idx))

    return text


def create_interaction_set(atom_obstacle_mappings, obstacle_sprite_mapped, atom_sprite_mapped, only_rules,
                           stepback_rules, mol_obs_rules):
    text = create_level_mapping(atom_obstacle_mappings, atom_sprite_mapped, obstacle_sprite_mapped)
    text.append("")
    text.append("  InteractionSet")
    text.append("    avatar wall > stepBack")
    text.append("    movable avatar > bounceForward")
    text.append("    movable wall > stepBack")
    text.append("    avatar obstacle > stepBack")
    for tup, res in only_rules.items():
        s1, s2 = tup
        # 17 19 > createSprite stype="10053"
        text.append("    " + str(s1) + " " + str(s2) + " > createSprite stype=\"" + str(res) + "\"")

    for mol_idx, obs_idx in mol_obs_rules.items():
        text.append("    " + str(obs_idx) + " " + str(mol_idx) + " > killBoth")
    text.append("    movable obstacle > stepBack")
    # explicitní pravidla movable movable > stepBack
    # for tup, res in stepback_rules.items():
    #     s1, s2 = tup
    #     text.append("    " + str(s1) + " " + str(s2) + " > stepBack")
    return text


def create_termination_set(atom_obstacle_mappings, obstacle_sprite_mapped, atom_sprite_mapped, only_rules,
                           stepback_rules, mol_obs_rules, final_mol, generated_definition):
    text = create_interaction_set(atom_obstacle_mappings, obstacle_sprite_mapped, atom_sprite_mapped, only_rules,
                                  stepback_rules, mol_obs_rules)
    text.append("")
    text.append("  TerminationSet")
    text.append("    MultiSpriteCounter stype=" + "\"" + str(final_mol) + "\"" + " limit=1 win=True")
    text.append("    Timeout limit=20000 win=False")
    write_lines_to_file(generated_definition, text)
    print("definition saved: generated_definition.txt")
    return text
