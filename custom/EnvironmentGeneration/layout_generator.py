import os
from custom.EnvironmentGeneration.utils.LayoutUtils import *
import pickle

def generate_layout(graph_instances_lst, instances_num, MOL_IDX, num_columns, grid_size, min_room_size):
    for instance in range(instances_num):
        molecules_G = graph_instances_lst[MOL_IDX]
        node_data, num_rooms, group_sizes, group_ids, groups_to_connect, obstacle_ids = load_parse_molecules(
            molecules_G)
        roomspace = grid_size - (num_columns + 1)

        divisions = get_divisions(roomspace, num_columns, min_room_size)
        walkable_points, chosen_rooms = get_choosen_rooms(divisions, min_room_size, num_rooms)

        distances = {}
        for i, r1 in enumerate(chosen_rooms):
            for j, r2 in enumerate(chosen_rooms):
                if i < j:
                    distances[(i, j)] = get_distance(r1, r2)

        sorted_pairs = list(sorted(map(lambda x: (x[0], x[1]), distances.items()), key=lambda x: x[1]))
        G = get_grid_graph(grid_size, walkable_points, chosen_rooms)
        paths = get_paths(G, sorted_pairs, chosen_rooms, grid_size)
        paths_lst = [(k, v) for k, v in paths.items()]
        rooms_points = get_room_points(chosen_rooms)
        room_G = get_rooms_graph(chosen_rooms, paths_lst, rooms_points)

        available_indices = set(range(num_rooms))  # num rooms=součet

        splits_it = iterate_G(available_indices, group_sizes, 0, room_G)

        found = False
        for i, split in enumerate(splits_it):
            skels_it = get_possible_split(split)
            for skels in skels_it:
                if len(skels) == len(group_sizes):
                    crossing_edges = get_crossing_edges_list(skels, room_G, groups_to_connect, paths_lst)
                    found = True
                    break
            if found == True:
                break

        vertices_edges_dict = nx.get_edge_attributes(room_G, 'corridor_key')

        skels_rooms_dict = get_rooms_from_skels(skels, vertices_edges_dict)

        room_ids_from_node_data = get_room_ids_for_group(skels_rooms_dict)
        obstacles = []
        idx = crossing_edges[0][0]
        grps = crossing_edges[0][1]
        intersect = intersect_first_group(idx, grps, room_G, skels, paths_lst)
        obstacle_room_id = get_obstacle_room(idx, grps, room_G, skels, paths_lst, room_ids_from_node_data)
        obstacle_pos = get_obstacle_pos(obstacle_room_id, idx, paths_lst, room_G)
        groups_rooms_points = get_room_points_from_node_data(room_ids_from_node_data, room_G)
        idx = crossing_edges[1][0]
        groups = crossing_edges[1][1]
        ob_room = get_obstacle_room(idx, groups, room_G, skels, paths_lst, room_ids_from_node_data)
        ob_pos = get_obstacle_pos(ob_room, idx, paths_lst, room_G)

        obstacles = get_obstacle_id_and_pos(crossing_edges, room_G, skels, paths_lst, room_ids_from_node_data,
                                            obstacle_ids)

        atom_placements = get_atom_placements(groups_rooms_points, node_data, obstacles)

        with open('pickles/obstacle_dic.pkl', 'rb') as file:
            obstacle_dic = pickle.load(file)
        with open('pickles/symbol_mappings.pkl', 'rb') as file:
            symbol_mappings = pickle.load(file)

        atom_obstacles_dict = get_atom_obstacle_ids_and_positions(atom_placements, obstacles, obstacle_dic)

        atom_obstacles_dict_converted = convert_to_ascii(atom_obstacles_dict, symbol_mappings)

        crossing_edge_this = [c_e[0] for c_e in crossing_edges]
        agent_pos = get_agent_pos(groups_rooms_points, molecules_G)

        layout = get_layout_str_with_atoms(walkable_points, skels, crossing_edge_this, paths_lst, grid_size,
                                           atom_placements, obstacles, atom_obstacles_dict_converted, agent_pos)
        instances_folder = "generated"
        if not os.path.exists(instances_folder):
            os.makedirs(instances_folder)

        filename = os.path.join(instances_folder, str(instance) + "_generated_layout.txt")

        with open(filename, 'w', encoding="utf-8") as file:
            file.write(layout)

        print("layout saved: ", filename)
