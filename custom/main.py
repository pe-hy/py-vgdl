import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import gym
import vgdl.interfaces.gym as vdgym
import vgdl
import numpy as np
import gnwrapper
import matplotlib.pyplot as plt
import itertools
import heapq as hq
from dataclasses import dataclass, field
from typing import Any, List, Tuple

vdgym.register_sample_games()
from itertools import product
import networkx as nx
from joblib import Memory
import imageio
import os

cachedir = "../cache"
memory = Memory(cachedir, verbose=0)


def get_sprite_positions(data, env):
    pos_lst = []
    block_size = env.game.block_size
    for i in data:
        rect = i['state']['rect']
        pos = (rect.top, rect.left)
        pos_lst.append(pos)
    return pos_lst


@dataclass(order=True)
class Subgoal:
    priority: int
    previous_actions: list
    info: Any = field(compare=False)


class PriorityQueue:
    def __init__(self):
        self.elements: List[Subgoal] = []

    def empty(self) -> bool:
        return not self.elements

    def put(self, item: Subgoal):
        hq.heappush(self.elements, item)

    def get(self) -> Subgoal:
        return hq.heappop(self.elements)

    def merge(self, items: List):
        for item in items:
            hq.heappush(self.elements, item)


# @memory.cache
def shortest_paths_length(G):
    dic = dict(nx.all_pairs_shortest_path_length(G))
    return dic


def get_distances(env):
    h, w = env.game.height, env.game.width
    bs = env.game.block_size
    grid = np.zeros((h * bs, w * bs))
    for w in env.game.get_game_state().data['sprites']['wall']:
        pos = w['state']['rect'].top, w['state']['rect'].left
        for i in range(-bs + 1, bs):
            for j in range(-bs + 1, bs):
                x = pos[0] + i
                y = pos[1] + j
                if x >= 0 and y >= 0:
                    grid[x, y] = 1
    coor = np.array(list(product(*map(range, grid.shape))))
    G = nx.grid_2d_graph(*grid.shape)

    G.remove_nodes_from(map(tuple, coor[grid.flatten() == 1]))
    return shortest_paths_length(G)


def get_bad_lst(env):
    temp = []
    temp_dict = env.game.sprite_registry.stypes  # returns all stypes in a dict
    for k, v in temp_dict.items():
        e = "enemy"
        if e in v:
            temp.append(v[2:])

    ret = [n for ret in temp for n in ret]
    # Returns a list of all stypes that contain "enemy" in their values.
    return ret


def evaluate_state(env, distances_dict, sub_goal):
    block_size = env.game.block_size
    bad_lst = get_bad_lst(env)
    bad_pos_lst = []
    pos = env.game.get_game_state().data['sprites'][sub_goal]
    pos = filter(lambda x: x["state"].data["alive"], pos)
    print("pos", pos)
    sub_goal_pos_lst = get_sprite_positions(pos, env)
    print("subgoal pos list: ", sub_goal_pos_lst)
    rect = env.game.get_avatars()[-1].rect
    agent_pos = rect.top, rect.left
    print("agent_pos: ", agent_pos, "subgoal: ", sub_goal)
    for i in bad_lst:
        pos = env.game.get_game_state().data['sprites'][i]
        bad_pos_lst += get_sprite_positions(pos, env)
    sub_goal_distances = []
    for i in sub_goal_pos_lst:
        if i in distances_dict[agent_pos]:
            sub_goal_distances.append(distances_dict[agent_pos][i])
    enemy_distances = [40]
    print("subgoal distances: ", sub_goal_distances)
    for i in bad_pos_lst:
        if i in distances_dict[agent_pos]:
            enemy_distances.append(distances_dict[agent_pos][i])
    if(len(sub_goal_distances) == 0):
        min_sub_goal_dist = 0
    else:
        min_sub_goal_dist = min(sub_goal_distances)

    min_enemy_dist = min(enemy_distances)

    if min_enemy_dist < 15:
        score = 10 ** 6
    else:
        score = 2 * min_sub_goal_dist - min_enemy_dist

    return score, min_sub_goal_dist, min_enemy_dist

def get_min_enemy(env, distances_dict):
    rect = env.game.get_avatars()[-1].rect
    agent_pos = rect.top, rect.left
    bad_lst = get_bad_lst(env)
    bad_pos_lst = []
    for i in bad_lst:
        pos = env.game.get_game_state().data['sprites'][i]
        bad_pos_lst += get_sprite_positions(pos, env)
    enemy_distances = [40]
    for i in bad_pos_lst:
        if i in distances_dict[agent_pos]:
            enemy_distances.append(distances_dict[agent_pos][i])

    return min(enemy_distances)

env = gym.make('vgdl_zelda-v0', obs_type="objects")
obs = env.reset()
first_state_info = env.game.get_game_state()

# @memory.cache
def compute_plan():
    first_state = Subgoal(priority=0, previous_actions=[], \
                          info=first_state_info)
    all_distances = get_distances(env)
    print('distances computed')
    frontier = PriorityQueue()
    frontier.put(first_state)
    seen_states = {first_state_info}
    switched = False
    finish = False
    subgoals = [("key", "withkey1"), ("key", "withkey2"), ("key", "withkey3"), ("goal", None)]
    avatar_condition = "nokey"
    for i in range(1000):
        if finish:
            break
        if len(frontier.elements) == 0:
            # print(f"no more subgoals on {i}th iteration")
            break
        # else: print('frontier size:',len(frontier.elements))
        current = frontier.get()
        state = current.info
        env.game.set_game_state(state)
        for i, a in enumerate(env.game.get_possible_actions()):
            next_obs, reward, done, info = env.step(i)
            if done and reward < 0:
                env.game.set_game_state(state)
                print('death', "*" * 20)
                print(reward)
                continue
            if not env.game.get_game_state().avatar_state['state']['alive']:
                env.game.set_game_state(state)
                print('death', "*" * 20)
                continue  # here should be continue...check for win
            new_state = env.game.get_game_state()

            if new_state in seen_states:
                env.game.set_game_state(state)
                continue
            seen_states.add(new_state)
            actions = current.previous_actions + [i]
            avatar_name = env.game.get_avatars()[-1].key
            if(avatar_name == subgoals[0][1]):
                subgoals.pop(0)
                frontier = PriorityQueue()
                seen_states = {new_state}
                new_goal = Subgoal(0, actions, new_state)
                frontier.put(new_goal)
                print("*" * 50)
                break

            subgoal = subgoals[0][0]
            score, min_subgoal, min_enemy = evaluate_state(env, all_distances, subgoal)
            print(score, min_subgoal, min_enemy)
            new_goal = Subgoal(score, actions, new_state)
            if done:
                finish = True
                break
            frontier.put(new_goal)

            env.game.set_game_state(state)

            if env.game.get_game_state().ended():
                tmp_state = new_goal
                print('done', '*' * 20)
                finish = True
                break
    return new_goal.previous_actions, all_distances


def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0:
        if not err:
            print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


def save_gif(env, action_list, current_state, all_distances):
    env = gym.make('vgdl_zelda-v0', obs_type="objects")
    env.reset()
    env.game.set_game_state(current_state)
    frames = []
    for j, i in enumerate(action_list):
        print(j)
        min_enemy = get_min_enemy(env, all_distances)
        print(min_enemy)
        if min_enemy < 20:
            next_obs, reward, done, info = env.step(5)
        rgb = env.render(mode='rgb_array')
        frame = repeat_upsample(rgb, 8, 8)
        frames.append(frame)
        next_obs, reward, done, info = env.step(i)
        print(done)
        print(reward)
    env.close()
    imageio.mimwrite(os.path.join('./videos/', 'last_run.gif'), frames, fps=60)


def main():
    # visualize_actions(env, [], first_state_info)
    actions, dst = compute_plan()
    save_gif(env, actions, first_state_info, dst)
    # visualize_actions(env,actions, first_state_info,dst)
    # plt.show()


if __name__ == "__main__":
    main()
