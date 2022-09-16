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
from typing import Any,List,Tuple
vdgym.register_sample_games()
from itertools import product
import networkx as nx
from joblib import Memory

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
    info: Any=field(compare=False)


class PriorityQueue:
    def __init__(self):
        self.elements: List[Subgoal] = []

    def empty(self) -> bool:
        return not self.elements

    def put(self, item: Subgoal):
        hq.heappush(self.elements,item)

    def get(self) -> Subgoal:
        return hq.heappop(self.elements)

    def merge(self,items: List):
        for item in items:
            hq.heappush(self.elements,item)

#@memory.cache
def shortest_paths_length(G):
    dic = dict(nx.all_pairs_shortest_path_length(G))
    return dic

def get_distances(env):
    h,w = env.game.height,env.game.width
    bs = env.game.block_size
    grid = np.zeros((h*bs,w*bs))
    for w in env.game.get_game_state().data['sprites']['wall']:
        pos = w['state']['rect'].top,w['state']['rect'].left
        for i in range(-bs+1,bs):
            for j in range(-bs+1,bs):
                x = pos[0]+i
                y = pos[1]+j
                if x>=0 and y>=0:
                    grid[x,y]=1
    coor = np.array(list(product(*map(range, grid.shape))))
    G = nx.grid_2d_graph(*grid.shape)

    G.remove_nodes_from(map(tuple, coor[grid.flatten() == 1]))
    return shortest_paths_length(G)



def evaluate_state(env, distances_dict):
    block_size = env.game.block_size
    bad_lst = ['monsterQuick', 'monsterNormal', 'monsterSlow']
    avatar_name = env.game.get_avatars()[-1].key
    if avatar_name == 'withkey2':
        sub_goal = 'goal'
    else:
        sub_goal = 'key'
    bad_pos_lst = []
    pos = env.game.get_game_state().data['sprites'][sub_goal]
    sub_goal_pos_lst = get_sprite_positions(pos, env)
    rect = env.game.get_avatars()[-1].rect
    agent_pos = rect.top, rect.left
    print(agent_pos)
    for i in bad_lst:
        pos = env.game.get_game_state().data['sprites'][i]
        bad_pos_lst += get_sprite_positions(pos, env)
    sub_goal_distances = []
    for i in sub_goal_pos_lst:
        if i in distances_dict[agent_pos]:
            sub_goal_distances.append(distances_dict[agent_pos][i])
    enemy_distances = [40]
    for i in bad_pos_lst:
        if i in distances_dict[agent_pos]:
            enemy_distances.append(distances_dict[agent_pos][i])

    min_sub_goal_dist = min(sub_goal_distances)
    min_enemy_dist = min(enemy_distances)
    if min_enemy_dist < 15:
        score = 10 ** 6
    else:
        score = 2 * min_sub_goal_dist - min_enemy_dist

    return score, min_sub_goal_dist, min_enemy_dist

def visualize_actions(env,action_list, current_state,all_distances):
    fig, ax = plt.subplots()
    env = gym.make('vgdl_zelda-v0', obs_type="objects")
    env.reset()
    env.game.set_game_state(current_state)
    print(len(action_list))
    for j,i in enumerate(action_list):
        print(j)

        score, min_subgoal, min_enemy = evaluate_state(env, all_distances)
        print(min_enemy)
        if min_enemy < 20:
            next_obs, reward, done, info = env.step(5)
        next_obs, reward, done, info = env.step(i)
        print(done)
        print(reward)
        a = env.render(mode='rgb_array')
        ax.cla()
        ax.imshow(a)
        plt.pause(0.01)
    return env

env = gym.make('vgdl_zelda-v0', obs_type="objects")
obs = env.reset()
first_state_info = env.game.get_game_state()


#@memory.cache
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
            score, min_subgoal, min_enemy = evaluate_state(env, all_distances)
            if len(env.game.get_avatars()) == 2 and not switched:
                frontier = PriorityQueue()
                seen_states = {new_state}
                switched = True
                new_goal = Subgoal(score, actions, new_state)
                frontier.put(new_goal)
                print("*" * 50)
                break
            else:
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
    return new_goal.previous_actions,all_distances
def main():

    #visualize_actions(env, [], first_state_info)
    actions,dst = compute_plan()

    visualize_actions(env,actions, first_state_info,dst)
    #plt.show()
if __name__ == "__main__":
    main()