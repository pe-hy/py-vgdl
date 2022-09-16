# first line: 141
@memory.cache
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
    return new_goal.previous_actions
