import gym
import vgdl.interfaces.gym as vdgym
## https://pypi.org/project/gym-notebook-wrapper/

def main():
    game = gym.make('vgdl_zelda-v0')
    game.reset()
    game.render(mode="human")

if __name__ == "__main__":
    main()
