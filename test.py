from nn_class import nn
from helper_functions import *
import gym
from reinforce_class import reinforce

obs_space ,action_space = 4,2
render = False

reinf = reinforce(obs_space ,action_space)
reinf.create_layer('Input',Relu)
reinf.create_layer(128 ,softmax)
reinf.create_model()

def finish_episode(reward_book):
    reinf.calc_discounted_reward(reward_book)

def main():
    reward_book = []
    for i_episode in range(20):
        obs = env.reset()
        for t in range(100):
            if render==True : env.render()
            action = select_action(obs)
            # print('action',action)
            obs, reward, done, _ = env.step(np.argmax(action))
            obs.shape = (4,1)
            reward_book.append(reward)
            finish_episode(reward_book)
            print("Episode finished after {} timesteps".format(t+1))


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
