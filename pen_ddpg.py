from ddpg import *
import gym
from copy import deepcopy
from policies import FCPolicy
import time
import numpy as np
import matplotlib.pyplot as plt




L2_WEIGHT_DECAY = 0.1
ACTOR_LR = 0.0001
CRITIC_LR = 0.001
GAMMA = 0.9
TAU = 0.001
HIDDEN_LAYERS_1 = 400
HIDDEN_LAYERS_2 = 300
BATCH_SIZE = 64
BUFFER_SIZE = 1000000
ACTOR_WEIGHT_SCALE = 3e-3
CRITIC_WEIGHT_SCALE = 3e-4

MAX_EPISODES = 1000
MAX_EP_STEPS = 1000

EPSILON = 1.0
EPSILON_DECAY = 1e-6

#Ornstein_Uhlenbeck parameters
THETA = 0.15
SIGMA = 0.2
MU = 0



class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)


def train(num_iterations, agent, env,warm_up, max_episode_length=None, debug=False):
    rewards = []
    step = episode = episode_steps = 0
    episode_reward = 0.
    s = None
    while step < num_iterations:
        # reset if it is the start of episode
        if s is None:
            s = deepcopy(env.reset())
            agent.reset()

        # agent pick action ...
        if step <= warm_up:
            action = agent.select_random_action()
        else:
            action = agent.select_action(s)
        
        # env response with next_observation, reward, terminate_info
        s_, reward, done, info = env.step(action)
        #print(s_, reward, done, info)
        s_ = deepcopy(s_)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True

        # agent observe and update policy
        #print(s,action,reward, s_, done)
        agent.remember(s,action,reward, s_, done)
        if step > warm_up:
            agent.update()
    


        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        s = deepcopy(s_)

        if done: # end of episode
            if debug: print('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

            agent.remember(s,agent.select_action(s),0.0, s_, False)

            # reset
            s = None
            episode_steps = 0
            rewards.append(episode_reward)
            episode_reward = 0.
            
            episode += 1

    env.close()
    return rewards

env = NormalizedEnv(gym.make("Pendulum-v0").env)
nb_states = env.observation_space.shape[0]
nb_actions = env.action_space.shape[0]



policy = FCPolicy(num_states = nb_states, num_actions = nb_actions, actor_weight_scale = ACTOR_WEIGHT_SCALE,critic_weight_scale = CRITIC_WEIGHT_SCALE ,hidden_1 = HIDDEN_LAYERS_1, hidden_2 = HIDDEN_LAYERS_2,use_bn = True)

agent = DDPG(env =env,
             policy = policy,
             gamma = GAMMA,
             tau = TAU,
             epsilon = EPSILON,
             epsilon_decay = EPSILON_DECAY,
             actor_lr=ACTOR_LR,
             critic_lr = CRITIC_LR,
             theta = THETA,
             sigma = SIGMA,
             mu = MU,
             buffer_size = BUFFER_SIZE)


bn_rewards = agent.train(  num_iterations = 100000,
        warm_up = 100,
        max_episode_length=500,
        debug=True,
        )


plt.plot(range(200),bn_rewards, color = 'blue', label = 'BN')
plt.title('Episode Reward')
plt.xlabel("Episode")
plt.ylabel('Reward')
plt.legend()
plt.show()