from duckietown.gym_duckietown.envs.duckietown_env  import DuckietownEnv
import numpy as np
from ddpg import DDPG
from wrappers import *
from policies import CNNPolicy, FCPolicy
import matplotlib.pyplot as plt
from copy import deepcopy


from hyperdash import Experiment

L2_WEIGHT_DECAY = 0.1
ACTOR_LR = 0.0001
CRITIC_LR = 0.001
GAMMA = 0.99
TAU = 0.001
HIDDEN_LAYERS_1 = 400
HIDDEN_LAYERS_2 = 300
BATCH_SIZE = 32
BUFFER_SIZE = 1000000
ACTOR_WEIGHT_SCALE = 3e-3
CRITIC_WEIGHT_SCALE = 3e-4

EPSILON = 1.0
EPSILON_DECAY = 1e-6

#Ornstein_Uhlenbeck parameters
THETA = 0.15
SIGMA = 0.2
MU = 0


exp = Experiment("duckietown_ddpg")


def train(env, agent, num_iterations, warm_up, max_episode_length=None, debug=False, update_freq = 5000,eval_episodes = 5):
    rewards = []
    eval_rewards = []
    step = episode = episode_steps = 0
    episode_reward = 0.
    s = None
    while step < num_iterations:
        # reset if it is the start of episode
        if s is None:
            s = deepcopy(env.reset()).reshape(1,-1)
            agent.reset()

        # agent pick action ...
        #if step <= warm_up:
            #action = agent.select_random_action()
        #else:
        action = agent.select_action(s)

        if action[0] > 0.8:
            action[0] = 0.8

        # env response with next_observation, reward, terminate_info
        s_, reward, done, info = env.step(action)
        print(action, reward)
        s_ = deepcopy(s_).reshape(1,-1)
        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True

        # agent observe and update policy
        # print(s,action,reward, s_, done)
        agent.remember(s, action, reward, s_, done)
        if step > warm_up:
            agent.update()

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        s = deepcopy(s_)

        if done:  # end of episode
            if debug:
                print('#{}: episode_reward:{} steps:{}'.format(episode, episode_reward, step))
                exp.metric("rewards",episode_reward)
            agent.remember(s, agent.select_action(s), 0.0, s_, False)

            # reset
            s = None
            episode_steps = 0
            rewards.append(episode_reward)
            episode_reward = 0.

            episode += 1

        if step % update_freq == 0:
            agent.save("models/duckietown_ddpg_small_loop_move_penalty.pt")
            eval_rewards = evaluate(agent, render = True, eval_episodes = eval_episodes, max_episode_length=max_episode_length)

    env.close()
    exp.end()
    return rewards, eval_rewards


def evaluate(agent, eval_episodes, max_episode_length,render = True):
    rewards = []
    episode_rewards = 0
    for i in range(eval_episodes):
        s = deepcopy(env.reset()).reshape(1,-1)

        agent.reset()
        i = 0
        while True:
            action = agent.select_action(s)

            if action[0] > 0.8:
                action[0] = 0.8

            # env response with next_observation, reward, terminate_info
            s_, reward, done, info = agent.env.step(action)
            print(action,reward)
            episode_rewards += reward
            s_ = deepcopy(s_).reshape(1,-1)


            if max_episode_length and i >= max_episode_length - 1:
                done = True

            i += 1

            if render:
                agent.env.render()

            if done:
                rewards.append(episode_rewards / i)
                break
            s = deepcopy(s_)
    agent.env.close()
    return np.sum(rewards) / eval_episodes

env = DuckietownEnv(
        map_name="loop_empty",
        max_steps=500001,  # we don't want the gym to reset itself
        domain_rand=0,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=4,  # start close to straight
        accept_start_dist = 0.05,
        full_transparency=False,
        distortion=False)

env = ResizeWrapper(env)
env = NormalizeWrapper(env)
env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
env = DtRewardWrapper(env)

nb_states = env.observation_space.shape[0]
nb_actions = env.action_space.shape[0]

'''
policy = CNNPolicy(num_actions = nb_actions,
                   hidden_actor = 512,
                   hidden_critic_1 = 256,
                   hidden_critic_2 = 128,
                   filter_size = 16,
                   kernal_size = 3,
                   stride_size = 2,
                   padding = 0,
                   use_bn = True)
'''

policy = FCPolicy(num_states = 64*64*3,
                  num_actions = nb_actions,
                  actor_weight_scale = ACTOR_WEIGHT_SCALE,
                  critic_weight_scale = CRITIC_WEIGHT_SCALE,
                  hidden_1=400,
                  hidden_2=300,
                use_bn=True)


agent = DDPG(env = env,
             policy = policy,
             gamma = GAMMA,
             tau = TAU,
             epsilon = EPSILON,
             epsilon_decay = EPSILON_DECAY,
             actor_lr = ACTOR_LR,
             critic_lr = CRITIC_LR,
             theta = THETA,
             sigma = SIGMA,
             mu = MU,
             buffer_size = BUFFER_SIZE)


rewards, _ = train(env,agent,num_iterations = 20000, warm_up = 1000, max_episode_length=200, debug=True, update_freq = 1000,eval_episodes = 5)
#agent.load("models/duckietown_ddpg_empty_loop_move_penalty.pt")
#eval_rewards = evaluate(agent,render = True, eval_episodes = 1000, max_episode_length=100)



plt.plot(range(len(rewards)),rewards, color = 'blue', label = 'Train Rewards')
plt.title('Episode Reward')
plt.xlabel("Episode")
plt.ylabel('Reward')
plt.legend()
plt.show()

'''
plt.plot(range(len(eval_rewards)),eval_rewards, color = 'red', label = 'Eval Rewards')
plt.title('Eval Reward')
plt.xlabel("Episode")
plt.ylabel('Reward')
plt.legend()
plt.show()
'''