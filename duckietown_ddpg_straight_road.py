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


#exp = Experiment("duckietown_ddpg_straight_road_no_vae")
model_name = 'models/duckietown_ddpg_straight_road_move_penalty.pt'


def train(env, agent, num_iterations, warm_up, max_episode_length=None, debug=False, update_freq = 5000,eval_episodes = 5):
    rewards = []
    eval_rewards = []
    step = episode = episode_steps = 0
    episode_reward = 0.
    total_reward = 0
    average_rewards = []
    returns = []
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


        #print(action, reward)
        s_ = deepcopy(s_).reshape(1,-1)
        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True

        # agent observe and update policy
        # print(s,action,reward, s_, done)
        agent.remember(s, action, reward, s_, done)

        if step > warm_up:
            agent.update()
            total_reward += reward
            average_rewards.append(total_reward / (step + 1))

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward

        s = deepcopy(s_)

        if done:  # end of episode
            if debug:
                print('#{}: episode_reward:{} steps:{} return: {}'.format(episode, episode_reward, step, total_reward / (step + 1)))
            #exp.metric("rewards",episode_reward)
            agent.remember(s, agent.select_action(s), 0.0, s_, False)
            # reset
            s = None
            episode_steps = 0
            episode_reward = 0.
            #returns.append(agent.get_return(rewards))


            episode += 1


        if step % update_freq == 0:
            agent.save(model_name)
            #eval_reward = evaluate(agent,render = True, eval_episodes = eval_episodes, max_episode_length=max_episode_length)

    env.close()
    #exp.end()
    return average_rewards

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
        map_name="straight_road",
        max_steps=500001,  # we don't want the gym to reset itself
        domain_rand=0,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=1,  # start close to straight
        accept_start_dist  = 0.1,
        full_transparency=False,
        distortion=False)

env = ResizeWrapper(env)
env = NormalizeWrapper(env)
env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
env = DtRewardWrapper(env)

nb_states = env.observation_space.shape[0]
nb_actions = env.action_space.shape[0]
NUM_EPISODES = 20000
NUM_TRIALS = 1
average_data = [[] for _ in range(NUM_EPISODES)]

for i in range(NUM_TRIALS):
    print(i)
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


    #returns = train(env,agent,num_iterations = 20000, warm_up = 100, max_episode_length=100, debug=True, update_freq = 1000,eval_episodes = 5)
    agent.load(model_name)
    eval_rewards = evaluate(agent,render = True, eval_episodes = 100, max_episode_length=200)
    '''
    for j in range(19899):
        average_data[j].append(returns[j])

    average_list = []
    std_list = []

    for i in range(len(average_data)):
        current_list = np.array(average_data[i]).astype(np.float)
        mean = np.mean(current_list)
        average_list.append(mean)
        std = np.std(current_list)
        std_list.append(std)
    average_list = np.array(average_list)
    std_list = np.array(std_list)
    '''
'''
np.save('duckietown_ddpg_straight_road_averages', average_list)
np.save('duckietown_ddpg_straight_road_stds', std_list)

straight_average = np.load('duckietown_ddpg_straight_road_averages.npy')
straight_std = np.load('duckietown_ddpg_straight_road_stds.npy')
vae_average = np.load('duckietown_ddpg_vae_straight_road_averages.npy')
vae_std = np.load('duckietown_ddpg_vae_straight_road_stds.npy')

plt.plot(np.arange(NUM_EPISODES), straight_average, color='#16af08', label = 'DDPG+RawImages')
plt.fill_between(np.arange(NUM_EPISODES), straight_average - straight_std, straight_average + straight_std, alpha=0.5, edgecolor='#16af08', facecolor='#7df473')

plt.plot(np.arange(NUM_EPISODES), vae_average, color='#e52029', label = 'DDPG+VAE')
plt.fill_between(np.arange(NUM_EPISODES), vae_average - vae_std, vae_average + vae_std, alpha=0.5, edgecolor='#e52029', facecolor='#ff666c')


plt.title("Straight Road DDPG Duckietown")
plt.xlabel("Steps")
plt.ylabel("Average Rewad")

plt.legend()
plt.show()


np.save('duckietown_ddpg_straight_road_single_trial', returns)
plt.plot(range(returns),returns)
plt.title('Average Reward vs Episode')
plt.xlabel("Steps")
plt.ylabel('Average Reward')
plt.legend()
plt.show()
'''
