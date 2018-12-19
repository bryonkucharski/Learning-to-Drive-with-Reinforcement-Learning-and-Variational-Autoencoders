import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from collections import deque
import random
from copy import deepcopy



from memory import SequentialMemory


USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)



class ExpcerienceReplay():
    """
    Note for future Bryon: 

    Do not use deque to implement the memory. This data structure may seem convenient but
    it is way too slow on random access. Instead, we use our own ring buffer implementation.
    """
    def __init__(self, size, batch_size):
        self.max_memory = size
        self.memory = deque(maxlen=size)
        self.batch_size = batch_size

    def sample(self,replacement = True):
        sample = random.sample(self.memory, self.batch_size)
        return sample

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def memory_length(self):
        return len(self.memory)
    
    def reset(self):
        self.memory.clear()
    def sample_and_split(self):

        sample = self.sample()

        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in sample:
            state0_batch.append(e[0])
            state1_batch.append(e[3])
            reward_batch.append(e[2])
            action_batch.append(e[1])
            terminal1_batch.append(0. if e[4] else 1.)

        # Prepare and validate parameters.
        state0_batch = np.array(state0_batch).reshape(self.batch_size,-1)
        state1_batch = np.array(state1_batch).reshape(self.batch_size,-1)
        terminal1_batch = np.array(terminal1_batch).reshape(self.batch_size,-1)
        reward_batch = np.array(reward_batch).reshape(self.batch_size,-1)
        action_batch = np.array(action_batch).reshape(self.batch_size,-1)

        return state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch

class Ornstein_Uhlenbeck():
    def __init__(self, mu = 0.0, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=1)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class DDPG():

    def __init__(self, env,policy,gamma,tau,epsilon,epsilon_decay, actor_lr, critic_lr,theta,sigma,mu,buffer_size):
        
        #self.num_states = num_states
        #self.num_actions = num_actions
        #self.is_training = False
        self.env = env

        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.buffer_size = buffer_size

        self.policy = policy
        self.actor = policy.actor
        self.critic = policy.critic
        self.actor_target = policy.actor_target
        self.critic_target = policy.critic_target
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr )
        self.critic_optim  =  optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.criterion = nn.MSELoss()


        
        #the actor/actor_target and critic/critic_target need to have the same weights to start with 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(),self.critic.parameters()):
            target_param.data.copy_(param.data)
        


        self.memory = SequentialMemory(limit=self.buffer_size, window_length=1)
        #self.replay = ExpcerienceReplay(BUFFER_SIZE,BATCH_SIZE)

        self.ou_noise = Ornstein_Uhlenbeck(theta = self.theta,sigma = self.sigma,mu = self.mu)

        if USE_CUDA: self.cuda()

    def update(self):
        
        s, a, r, s_, done = self.memory.sample_and_split(64)
        #turn all numpy arrays into pytorch variables
        s = Variable(torch.from_numpy(s), requires_grad=False).type(FLOAT)
        a = Variable(torch.from_numpy(a),  requires_grad=False).type(FLOAT)
        s_ = Variable(torch.from_numpy(s_), requires_grad=True).type(FLOAT)
        r =  Variable(torch.from_numpy(r),  requires_grad=False).type(FLOAT)
        done =  Variable(torch.from_numpy(done), requires_grad=False).type(FLOAT)
        #get target q value

        q = self.critic_target(s_, self.actor_target(s_,))
    
        q_target_batch = r + self.gamma * done * q

        #update Critic my minimizing MSE Loss
        self.critic.zero_grad()
        q_batch = self.critic(s,a)
        L = self.criterion(q_batch,q_target_batch)
        L.backward()
        self.critic_optim.step()

        #update Actor by using the sampled policy gradient
        self.actor.zero_grad()
        policy_loss = -self.critic(s, self.actor(s))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        #update targets for the target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau
        )

        for target_param, param in zip(self.critic_target.parameters(),self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau
        )

        self.epsilon -= self.epsilon_decay

    def remember(self, s,a,r, s_, done):
        #self.replay.remember(s,a,r,s_,done)
        self.memory.append(s,a,r,done)

    def select_random_action(self):
        return np.random.uniform(low=[0,-1], high=[1,1], size=(2,))

    def select_action(self, s):
        self.eval_mode()
        s = Variable(torch.from_numpy(s), volatile=False, requires_grad=False).type(FLOAT)
        noise = Variable(torch.from_numpy(self.ou_noise.sample()), volatile=False, requires_grad=False).type(FLOAT)
        noise = self.epsilon *noise
        #noise = Variable(torch.from_numpy(np.random.normal(0,0.02, size=self.env.action_space.shape[0])), volatile=False, requires_grad=False).type(FLOAT)
        #s  = torch.FloatTensor(s).to(device)
        #print(s.size())
        #s.view(1, -1)
        action_pytorch = self.actor(s).squeeze(0)
        action = action_pytorch + noise
        #print(action_pytorch, action)
        action = action_pytorch.cpu().data.numpy() if USE_CUDA else action_pytorch.data.numpy()
        action[0] = np.clip(action[0], 0., 1.)
        action[1] = np.clip(action[1], -1., 1.)
        self.train_mode()
        return action

    def get_return(self, trajectory):
        """
        Calcualte discounted future rewards base on the trajectory of an entire episode
        """
        r = 0.0
        for i in range(len(trajectory)):
            r += self.gamma ** i * trajectory[i]
        return r

    def reset(self):
        self.ou_noise.reset()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def eval_mode(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
    
    def train_mode(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def seed(self,s):
        
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)

    def save(self,PATH):
        self.policy.save(PATH)

    def load(self,PATH):
        self.policy.load(PATH)
    