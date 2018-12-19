import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from collections import deque
import random
from copy import deepcopy


class ActorFC(nn.Module):
    def __init__(self, num_states, num_actions, hidden_1, hidden_2, weight_scale, use_bn=True):
        """
        From the DDPG Paper:

        The neural networks used the rectified non-linearity  for all hidden layers. The final
        output layer of the actor was a tanh layer, to bound the actions. The low-dimensional networks
        had 2 hidden layers with 400 and 300 units respectively

        """

        super(ActorFC, self).__init__()

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(num_states, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, num_actions)
        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.bn2 = nn.BatchNorm1d(hidden_2)
        self.input_bn = nn.BatchNorm1d(num_states)

        #self.initialize_weights(weight_scale)
        self.use_bn = use_bn

    def initialize_weights(self, scale):
        '''
        From the paper

        The final layer weights and biases of both the actor and critic were initialized from a uniform distribution [−3 × 10−3, 3 × 10−3] and [3 × 10−4, 3 × 10−4] for the
        low dimensional and pixel cases respectively. This was to ensure the initial outputs for the policy
        and value estimates were near zero. The other layers were initialized from uniform distributions [− sqrt(1/f), sqrt(1/f) where f is the fan-in of the layer.

        '''

        self.fc1_fanin = np.sqrt(1 / self.fc1.weight.data.size()[0])
        self.fc1.weight.data = torch.Tensor(self.fc1.weight.data.size()).uniform_(-self.fc1_fanin, self.fc1_fanin)
        # self.fc1.weight.data.uniform_(-self.fc1_fanin,self.fc1_fanin)

        self.fc2_fanin = np.sqrt(1 / self.fc2.weight.data.size()[0])
        self.fc2.weight.data = torch.Tensor(self.fc2.weight.data.size()).uniform_(-self.fc2_fanin, self.fc2_fanin)
        # self.fc2.weight.data.uniform_(-self.fc2_fanin,self.fc2_fanin)

        self.fc3.weight.data.uniform_(-scale, scale)

    def forward(self, state):
        #state = state.view(state.size(0),-1)
        #print(state.size())
        if self.use_bn:
            state = self.input_bn(state)

        z1 = self.fc1(state)
        a1 = self.relu(z1)

        if self.use_bn:
            a1 = self.bn1(a1)

        z2 = self.fc2(a1)
        a2 = self.relu(z2)

        if self.use_bn:
            a2 = self.bn2(a2)

        z3 = self.fc3(a2)
        z3[:, 0] = self.sigmoid(z3[:, 0])
        z3[:, 1] = self.tanh(z3[:, 1])

        return z3

class ActorCNN(nn.Module):
    def __init__(self, num_actions, hidden_units, filter_size,kernal_size,stride_size,padding, use_bn=True):
        """
        From the DDPG Paper:

        The neural networks used the rectified non-linearity  for all hidden layers. The final
        output layer of the actor was a tanh layer, to bound the actions. The low-dimensional networks
        had 2 hidden layers with 400 and 300 units respectively

        """

        super(ActorCNN, self).__init__()

        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()



        #self.initialize_weights(weight_scale)
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(3,           filter_size,   kernel_size = kernal_size, stride=stride_size,padding = padding)
        self.conv2 = nn.Conv2d(filter_size, filter_size,  kernel_size = kernal_size, stride=stride_size, padding = padding)
        self.conv3 = nn.Conv2d(filter_size, filter_size,  kernel_size = kernal_size, stride=stride_size, padding = padding)
        self.conv4 = nn.Conv2d(filter_size, filter_size,  kernel_size = kernal_size, stride=stride_size, padding = padding)

        self.dropout = nn.Dropout(.5)

        #TODO: add state vector here
        flattened_size =144#hardcoded oops

        self.fc1 = nn.Linear(flattened_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, num_actions)

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(filter_size)
            self.bn2 = nn.BatchNorm2d(filter_size)
            self.bn3 = nn.BatchNorm2d(filter_size)
            self.bn4 = nn.BatchNorm2d(filter_size)

    def initialize_weights(self, scale):
        '''
        From the paper

        The final layer weights and biases of both the actor and critic were initialized from a uniform distribution [−3 × 10−3, 3 × 10−3] and [3 × 10−4, 3 × 10−4] for the
        low dimensional and pixel cases respectively. This was to ensure the initial outputs for the policy
        and value estimates were near zero. The other layers were initialized from uniform distributions [− sqrt(1/f), sqrt(1/f) where f is the fan-in of the layer.

        '''

        self.fc1_fanin = np.sqrt(1 / self.fc1.weight.data.size()[0])
        self.fc1.weight.data = torch.Tensor(self.fc1.weight.data.size()).uniform_(-self.fc1_fanin, self.fc1_fanin)
        # self.fc1.weight.data.uniform_(-self.fc1_fanin,self.fc1_fanin)

        self.fc2_fanin = np.sqrt(1 / self.fc2.weight.data.size()[0])
        self.fc2.weight.data = torch.Tensor(self.fc2.weight.data.size()).uniform_(-self.fc2_fanin, self.fc2_fanin)
        # self.fc2.weight.data.uniform_(-self.fc2_fanin,self.fc2_fanin)

        self.fc3.weight.data.uniform_(-scale, scale)

    def forward(self, state):
        z1 = self.conv1(state)
        a1 = self.leaky_relu(z1)
        if self.use_bn:
            a1 = self.bn1(a1)

        z2 = self.conv2(a1)
        a2 = self.leaky_relu(z2)
        if self.use_bn:
            a2 = self.bn2(a2)

        z3 = self.conv3(a2)
        a3 = self.leaky_relu(z3)
        if self.use_bn:
            a3 = self.bn3(a3)

        z4 = self.conv4(a3)
        a4 =  self.leaky_relu(z4)
        if self.use_bn:
            a4 = self.bn4(a4)

        a4_flat = a4.view(a4.size(0), -1)  # flatten

        a4_flat = self.dropout(a4_flat)

        z5 = self.fc1(a4_flat)
        a5 = self.leaky_relu(z5)

        output = self.fc2(a5)

        output[:,0] = self.sigmoid(output[:,0])
        output[:, 1] = self.tanh(output[:, 1])

        return output


class CriticFC(nn.Module):
    """
    This network arch is amost identical to Actor, exept is adds in the number of actions and outputs a single Q Value instead of all actions
    """

    def __init__(self, num_states, num_actions, weight_scale, hidden_1=None, hidden_2=None, use_bn=True):
        super(CriticFC, self).__init__()
        self.fc1 = nn.Linear(num_states, hidden_1)
        self.fc2 = nn.Linear(hidden_1 + num_actions, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(hidden_1)
        #self.initialize_weights(weight_scale)
        self.use_bn = use_bn

    def initialize_weights(self, scale):
        '''
        From the paper

        The final layer weights and biases of both the actor and critic were initialized from a uniform distribution [−3 × 10−3, 3 × 10−3] and [3 × 10−4, 3 × 10−4] for the
        low dimensional and pixel cases respectively. This was to ensure the initial outputs for the policy
        and value estimates were near zero. The other layers were initialized from uniform distributions [− sqrt(1/f), sqrt(1/f) where f is the fan-in of the layer.

        '''

        self.fc1_fanin = np.sqrt(1 / self.fc1.weight.data.size()[0])
        self.fc1.weight.data = torch.Tensor(self.fc1.weight.data.size()).uniform_(-self.fc1_fanin, self.fc1_fanin)
        # self.fc1.weight.data.uniform_(-self.fc1_fanin,self.fc1_fanin)

        self.fc2_fanin = np.sqrt(1 / self.fc2.weight.data.size()[0])
        self.fc2.weight.data = torch.Tensor(self.fc2.weight.data.size()).uniform_(-self.fc2_fanin, self.fc2_fanin)
        # self.fc2.weight.data.uniform_(-self.fc2_fanin,self.fc2_fanin)

        self.fc3.weight.data.uniform_(-scale, scale)

    def forward(self, state, action):
        state = state.view(state.size(0), -1)
        z1 = self.fc1(state)
        a1 = self.relu(z1)
        if self.use_bn:
            a1 = self.bn1(a1)
        z2 = self.fc2(torch.cat([a1, action], 1))
        a2 = self.relu(z2)
        output = self.fc3(a2)

        return output



class CriticCNN(nn.Module):
    """
    This network arch is amost identical to Actor, exept is adds in the number of actions and outputs a single Q Value instead of all actions
    """

    def __init__(self, num_actions, hidden_1,hidden_2, filter_size,kernal_size,stride_size,padding, use_bn):
        super(CriticCNN, self).__init__()
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(3, filter_size, kernel_size=kernal_size, stride=stride_size, padding=padding)
        self.conv2 = nn.Conv2d(filter_size, filter_size, kernel_size=kernal_size, stride=stride_size, padding=padding)
        self.conv3 = nn.Conv2d(filter_size, filter_size, kernel_size=kernal_size, stride=stride_size, padding=padding)
        self.conv4 = nn.Conv2d(filter_size, filter_size, kernel_size=kernal_size, stride=stride_size, padding=padding)

        self.dropout = nn.Dropout(.5)

        # TODO: add state vector here
        flattened_size = 144

        self.fc1 = nn.Linear(flattened_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1 + num_actions, hidden_2)
        self.fc3 = nn.Linear(hidden_2,1)

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(filter_size)
            self.bn2 = nn.BatchNorm2d(filter_size)
            self.bn3 = nn.BatchNorm2d(filter_size)
            self.bn4 = nn.BatchNorm2d(filter_size)

    def initialize_weights(self, scale):
        '''
        From the paper

        The final layer weights and biases of both the actor and critic were initialized from a uniform distribution [−3 × 10−3, 3 × 10−3] and [3 × 10−4, 3 × 10−4] for the
        low dimensional and pixel cases respectively. This was to ensure the initial outputs for the policy
        and value estimates were near zero. The other layers were initialized from uniform distributions [− sqrt(1/f), sqrt(1/f) where f is the fan-in of the layer.

        '''

        self.fc1_fanin = np.sqrt(1 / self.fc1.weight.data.size()[0])
        self.fc1.weight.data = torch.Tensor(self.fc1.weight.data.size()).uniform_(-self.fc1_fanin, self.fc1_fanin)
        # self.fc1.weight.data.uniform_(-self.fc1_fanin,self.fc1_fanin)

        self.fc2_fanin = np.sqrt(1 / self.fc2.weight.data.size()[0])
        self.fc2.weight.data = torch.Tensor(self.fc2.weight.data.size()).uniform_(-self.fc2_fanin, self.fc2_fanin)
        # self.fc2.weight.data.uniform_(-self.fc2_fanin,self.fc2_fanin)

        self.fc3.weight.data.uniform_(-scale, scale)

    def forward(self, state, action):
        z1 = self.conv1(state)
        a1 = self.leaky_relu(z1)
        if self.use_bn:
            a1 = self.bn1(a1)

        z2 = self.conv2(a1)
        a2 = self.leaky_relu(z2)
        if self.use_bn:
            a2 = self.bn2(a2)

        z3 = self.conv3(a2)
        a3 = self.leaky_relu(z3)
        if self.use_bn:
            a3 = self.bn3(a3)

        z4 = self.conv4(a3)
        a4 = self.leaky_relu(z4)
        if self.use_bn:
            a4 = self.bn4(a4)

        a4_flat = a4.view(a4.size(0), -1)  # flatten

        z5 = self.fc1(a4_flat)
        a5 = self.leaky_relu(z5)

        z6 = self.fc2(torch.cat([a5, action], 1))
        output = self.fc3(z6)


        return output