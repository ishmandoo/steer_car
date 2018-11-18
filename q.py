from steer_car import SteerCar

import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

target = [10.,10.]
epsilon = 0.3
discount = 0.99

class SQN(nn.Module):

    def __init__(self):
        super(SQN, self).__init__()
        self.l1 = nn.Linear(3, 10) # x, y and theta input, ten hidden layers
        self.l2 = nn.Linear(10, 9) # 3 steering options x 3 accel options

    def forward(self, x):
        x = F.relu(self.l1(x))
        return self.l2(x)

def reward(state):
    x, y, theta = state
    pos = np.array([x,y])
    print(pos)
    dist = np.linalg.norm(pos-target)
    '''
    if dist < 1.:
        return 1.
    else:
        return 0.
    '''
    return -dist

network = SQN()
network.double()
optimizer = optim.Adam(network.parameters())


car = SteerCar()
car.reset()

for ep in range(1000):
    car.reset()
    for i in range(500):

        old_state = torch.tensor(car.state)
                
        if random.random() > 100./(ep+1):
            with torch.no_grad():        
                action_index = torch.argmax(network.forward(old_state))
        else:
            action_index = random.randint(0,8)

        car.step(np.array([(action_index//3) - 1, (action_index%3) -1]))

        #accel,steer = action
        #accel_index = accel + 1
        #steer_index = steer + 1
        #action_index = 3 * accel_index + steer_index

        with torch.no_grad():
            next_q = torch.max(network.forward(torch.tensor(car.state)))

        loss = (network.forward(old_state)[action_index] - reward(old_state) + discount * next_q)**2
        print(reward(old_state))
        loss.backward()
        optimizer.step()

        car.render()
        #time.sleep(0.01)
