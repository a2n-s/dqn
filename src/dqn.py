import numpy as np
import torch

import sys
sys.path.append("./src")
from buffer import ReplayBuffer
from utils import greedy_action


# the pseudo code of DQN.
"""
state = init()
loop:
   action = greedy_action(DQN) or random_action()
   new_state, reward = step(state, action)
   replay_memory.add(state, action, reward, new_state)
   minibatch = replay_memory.sample(minibatch_size)
   X_train = Y_train = []
   for (s,a,r,s') in minibatch:
       Q  = DQN.predict(s)
       Q' = DQN.predict(s')
       if non-terminal(s'):
           update = r + gamma * max(Q')
       else:
           update = r
       Q[a] = update
       X_train.add(s)
       Y_train.add(Q)
   DQN.train_one_step(X_train,Y_train)
   state = new_state
"""


class DQN_agent:
    def __init__(self, config, model):
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'])
        self.eps_max = config['epsilon_max']
        self.eps_min = config['epsilon_min']
        self.eps_stop = config['epsilon_decay_period']
        self.eps_delay = config['epsilon_delay_decay']
        self.eps_step = (self.eps_max - self.eps_min) / self.eps_stop
        self.total_steps = 0
        self.model = model
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=config['learning_rate'])

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state = env.reset()
        epsilon = self.eps_max
        step = 0

        while episode < max_episode:
            # update epsilon
            if step > self.eps_delay:
                epsilon = max(self.eps_min, epsilon-self.eps_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = np.random.randint(self.nb_actions)
            else:
                action = greedy_action(self.model, state)

            # step
            next_state, reward, done, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            self.gradient_step()

            # next transition
            step += 1
            if done:
                episode += 1
                print("Episode ", '{:3d}'.format(episode),
                      ", epsilon ", '{:6.2f}'.format(epsilon),
                      ", batch size ", '{:5d}'.format(len(self.memory)),
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return
