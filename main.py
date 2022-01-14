#!/usr/bin/env python
import matplotlib.pyplot as plt
import gym
import torch

from src.model import FC
from src.dqn import DQN_agent

from src.utils import greedy_action


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make("CartPole-v1")

    net = FC(
            input_dim=env.observation_space.shape[0],
            hidden_dim=16,
            output_dim=env.action_space.n,
        ).to(device)

    optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)

    config = dict(
        observation_space=env.observation_space.shape[0],
        nb_actions=env.action_space.n,
        learning_rate=0.001,
        gamma=0.95,
        buffer_size=1000000,
        epsilon_min=0.01,
        epsilon_max=1.,
        epsilon_decay_period=1000,
        epsilon_delay_decay=20,
        batch_size=20,


        replay_size=int(1e6),
        optimizer=optimizer,
        tau_delay=20,
        epsilon_step=0.01,
        tau_period=1000,
        env=env,
        agent=net,
        )

    agent = DQN_agent(config, net)
    scores = agent.train(env, 200)
    plt.plot(scores)
    plt.show()

    x = env.reset()
    env.render()
    for i in range(1000):
        a = greedy_action(net, x)
        y, _, d, _ = env.step(a)
        env.render()
        x = y
        if d:
            print(i)
            break
    env.close()


if __name__ == "__main__":
    main()
