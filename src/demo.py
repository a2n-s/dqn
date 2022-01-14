import argparse

import numpy as np
import matplotlib.pyplot as plt
import gym
import time

from environments.swingup import CartPoleSwingUp
from environments.pongwrapper import PongWrapper


plt.style.use("dark_background")


def demo_cartpole():
    cartpole = gym.make('CartPole-v1')
    cartpole.reset()
    cartpole.render()
    for i in range(2000):
        _, _, d, _ = cartpole.step(np.random.randint(2))
        cartpole.render()
        if d:
            print(i)
            break
        time.sleep(cartpole.tau)
    cartpole.close()


def demo_swingup():
    swingup = CartPoleSwingUp()
    swingup.reset()
    swingup.render()
    for i in range(1000):
        _, _, d, _ = swingup.step(np.random.randint(2))
        swingup.render()
        if d:
            print(i)
            break
    swingup.close()


def demo_pong():
    pong = PongWrapper(noop_max=0,
                       frame_skip=4,
                       terminal_on_life_loss=True,
                       grayscale_obs=True,
                       scale_obs=True)
    x = pong.reset()
    pong.render()
    for i in range(60):
        a = np.random.randint(2)
        x, r, _, _ = pong.step(a)
        pong.render()
        # print('\r', "reward", r, end="")
        time.sleep(0.1)
    pong.close()
    print(f"shape: {x.shape} min = {x.min()} max = {x.max()}")
    plt.imshow(x, cmap='gray')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cartpole", action="store_true")
    parser.add_argument("-s", "--swingup", action="store_true")
    parser.add_argument("-p", "--pong", action="store_true")

    args = parser.parse_args()

    if args.cartpole:
        demo_cartpole()
    if args.swingup:
        demo_swingup()
    if args.pong:
        demo_pong()
