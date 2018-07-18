# -*- coding: utf-8 -*-

import enum
import numpy as np

sin_original = []
class Action(enum.Enum):
    SELL = 2
    HOLD = 0
    BUY  = 1

class Position():
    def __init__(self):
        self.NONE = 0
        self.BUY = 1
        self.SELL = 2

        self.bid = 0.0
        self.pos_state = self.NONE

    def state(self):
        return np.array([self.bid, self.pos_state])

    def change(self, action, bid):
        self.bid = bid
        self.pos_state = action

    def close(self, action, bid):
        if self.pos_state == action:
            return 0

        self.pos_state = self.NONE
        if action == Action.BUY.value:
            reward = (bid - self.bid) * 1000
        else:
            reward = (self.bid - bid) * 1000
        return reward

import gym
import gym.spaces
from numpy.random import *


class FXTrade(gym.core.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)
        high = np.array([1.0, 1.0])
        self.observation_space = gym.spaces.Box(low=-high, high=high)

        self._position = Position()

        self._sin_list = []
        k=0.05
        i=0
        for t in np.linspace(0, 48, 240):
            self._sin_list.append(np.sin(t)*np.exp(-k*t)+0.1*randn())
            #self._sin_list.append(np.sin(t))
        for i in range(len(self._sin_list)):
            sin_original.append(self._sin_list[i])
        self.cur_id = 0
        

    def step(self, action):
        bid = self._sin_list[self.cur_id]
        self.cur_id +=1
        done = True if self.cur_id == 240 else False

        if action == Action.HOLD.value:
            reward = 0
        else:
            if self._position.pos_state == self._position.NONE:
                self._position.change(action, bid)
                reward = 0
            else:
                reward = self._position.close(action, bid)
        return np.array([bid, self._position.pos_state]), reward, done ,{}

    def reset(self):
        self.cur_id = 0
        self._position = Position()
        return np.array([0.0, 0.0])

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = FXTrade()
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
model.summary()

# DQNに必要な Experience Replayのためのメモリ
memory = SequentialMemory(limit=50000, window_length=1)
# ボルツマン分布に従うポリシー
policy = BoltzmannQPolicy()
# DQN Agentの生成
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
               target_model_update=1e-2, policy=policy)
# optimizersはAdam, 損失関数は 平均二乗誤差
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# トレーニングを開始。同じ正弦曲線を9600 = 240 x 400回 回す。
histry = dqn.fit(env, nb_steps=9600, visualize=False, verbose=1)

# トレーニング結果を確認
dqn.test(env, nb_episodes=5, visualize=False)

import rl.callbacks
class EpisodeLogger(rl.callbacks.Callback):
    def __init__(self):
        self.observations = {}
        self.rewards = {}
        self.actions = {}

    def on_episode_begin(self, episode, logs):
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])

cb_ep = EpisodeLogger()
dqn.test(env, nb_episodes=10, visualize=False, callbacks=[cb_ep])


#matplotlib inline
import matplotlib.pyplot as plt

for obs in cb_ep.observations.values():
    plt.plot([o[0] for o in obs],"r")
t = np.linspace(0,240,240)
plt.plot(t,sin_original,"b")    
plt.xlabel("step")
plt.ylabel("pos")
plt.pause(3)
plt.savefig('plot_epoch_{0:03d}_FXTrade_dqn.png'.format(9600), dpi=60)
plt.close()

print('Plotting Results')
plt.subplot(2, 1, 1)
plt.title('Original')
t = np.linspace(0,240,240)
plt.plot(t,sin_original,"b")    
plt.xlabel("step")
plt.ylabel("pos")
plt.pause(3)

plt.subplot(2, 1, 2)
plt.title('observations&original')
for obs in cb_ep.observations.values():
    plt.plot([o[0] for o in obs],"r")
t = np.linspace(0,240,240)
plt.plot(t,sin_original,"b")    
plt.xlabel("step")
plt.ylabel("pos")
plt.pause(3)

plt.savefig('plot_epoch_{0:03d}_FXTrade_obvs&original.png'.format(9600), dpi=60)
plt.close()
