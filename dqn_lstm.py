# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
#import timeit
import json
import os
import numpy

from my_callback import MyCallback
import gym.spaces
import numpy
import pandas

LAG = 3

def calc_profit(action, df, index):
    if action == 0: # long
        p = 1
    elif action == 1: # short
        p = -1
    else: # stay
        p = 0
    return  p * df["c"][index]

def calc_observation(df, index, columns):
    return numpy.array(
        [
            [df[col][index-t] for col in columns] for t in range(LAG)
        ] 
    )

class Game(gym.core.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, df, columns):
        self.df = df.reset_index(drop=True)
        self.columns = columns
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(0, 999, shape=(LAG, len(columns)))
        self.time = LAG
        self.profit = 0
        
    def step(self, action):
        reward = calc_profit(action, self.df, self.time)
        self.time += 1
        self.profit += reward       
        done = self.time == (len(self.df) - 1)
        if done:
            print("profit___{}".format(self.profit))
        info = {}
        observation = calc_observation(self.df, self.time, self.columns)
        return observation, reward, done, info

    def reset(self):
        self.time = LAG
        self.profit = 0
        return calc_observation(self.df, self.time, self.columns)
    
    def render(self, mode):
        pass
    
    def close(self):
        pass
    
    def seed(self):
        pass
from keras.models import Sequential, Model
from keras.layers import (
    Dense,
    Activation,
    Flatten,
    Input,
    concatenate,
    Dropout,
    LSTM,
    Reshape
)

class Network(object):
    def __init__(self):
        self.model = None

    def sample_model(self, observation_space):
        
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + observation_space.shape))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(n_action))
        model.add(Activation('linear'))
        print(model.summary())
        
        """
        model = Sequential()
        model.add(Reshape(observation_space.shape,
                      input_shape=(1,)+observation_space.shape))
        model.add(LSTM(LAG))
        model.add(Dense(32))
        model.add(Activation('relu'))
        #model.add(Dropout(0.6))
        model.add(Dense(16))
        model.add(Activation('relu'))
        #model.add(Dropout(0.6))
        model.add(Dense(n_action))
        model.add(Activation('linear'))
        """
        """
        model = Sequential()
        model.add(Reshape(observation_space.shape,
                      input_shape=(1,)+observation_space.shape))
        model.add(LSTM(16, input_shape=(3, 2), 
               #batch_size=25,
               return_sequences=False,
               #stateful=True,
              dropout=0.0))
        model.add(Dense(n_action))
        model.add(Activation('linear'))
        """
        print(model.summary())
        self.model = model
        return model        
    
    def from_json(self, file_path):
        pass
    
    def to_json(self, output_path):
        pass
        
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from keras.optimizers import Adam

def agent(model, n_action):
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy(eps=0.1)
    dqn_agent = DQNAgent(model=model, nb_actions=n_action,
                         memory=memory, nb_steps_warmup=100,
                         target_model_update=1e-2, policy=policy)
    dqn_agent.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn_agent
    
import pandas
import numpy

df = pandas.DataFrame({"a": numpy.random.rand(1000), "b": numpy.random.rand(1000)})
df["c"]= ((df["a"] - df["b"]).shift(2)).fillna(0)
df.head(10)
columns = ["a", "b"]
env = Game(df, columns)
n_action = 3
network = Network()
model = network.sample_model(env.observation_space)
agent_v6 = agent(model, n_action)

callback= MyCallback("tmp")
agent_v6.fit(env, nb_steps=10000, visualize=False,
                  verbose=2, callbacks=[callback])

agent_v6.load_weights("tmp/best_lstm_weight.hdf5")
al = []
pl = []
for i in range(3, len(df)):
    obs = calc_observation(df, i, columns)
    action = agent_v6.forward(obs)
    profit = calc_profit(action, df, i)
    al.append(action)
    pl.append(profit)

                  
df["pred"] = pandas.Series(al).shift(2)
df["profit"] = pandas.Series(pl).shift(3)
print(df)

print(df["c"].abs().sum(), df["profit"].sum())


"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
flatten_1 (Flatten)          (None, 6)                 0
_________________________________________________________________
dense_1 (Dense)              (None, 16)                112
_________________________________________________________________
activation_1 (Activation)    (None, 16)                0
_________________________________________________________________
dense_2 (Dense)              (None, 16)                272
_________________________________________________________________
activation_2 (Activation)    (None, 16)                0
_________________________________________________________________
dense_3 (Dense)              (None, 16)                272
_________________________________________________________________
activation_3 (Activation)    (None, 16)                0
_________________________________________________________________
dense_4 (Dense)              (None, 3)                 51
_________________________________________________________________
activation_4 (Activation)    (None, 3)                 0
=================================================================
Total params: 707
Trainable params: 707
Non-trainable params: 0
_________________________________________________________________

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
reshape_1 (Reshape)          (None, 3, 2)              0
_________________________________________________________________
lstm_1 (LSTM)                (None, 3)                 72
_________________________________________________________________
dense_1 (Dense)              (None, 32)                128
_________________________________________________________________
activation_1 (Activation)    (None, 32)                0
_________________________________________________________________
dense_2 (Dense)              (None, 16)                528
_________________________________________________________________
activation_2 (Activation)    (None, 16)                0
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 51
_________________________________________________________________
activation_3 (Activation)    (None, 3)                 0
=================================================================
Total params: 779
Trainable params: 779
Non-trainable params: 0
_________________________________________________________________

model = Sequential()
        model.add(Reshape(observation_space.shape,
                      input_shape=(1,)+observation_space.shape))
        model.add(LSTM(16, input_shape=(3, 2), 
               #batch_size=25,
               return_sequences=False,
               #stateful=True,
              dropout=0.0))
        model.add(Dense(n_action))
        model.add(Activation('linear'))

[1000 rows x 5 columns]
345.331418387 298.854889973

"""