# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import json
import os
import numpy

from my_callback import MyCallback
import gym.spaces
import numpy
import pandas
import pandas as pd

LAG = 1

def calc_profit(action, df, index):
    if action == 0: 
        if df["c"][index] ==0:
            return 1
        else:
            return 0 
    elif action == 1: 
        if df["c"][index] ==1:
            return 1
        else:
            return 0 
    return 0
    

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
        self.action_space = gym.spaces.Discrete(2) #3
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
        """
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(n_action))
        model.add(Activation('linear'))
        """
        model = Sequential()
        model.add(Reshape(observation_space.shape,
                      input_shape=(1,)+observation_space.shape))
        model.add(LSTM(50, input_shape=(3, 1), 
               return_sequences=False,
              dropout=0.0))
        model.add(Dense(n_action))
        model.add(Activation('linear'))
        
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
                         memory=memory, nb_steps_warmup=1,
                         target_model_update=1e-2, policy=policy)
    dqn_agent.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn_agent
    
import pandas
import numpy

df1 = pd.DataFrame({"s": 0,"a": numpy.random.rand(500)//0.5, "b": numpy.random.rand(500)//0.5})  #//0.5
df1["c"]= ((df1["a"]  + df1["b"]-df1["a"]*df1["b"]).shift(LAG-1)).fillna(0) #論理学習のため
df2 = pd.DataFrame({"s": 10,"a": numpy.random.rand(500)//0.5, "b": numpy.random.rand(500)//0.5})  #//0.5
df2["c"]= ((df2["a"]  * df2["b"]).shift(LAG-1)).fillna(0) #論理学習のため
#df2.index   =[ 500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549] #インデックスをふる
df2.index = pd.RangeIndex(start=500, stop=1000, step=1)
df=pd.concat([df1,df2],axis=0)  #結合を行う =rbind
#df = df1

print(df1.head(1000))
print(df2.head(1000))
print(df.head(1000))

columns = ["s","a", "b"]
env = Game(df, columns)
n_action = 2  #3
network = Network()
model = network.sample_model(env.observation_space)
agent_v6 = agent(model, n_action)

#before learning
al = []
pl = []
for i in range(LAG, len(df)):
    obs = calc_observation(df, i, columns)
    action = agent_v6.forward(obs)
    profit = calc_profit(action, df, i)
    al.append(action)
    pl.append(profit)

                  
df["pred"] = pandas.Series(al).shift(LAG)   #2)
df["profit"] = pandas.Series(pl).shift(LAG) #3)
print(df)

print(df["c"].abs().sum(), df["profit"].sum())

callback= MyCallback("tmp")
agent_v6.fit(env, nb_steps=10000, visualize=False,
                  verbose=2, callbacks=[callback])

#after learning
agent_v6.load_weights("tmp/best_lstm_game00_weight.hdf5")
al = []
pl = []
for i in range(LAG, len(df)):
    obs = calc_observation(df, i, columns)
    action = agent_v6.forward(obs)
    profit = calc_profit(action, df, i)
    al.append(action)
    pl.append(profit)

                  
df["pred"] = pandas.Series(al).shift(LAG)   #2)
df["profit"] = pandas.Series(pl).shift(LAG) #3)
print(df)

print(df["c"].abs().sum(), df["profit"].sum())


"""
Layer (type)                 Output Shape              Param #
=================================================================
reshape_1 (Reshape)          (None, 1, 2)              0
_________________________________________________________________
lstm_1 (LSTM)                (None, 50)                10600
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 102
_________________________________________________________________
activation_1 (Activation)    (None, 2)                 0
=================================================================
Total params: 10,702
Trainable params: 10,702
Non-trainable params: 0
_________________________________________________________________
done, took 690.205 seconds
       a    b    c  pred  profit
0    1.0  1.0  1.0   NaN     NaN
1    1.0  1.0  1.0   1.0     1.0
2    1.0  0.0  0.0   0.0     1.0
3    0.0  1.0  0.0   1.0     0.0  #間違っている
4    1.0  1.0  1.0   1.0     1.0
5    0.0  0.0  0.0   0.0     1.0
6    1.0  1.0  1.0   1.0     1.0
7    0.0  1.0  0.0   0.0     1.0
8    0.0  0.0  0.0   0.0     1.0
9    0.0  1.0  0.0   0.0     1.0
10   1.0  1.0  1.0   1.0     1.0
11   0.0  0.0  0.0   0.0     1.0
12   0.0  1.0  0.0   0.0     1.0
13   0.0  1.0  0.0   0.0     1.0
14   1.0  1.0  1.0   1.0     1.0
15   1.0  1.0  1.0   0.0     0.0　#間違っている
16   1.0  0.0  0.0   0.0     1.0
17   1.0  0.0  0.0   0.0     1.0
18   1.0  0.0  0.0   0.0     1.0
19   0.0  1.0  0.0   0.0     1.0
20   1.0  0.0  0.0   0.0     1.0
21   0.0  1.0  0.0   0.0     1.0
22   0.0  1.0  0.0   0.0     1.0
23   0.0  1.0  0.0   0.0     1.0
24   1.0  0.0  0.0   0.0     1.0
25   0.0  1.0  0.0   0.0     1.0
26   1.0  0.0  0.0   0.0     1.0
27   1.0  0.0  0.0   0.0     1.0
28   0.0  1.0  0.0   0.0     1.0
29   0.0  1.0  0.0   0.0     1.0
..   ...  ...  ...   ...     ...
970  1.0  1.0  1.0   1.0     1.0
971  0.0  0.0  0.0   0.0     1.0
972  0.0  0.0  0.0   0.0     1.0
973  1.0  0.0  0.0   0.0     1.0
974  1.0  1.0  1.0   1.0     1.0
975  1.0  0.0  0.0   0.0     1.0
976  1.0  0.0  0.0   0.0     1.0
977  0.0  1.0  0.0   0.0     1.0
978  1.0  1.0  1.0   1.0     1.0
979  1.0  1.0  1.0   1.0     1.0
980  0.0  0.0  0.0   0.0     1.0
981  1.0  0.0  0.0   0.0     1.0
982  1.0  0.0  0.0   0.0     1.0
983  1.0  1.0  1.0   1.0     1.0
984  0.0  0.0  0.0   0.0     1.0
985  0.0  1.0  0.0   0.0     1.0
986  1.0  1.0  1.0   1.0     1.0
987  0.0  1.0  0.0   0.0     1.0
988  1.0  1.0  1.0   1.0     1.0
989  1.0  1.0  1.0   1.0     1.0
990  0.0  1.0  0.0   0.0     1.0
991  1.0  1.0  1.0   1.0     1.0
992  1.0  0.0  0.0   0.0     1.0
993  0.0  0.0  0.0   0.0     1.0
994  0.0  0.0  0.0   0.0     1.0
995  0.0  1.0  0.0   0.0     1.0
996  1.0  1.0  1.0   1.0     1.0
997  1.0  1.0  1.0   1.0     1.0
998  0.0  0.0  0.0   0.0     1.0
999  0.0  1.0  0.0   NaN     NaN

[1000 rows x 5 columns]
245.0 949.0

"""