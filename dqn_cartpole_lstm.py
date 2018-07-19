import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Reshape
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from keras.layers import LSTM

ENV_NAME = 'CartPole-v0'

env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Reshape((2, 2), input_shape=(4,)))
model.add(LSTM(100, input_shape=(2, 2), 
               #batch_size=25,
               return_sequences=False,
               #stateful=True,
              dropout=0.0))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
"""
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
"""
print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy() 
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
#dqn.compile(loss="mse", Adam(lr=1e-3), metrics=['mae'])

history = dqn.fit(env, nb_steps=10000, visualize=True, verbose=2)

dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

dqn.test(env, nb_episodes=5, visualize=True)

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
flatten_1 (Flatten)          (None, 4)                 0
_________________________________________________________________
reshape_1 (Reshape)          (None, 2, 2)              0
_________________________________________________________________
lstm_1 (LSTM)                (None, 100)               41200
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 202
_________________________________________________________________
activation_1 (Activation)    (None, 2)                 0
=================================================================
Total params: 41,402
Trainable params: 41,402
Non-trainable params: 0
_________________________________________________________________

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
flatten_1 (Flatten)          (None, 4)                 0
_________________________________________________________________
reshape_1 (Reshape)          (None, 1, 4)              0
_________________________________________________________________
lstm_1 (LSTM)                (None, 100)               42000
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 202
_________________________________________________________________
activation_1 (Activation)    (None, 2)                 0
=================================================================
Total params: 42,202
Trainable params: 42,202
Non-trainable params: 0
_________________________________________________________________
"""