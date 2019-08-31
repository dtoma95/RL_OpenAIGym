import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make('MountainCar-v0')
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

model = Sequential()
model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(action_space, activation="linear"))
model.compile(loss="mse", optimizer=Adam(lr=0.001))
model.load_weights("mountcar_saves\\run_1734")

observation = env.reset()
for t in range(1000):
    env.render()
    print(observation)
    observation = np.reshape(observation, [1, observation_space])
    action = model.predict([observation])
    print(action)
    action = np.argmax(action[0])

    observation, reward, done, info = env.step(action)
    print(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()
