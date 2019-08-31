import gym
import random
import numpy as np
env = gym.make('MountainCarContinuous-v0')
observation_space = env.observation_space
action_space = env.action_space
print(observation_space)
print(action_space)
observation = env.reset()


print(env.observation_space.high, env.observation_space.low)
print(env.action_space.high, env.action_space.low)
for t in range(1000):
    env.render()

    #observation = np.reshape(observation, [1, observation_space])


    action = [random.uniform(-1, 1)]# env.action_space.sample()


    observation, reward, done, info = env.step(action)
    print(action, observation[0])
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()
