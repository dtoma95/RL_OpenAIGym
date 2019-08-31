import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make('Acrobot-v1')
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

model = Sequential()
model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
model.add(Dense(action_space, activation="linear"))
model.compile(loss="mse", optimizer=Adam(lr=0.001))
model.load_weights("acrobot_saves\\run_1027")

observation = env.reset()
for t in range(1000):
    env.render()
    print(observation)
    observation = np.reshape(observation, [1, observation_space])
    action = np.argmax(model.predict([observation])[0])
    print(action)
    observation, reward, done, info = env.step(action)
    print(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()


#polako konvergira i pokazuje dosta dobre rezultate, mnogi i ispod 100 koraka
#ali odjednom oko 170. iteracije je prestao u potupnosti da resava okruzenje i svaka epizoda ima 500 koraka (najgore moguce)
#mozda treba veci range na discountu? nzm acrobot_saves_old\\run_231
