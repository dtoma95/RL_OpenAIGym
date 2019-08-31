# \Desktop\Ultimate Folder\SCHOOL\master\Poslovni

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


#from scores.score_logger import ScoreLogger

ENV_NAME = "MountainCarContinuous-v0"

GAMMA = 0.95
LEARNING_RATE = 0.0091

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 2.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="tanh"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return [random.uniform(-1,1)]
        q_values = self.model.predict(state)

        return q_values

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * self.model.predict(state_next)[0])
            q_values = self.model.predict(state)
            q_values[0] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def reward_calc(state, terminal, reward):
    if terminal and state[0]> 0.5:
        return 2000

    retval = 0
    if(state[0] > -0.3):
        #print ("hoho", str(state[0]))
        retval += 1
    if (state[0] > 0):
        retval += 1
    if (state[0] > 0.1):
        retval += 1
    if (state[0] > 0.2):
        retval += 1
    if (state[0] > 0.3):
        retval += 1
    if (state[0] > 0.4):
        retval += 1
    retval -= 1
    return retval


def cartpole():
    env = gym.make(ENV_NAME)

   # score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        if run % 100 == 0:
            dqn_solver.save_model("mtncc\\run_"+str(run))
        state = env.reset()

        state = np.reshape(state, [1, observation_space])
        step = 0
        score = 0
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state)

            state_next, reward, terminal, info = env.step(action)
            #state_next = np.reshape(state_next, [1, observation_space])
            reward = reward_calc(state_next, terminal, reward)
            score += reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if step == 200:
                terminal = True
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(score) + ", steps:  "+ str(step))
                #score_logger.add_score(step, run)
                if step < 200:
                    dqn_solver.save_model("mtncc\\run_" + str(run))
                break
            dqn_solver.experience_replay()


if __name__ == "__main__":
    cartpole()