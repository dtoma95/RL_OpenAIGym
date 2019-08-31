# Desktop\Ultimate Folder\SCHOOL\master\Poslovni

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


#from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.01

MEMORY_SIZE = 1000000
BATCH_SIZE = 0

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

DISCOUNTING_MAX = -400
DISCOUNTING_RANGE = 40
DISCOUNTING_DECAY = 0.8

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)

    def discounting(self):
        last = len (self.memory)-1
        new_reward = DISCOUNTING_MAX
        for i in range (0,DISCOUNTING_RANGE):
            self.memory[last] = (self.memory[last][0], self.memory[last][1], new_reward, self.memory[last][3], self.memory[last][4]) #reward
            last-=1
            new_reward = new_reward*DISCOUNTING_DECAY
            if last <0:
                break
    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def assertion(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = self.memory
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.memory = []
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def cartpole():
    env = gym.make(ENV_NAME)
   # score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        if run % 100 == 0:
            dqn_solver.save_model("cartpole_saves_disc\\run_"+str(run))
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            #reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                if (step < 500):
                    dqn_solver.discounting()
                dqn_solver.assertion()
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", step: " + str(step))
                if (step == 500):
                    dqn_solver.save_model("cartpole_saves_disc\\run_" + str(run))
                #score_logger.add_score(step, run)
                break
            #dqn_solver.assertion()
            #dqn_solver.experience_replay()


if __name__ == "__main__":
    cartpole()