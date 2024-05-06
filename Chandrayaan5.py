import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.layers import Dense, Activation, Input
from keras.models import Sequential, load_model
from keras.optimizers import Adam

import numpy as np
import gymnasium as gym

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_counter = 0
        self.mem_size = max_size
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal
    
def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = Sequential([
                Input(shape=(input_dims, )),
                Dense(fc1_dims),
                Activation('relu'),
                Dense(fc2_dims),
                Activation('relu'),
                Dense(n_actions)])
    
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    
    return model

class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                input_dims, epsilon_dec=0.996, epsilon_end=0.01, mem_size=1000000,
                fname='chandrayaan.keras', load = False):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        
        if load:
            self.q_eval = load_model(fname)
        else:
            self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        rand = np.random.random()
        state = state[np.newaxis, :]

        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state, verbose=0)
            action = np.argmax(actions)
    
        return action

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        q_eval = self.q_eval.predict(state, verbose=0)
        q_next = self.q_eval.predict(new_state, verbose=0)

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1)*done

        _ = self.q_eval.fit(state, q_target, verbose=0)

        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_end else self.epsilon_end
    
    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)


env = gym.make('LunarLander-v2')
n_games = 500
agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=8, n_actions=4, mem_size=1000000, batch_size=64, epsilon_end=0.01, load=False)

scores = []

for i in range(1,n_games+1):
    done = False
    score = 0
    observation = env.reset()
    observation = observation[0]
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info, _ = env.step(action)
        score += reward   
        agent.remember(observation, action, reward, observation_, done)
        observation = observation_
        agent.learn()
    
    scores.append(score)
    avg_score = np.mean(scores[max(0,i-100):(i+1)])
    print('episode:', i,'score:', score, 'avg_score:', avg_score)

    if i%50 == 0:
        agent.save_model()