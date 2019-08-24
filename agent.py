import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.1
        self.alpha = 0.11
        self.nb_episodes = 1
        self.gamma = 0.9
        
    def epsilon_greedy_policy(self, state):
        max_action = np.argmax(self.Q[state])
        policy = np.ones(self.nA) * self.epsilon / self.nA
        policy[max_action] += 1 - self.epsilon
        
        return policy 

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(self.nA, p=self.epsilon_greedy_policy(state))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        if done : 
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
            self.nb_episodes += 1
            self.epsilon = 1. / self.nb_episodes
        else : 
            self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])