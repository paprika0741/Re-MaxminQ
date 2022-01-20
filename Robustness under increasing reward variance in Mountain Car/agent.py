from memory import ReplayBuffer
from table import TiledQTable
import numpy as np
import random
import copy

class QLearningAgent:
    """Q-Learning agent that can act on a continuous state space by discretizing it."""
    def __init__(self, env, tiling_specs, alpha=0.02, gamma=0.99,
                 epsilon=1.0, epsilon_decay_speed = 10000, min_epsilon=.01, seed=0, batch_size = 32, memory_szie=100):
        """Initialize variables, create grid for discretization."""
        # Environment info
        self.env = env
        self.obs_dim =  env.observation_space.shape[0]
        self.tq = TiledQTable(env.observation_space.low, 
                    env.observation_space.high, 
                    tiling_specs, 
                    env.action_space.n)
        self.state_sizes = self.tq.state_sizes           # list of state sizes for each tiling
        self.action_size = self.env.action_space.n  # 1-dimensional discrete action space
        self.seed = np.random.seed(seed)

        # Learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = self.initial_epsilon = epsilon  # initial exploration rate
        self.epsilon_decay_speed  = epsilon_decay_speed    # how quickly should we decrease epsilon
        self.min_epsilon = min_epsilon
        #mempry
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.obs_dim , memory_szie, batch_size)

        # epsilon decay
        
    def epsilon_decay(self):
        self.epsilon  -= (self.initial_epsilon - self.min_epsilon) / self.epsilon_decay_speed
        self.epsilon = max(self.epsilon, self.min_epsilon)
    
    #  comput q value for current state
    def get_choose_action_q_value(self, state):
        Q_s = [self.tq.get(state, action) for action in range(self.action_size)]
        return  Q_s
    
    def choose_action(self,state, training):
        Q_s =  self.get_choose_action_q_value(state)
        greedy_action = np.argmax(Q_s)
        if training:
            if np.random.uniform(0, 1) < self.epsilon:
                # Pick a random action
                action = np.random.randint(0, self.action_size)
            else:
                action = greedy_action
        else:
            action = greedy_action
        return action
    
    # comput q value for next state batch
    def get_q_next_values(self, next_state):
        Q_s = [self.tq.get(next_state, action) for action in range(self.action_size)]
        value_next = max(Q_s)
        return  value_next


    
    def learn(self):
        state_batch , action_batch, reward_batch,  next_batch, done_batch = self.memory.sample( )
        for i in range (self.batch_size):
            state = state_batch[i]
            action = int( action_batch[i])
            reward = reward_batch[i]
            next_state = next_batch[i]
            done = done_batch[i]
            value_next =  self.get_q_next_values(next_state)
            value_target =  reward + self.gamma * value_next * (1-done)
            self.tq.update( state, action, value_target, self.alpha)
            
    def train (self,sigma): #one epoch
        state,done = self.env.reset(),False
        epoch_step = 0
        while True:
            epoch_step += 1
            action = self.choose_action(state,training=True)
            next_state,reward, done, _ = self.env.step(action)
            reward = random.gauss(-1, sigma)
            self.memory.store(state,action,reward,next_state,done )
            if len(self.memory)>= self.batch_size:
                self.learn()
            if done:
                break
            state = next_state
        return epoch_step
    
class DoubleQ(QLearningAgent):
    # comput q value for next state batch
    def __init__(self, env, tiling_specs, alpha=0.02, gamma=0.99,
                 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=0, batch_size = 32, memory_szie=100):
        super(DoubleQ,self).__init__( env, tiling_specs, alpha , gamma,epsilon, epsilon_decay_rate, min_epsilon, seed, batch_size , memory_szie)
        self.tq1 = copy.deepcopy(self.tq)
    
    #  comput q value for current state, use 
    def get_choose_action_q_value(self, state):
        Q_s1 = [self.tq.get(state, action) for action in range(self.action_size)]
        Q_s2 = [self.tq1.get(state, action) for action in range(self.action_size)]
        Q_s = [i+j for i,j in zip(Q_s1,Q_s2)]
        return Q_s
    
    def learn(self):
        state_batch , action_batch, reward_batch,  next_batch, done_batch = self.memory.sample( )
        for i in range (self.batch_size):
            state = state_batch[i]
            action = int( action_batch[i])
            reward = reward_batch[i]
            next_state = next_batch[i]
            done = done_batch[i]
            # 0.5 probability 
            if np.random.uniform(0, 1) < 0.5: 
                #update tq, use tq to select next_action, use tq1 to compute next value
                Q_s = [self.tq.get(next_state, i) for i in range(self.action_size)]
                next_action = np.argmax(Q_s  )
                value_next = self.tq1.get( next_state,next_action )
                value_target =  reward + self.gamma * value_next * (1-done)
                self.tq.update( state, action, value_target, self.alpha)
            else:
                #update tq1, use tq1 to select next_action, use tq to compute next value
                Q_s = [self.tq1.get(next_state, i) for i in range(self.action_size)]
                next_action = np.argmax(Q_s)
                value_next = self.tq.get( next_state,next_action )
                value_target =  reward + self.gamma * value_next * (1-done)
                self.tq.update( state, action, value_target, self.alpha)



class DoubleQ1(QLearningAgent):
    # comput q value for next state batch
    def __init__(self, env, tiling_specs, alpha=0.02, gamma=0.99,
                 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=0, batch_size = 32, memory_szie=100, update_freq = 20):
        super(DoubleQ1,self).__init__( env, tiling_specs, alpha , gamma,epsilon, epsilon_decay_rate, min_epsilon, seed, batch_size , memory_szie)
        self.tq1 = copy.deepcopy(self.tq)
        self.update_cnt = 0
        self.update_freq = update_freq

    #update tq, use tq to select next_action, use tq1 to compute next value
    def get_q_next_values(self, next_state):
        Q_s = [self.tq.get(next_state, i) for i in range(self.action_size)]
        next_action = np.argmax(Q_s  )
        value_next = self.tq1.get( next_state,next_action )
        return value_next
    
    def learn(self):
        state_batch , action_batch, reward_batch,  next_batch, done_batch = self.memory.sample( )
        for i in range (self.batch_size):
            state = state_batch[i]
            # must be int 
            action = int( action_batch[i])
            reward = reward_batch[i]
            next_state = next_batch[i]
            done = done_batch[i]
    
            value_next = self.get_q_next_values(next_state)
            value_target =  reward + self.gamma * value_next * (1-done)
            self.tq.update( state, action, value_target, self.alpha)
        self.update_cnt+=1
        if self.update_cnt % self.update_freq == 0:
            self.tq1 = copy.deepcopy(self.tq)



class AverageQ(QLearningAgent):
    # comput q value for next state batch
    def __init__(self, env, tiling_specs, alpha=0.02, gamma=0.99,
                 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=0, batch_size = 32, memory_szie=100,   net_num = 2):
        super(AverageQ,self).__init__( env, tiling_specs, alpha , gamma,epsilon, epsilon_decay_rate, min_epsilon, seed, batch_size , memory_szie)
        self.net_num = net_num 
        #tq  is Q
        #tq[0]...[net_num-1] is Q_target/  namely, previous table
        self.tq_target= [None]*net_num 
        for i in range(0,net_num ):
            self.tq_target[i] = TiledQTable(env.observation_space.low, 
                    env.observation_space.high, 
                    tiling_specs, 
                    env.action_space.n)
        self.update_cnt = 0
        self.update_freq = 20
        self.update_index = 0
            
    def get_q_next_values(self, next_state):
        Q_sum =   [self.tq_target[0].get(next_state, action) for action in range(self.action_size)]
        for i in range(1, self.net_num):
            Q =  [self.tq_target[i].get(next_state, action) for action in range(self.action_size)]
            Q_sum =  [ m + n  for m , n in zip(Q_sum ,Q)  ]
        Q_avg =   [i/self.net_num for i in Q_sum]
        return max(Q_avg)
    
    def learn(self):
        state_batch , action_batch, reward_batch,  next_batch, done_batch = self.memory.sample( )
        for i in range (self.batch_size):
            state = state_batch[i]
            action = int( action_batch[i])
            reward = reward_batch[i]
            next_state = next_batch[i]
            done = done_batch[i]
            value_next =  self.get_q_next_values(next_state)
            value_target =  reward + self.gamma * value_next * (1-done)
            self.tq.update( state, action, value_target, self.alpha)
        self.update_cnt+=1
        if self.update_cnt % self.update_freq == 0:
            #choose one table target to update
            #deep copy  
            self.tq_target[self.update_index] = copy.deepcopy(self.tq)
            self.update_index = (self.update_index+1)%self.net_num

class MaxminQ(QLearningAgent):
    # comput q value for next state batch
    def __init__(self, env, tiling_specs, alpha=0.02, gamma=0.99,
                 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=0, batch_size = 32, memory_szie=100,   net_num = 2, set_num=1):
        super(MaxminQ,self).__init__( env, tiling_specs, alpha , gamma,epsilon, epsilon_decay_rate, min_epsilon, seed, batch_size , memory_szie)
        self.net_num = net_num 
        self.set_num = set_num
        # net_num个tq
        self.tq = [None]* self.net_num
        for i in range(self.net_num):
            self.tq[i] = TiledQTable(env.observation_space.low, 
                    env.observation_space.high, 
                    tiling_specs, 
                    env.action_space.n)
    
    #compute q_min for K q
    def get_q_min(self,state):
        Q_min =   [self.tq[0].get(state, action) for action in range(self.action_size)]
        for i in range(1, self.net_num):
            Q =  [self.tq[i].get(state, action) for action in range(self.action_size)]
            Q_min = [min(m,n) for m,n in zip(Q,Q_min)]
        return Q_min
        
    def get_choose_action_q_value(self, state):
        #compute q min for choose action 
        return self.get_q_min(state)
    
    def get_q_next_values(self, next_state):
        Q_min = self.get_q_min(next_state)
        next_value =  max(Q_min)
        return next_value

    def learn(self):
        #select a subset from net_num to update
        #range(0,m)生成[0,m-1]的数
        #random.sample (list,num) 从list中随机采样num个数
        subset = random.sample(list(range(self.net_num)),
                                    self.set_num)
        for index in subset:
            state_batch , action_batch, reward_batch,  next_batch, done_batch = self.memory.sample( )
            for i in range (self.batch_size):
                state = state_batch[i]
                action = int( action_batch[i])
                reward = reward_batch[i]
                next_state = next_batch[i]
                done = done_batch[i]
                value_next =  self.get_q_next_values(next_state)
                value_target =  reward + self.gamma * value_next * (1-done)
                self.tq[index].update( state, action, value_target, self.alpha)