from agent.DQN import DQN
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import torch.optim as optim
from utils. networks import ANetwork

class MaxminDQNV2(DQN):
    def __init__(
        self,
        action_dim: int,
        device,
        gamma: float = 0.99,
        seed: int = 0,
        eps_start: float = 1.,
        eps_final: float = 0.1,
        eps_decay: float = 1000_000,
        lr: float = 0.0000625,
        batch_size: int = 16,
        memory_size: int = 100_000,
        stack_size: int = 4,
        load_model: bool = False,
        model_index: int = 5000,
        network_number=1,
        subset_size=1,
    ):
        super(MaxminDQNV2,
              self).__init__(action_dim, device, gamma, seed, eps_start,
                             eps_final, eps_decay, lr, batch_size, memory_size,
                             stack_size, load_model, model_index)
        self.network_number = network_number
        self.subset_size = subset_size

        self.Q_net = [None] * self.network_number
        self.Q_target = [None] * self.network_number
        self.optimizer = [None] * self.network_number
        for i in range(self.network_number):
            # init network
            self.Q_net[i] = ANetwork(action_dim).to(device)
            self.Q_target[i] = ANetwork(action_dim).to(device)
            # init network parameters
            self.Q_net[i].apply(ANetwork.init_weights)
            self.Q_target[i].load_state_dict(self.Q_net[i].state_dict())
            self.Q_target[i].eval()
            self.optimizer[i] = optim.Adam(
                self.Q_net[i].parameters(),
                lr=lr,
                eps=1.5e-4,
            )
    
    def get_q_next_values(self, next_batch):
        with torch.no_grad():
            q_min = self.Q_target[0](next_batch.float()) 
            for i in range(1, self.network_number):
                q = self.Q_target[i](next_batch.float())
                q_min = torch.min(q_min, q)
        value_next = q_min.max(1).values.detach()
#         print( value_next.shape)
        return value_next

#     find Qmin = min{1,2,...,M}Q(s,a)
    def get_choose_action_q_value(self, state):
        with torch.no_grad():
            q_min = self.Q_net[0](state)
            for i in range(1, self.network_number):
                q = self.Q_net[i](state) 
                q_min = torch.min(q_min, q)
        return q_min

    # select a subset S from {1,2,3...N} to update
    def learn(self):
        #range(0,m)生成[0,m-1]的数
        #random.sample (list,num) 从list中随机采样num个数
        subset = random.sample(list(range(self.network_number)),
                                    self.subset_size)
        #print(self.subset)
        #for i in subset
        for index in subset:
            #sample mini-batch
            state_batch, action_batch, reward_batch, next_batch, done_batch = \
                self.memory.sample(self.batch_size)

            values = self.Q_net[index](state_batch.float()).gather(
                1, action_batch)
            values_next = self.get_q_next_values(next_batch)
            expected = (self.gamma * values_next.unsqueeze(1)) * (
                1. - done_batch) + reward_batch
            loss = F.smooth_l1_loss(values, expected)
            #update network
            self.optimizer[index].zero_grad()
            loss.backward()
            for param in self.Q_net[index].parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer[index].step()

    #update Q target network
    def hard_copy(self):
        for i in range(self.network_number):
            self.Q_target[i].load_state_dict(self.Q_net[i].state_dict())

    #save model
    def save_model(self, path):
        state_dicts = {}
        for i in range(self.network_number):
            state_dicts[i] = self.Q_net[i].state_dict()
        torch.save(state_dicts, path)

    def load_model(self, path):
        state_dicts = torch.load(path)
        for i in range(self.network_number):
            self.Q_net[i].load_state_dict(state_dicts[i])
            self.Q_net[i] = self.Q_net[i].to(self.device)

class MaxminDQN_V2(DQN):
    def __init__(
        self,
        action_dim: int,
        device,
        gamma: float = 0.99,
        seed: int = 0,
        eps_start: float = 1.,
        eps_final: float = 0.1,
        eps_decay: float = 1000_000,
        lr: float = 0.0000625,
        batch_size: int = 16,
        memory_size: int = 100_000,
        stack_size: int = 4,
        load_model: bool = False,
        model_index: int = 5000,
        network_number=1,
        subset_size=1,
    ):
        super(MaxminDQN_V2,
              self).__init__(action_dim, device, gamma, seed, eps_start,
                             eps_final, eps_decay, lr, batch_size, memory_size,
                             stack_size, load_model, model_index)
        self.network_number = network_number
        self.subset_size = subset_size

        self.Q_net = [None] * self.network_number
        self.Q_target = [None] * self.network_number
        self.optimizer = [None] * self.network_number
        for i in range(self.network_number):
            # init network
            self.Q_net[i] = Network(action_dim).to(device)
            self.Q_target[i] = Network(action_dim).to(device)
            # init network parameters
            self.Q_net[i].apply(Network.init_weights)
            self.Q_target[i].load_state_dict(self.Q_net[i].state_dict())
            self.Q_target[i].eval()
            self.optimizer[i] = optim.Adam(
                self.Q_net[i].parameters(),
                lr=lr,
                eps=1.5e-4,
            )
    
    def get_q_next_values(self, next_batch):
        with torch.no_grad():
            q_min = self.Q_target[0](next_batch.float()) 
            for i in range(1, self.network_number):
                q = self.Q_target[i](next_batch.float())
                q_min = torch.min(q_min, q)
        value_next = q_min.max(1).values.detach()
#         print( value_next.shape)
        return value_next

#     find Qmin = min{1,2,...,M}Q(s,a)
    def get_choose_action_q_value(self, state):
        with torch.no_grad():
            q_min = self.Q_net[0](state)
            for i in range(1, self.network_number):
                q = self.Q_net[i](state) 
                q_min = torch.min(q_min, q)
        return q_min

    # select a subset S from {1,2,3...N} to update
    def learn(self):
        #range(0,m)生成[0,m-1]的数
        #random.sample (list,num) 从list中随机采样num个数
        subset = random.sample(list(range(self.network_number)),
                                    self.subset_size)
        #print(self.subset)
        #for i in subset
        for index in subset:
            #sample mini-batch
            state_batch, action_batch, reward_batch, next_batch, done_batch = \
                self.memory.sample(self.batch_size)

            values = self.Q_net[index](state_batch.float()).gather(
                1, action_batch)
            values_next = self.get_q_next_values(next_batch)
            expected = (self.gamma * values_next.unsqueeze(1)) * (
                1. - done_batch) + reward_batch
            loss = F.smooth_l1_loss(values, expected)
            #update network
            self.optimizer[index].zero_grad()
            loss.backward()
            for param in self.Q_net[index].parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer[index].step()

    #update Q target network
    def hard_copy(self):
        for i in range(self.network_number):
            self.Q_target[i].load_state_dict(self.Q_net[i].state_dict())

    #save model
    def save_model(self, path):
        state_dicts = {}
        for i in range(self.network_number):
            state_dicts[i] = self.Q_net[i].state_dict()
        torch.save(state_dicts, path)

    def load_model(self, path):
        state_dicts = torch.load(path)
        for i in range(self.network_number):
            self.Q_net[i].load_state_dict(state_dicts[i])
            self.Q_net[i] = self.Q_net[i].to(self.device)
            
            
