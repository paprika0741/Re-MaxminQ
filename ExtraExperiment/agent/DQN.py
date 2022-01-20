import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils. networks import Network
from utils. memory import ReplayMemory


class DQN(object):
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
        model_index: int = 0,
    ):
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.seed = seed
        #epsilon for epsilon greedy policy
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.eps = eps_start

        self.r = random.Random()
        self.r.seed(seed)
        # init network
        self.Q_net = Network(action_dim).to(device)
        self.Q_target = Network(action_dim).to(device)
        # init network parameters
        self.Q_net.apply(Network.init_weights)
        self.Q_target.load_state_dict(self.Q_net.state_dict())
        #
        self.Q_target.eval()

        self.optimizer = optim.Adam(
            self.Q_net.parameters(),
            lr=lr,
            eps=1.5e-4,
        )

        self.batch_size = batch_size
        #replay buffer
        self.memory = ReplayMemory(stack_size + 1, memory_size, device)

    # epsilon decay
    def epsilon_decay(self):
        self.eps -= (self.eps_start - self.eps_final) / self.eps_decay
        self.eps = max(self.eps, self.eps_final)

    # comput q value for state , use Q net work
    def get_q_values(self, state_batch, action_batch):
        return self.Q_net(state_batch.float()).gather(1, action_batch)

    # comput q value for next state , use Q target network
    def get_q_next_values(self, next_batch):
        return self.Q_target(next_batch.float()).max(1).values.detach()

    #  comput q value for current state, use q network,to select action
    def get_choose_action_q_value(self, state):
        return self.Q_net(state)

    '''
    when  trainig, use policy greedy policy,
    when testing use greedy policy 
    '''

    def choose_action(self, state, training: bool, testing: bool):
        if training :
            # epsilon decay
            self.epsilon_decay()
            # epsilon greedy policy
            if self.r.random() > self.eps:
                with torch.no_grad():
                    return self.get_choose_action_q_value(state).max(
                        1).indices.item()
            else:
                return self.r.randint(0, self.action_dim - 1)
        # test epsilon=0.01 fixed
        if testing:
            if self.r.random() > 0.01:
                return self.get_choose_action_q_value(state).max(
                    1).indices.item()
            else:
                return self.r.randint(0, self.action_dim - 1)
        #random policy
        return  self.r.randint(0, self.action_dim - 1)

    def learn(self):
        #sample a mini batch
        state_batch, action_batch, reward_batch, next_batch, done_batch = \
            self.memory.sample(self.batch_size)
        # compute current q valuse [32,1]
        values = self.get_q_values(state_batch, action_batch)
        # compute next q valuse [32]
        values_next = self.get_q_next_values(next_batch)
        # compute q target
        expected = (self.gamma * values_next.unsqueeze(1)) * (
            1. - done_batch) + reward_batch
        # TD - learning
        loss = F.smooth_l1_loss(values, expected)
        # update q network
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.Q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    #update Q target network, use hard copy
    def hard_copy(self):
        self.Q_target.load_state_dict(self.Q_net.state_dict())

    #save model
    def save_model(self, path):
        torch.save(self.Q_net.state_dict(), path)

    #load model
    def save_model(self, path):
        state_dicts = torch.load(path)
        self.Q_net.load_state_dict(state_dicts)
        self.Q_net = self.Q_net.to(self.device)
