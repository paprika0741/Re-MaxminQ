from agent.DQN import DQN
import torch
from utils. networks import Network
from utils. memory import ReplayMemory


class AveragedDQN(DQN):
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
        network_number=1,
    ):
        super(AveragedDQN,
              self).__init__(action_dim, device, gamma, seed, eps_start,
                             eps_final, eps_decay, lr, batch_size, memory_size,
                             stack_size, load_model, model_index)
        self.network_number = network_number
        # one online network, one optimizer  
        # create network_number(K)  Q target networks 
        self.Q_target = [None] * self.network_number
        self.update_target_net_index = 0
        for i in range(self.network_number):
            # init K target netwoks 
            self.Q_target[i] = Network(action_dim).to(device)
            # init network parameters
            self.Q_target[i].load_state_dict(self.Q_net.state_dict())
            self.Q_target[i].eval()
        
    '''
    use the K previously learned Q-values estimates to produce the current action-value estimate
    use K target network to  calculate average Q_next
    '''       
    def get_q_next_values(self,next_batch):
        with torch.no_grad():
            q_sum = self.Q_target[0](next_batch) 
            for i in range(1, self.network_number):
                q_sum += self.Q_target[i](next_batch)

        return (q_sum/self.network_number).max(1).values.detach()
    
    #update Q target network , the 
    def hard_copy(self):
        self.Q_target[self.update_target_net_index].load_state_dict(self.Q_net.state_dict())
        self.update_target_net_index = (self.update_target_net_index + 1) % self.network_number

