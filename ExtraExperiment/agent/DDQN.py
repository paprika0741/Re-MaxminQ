from agent.DQN import DQN
from utils. networks import Network
from utils. memory import ReplayMemory


class DDQN(DQN):
    '''
    Implementation of Double DQN with target network and replay buffer
    '''
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
    ):
        super(DDQN,
              self).__init__(action_dim, device, gamma, seed, eps_start,
                             eps_final, eps_decay, lr, batch_size, memory_size,
                             stack_size, load_model, model_index)

    # compute q value for next state batch
    def get_q_next_values(self, next_batch):
        next_action = self.Q_net(next_batch.float()).detach().argmax(
            dim=1, keepdim=True)
        value_next = self.Q_target(next_batch.float()).gather(1, next_action).squeeze()
        #计算 target 有unsqueeze
        # 这里要squeeze
        return value_next 


