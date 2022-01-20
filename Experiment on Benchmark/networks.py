import torch
import torch.nn as nn
import torch.nn.functional as F


class Network2D(nn.Module):
    def __init__(self, action_dim):
        super(Network2D, self).__init__()
        self.__conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, bias=False)
        self.__fc2 = nn.Linear(107584, action_dim)

    def forward(self, x):
        x = F.relu(self.__conv1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.__fc2(x))
        return x

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

                    
class Network (nn.Module):

    def __init__(self, action_dim ):
        super(Network, self).__init__()
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        linear_input_size = 7 * 7 * 64
        self.__fc4 = nn.Linear(linear_input_size, action_dim)
        

    def forward(self, x):
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        x = F.relu(self.__fc4(x.view(x.size(0), -1)))
        return x

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            
            
            

class ANetwork(nn.Module):

    def __init__(self, action_dim ):
        super(ANetwork, self).__init__()
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.__fc1 = nn.Linear(64*7*7, 512)
        self.__fc2 = nn.Linear(64*7*7, 512)
        
        self.__advantage_layer = nn.Linear(512, action_dim)
        # value layer
        self.__value_layer = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        adv = F.relu(self.__fc1(x.view(x.size(0), -1)))
        val = F.relu(self.__fc2(x.view(x.size(0), -1)))
        
        advantage = self.__advantage_layer(adv)
        value = self.__value_layer(val)
        q = value + advantage -  advantage.mean(dim=-1, keepdim=True)
        return q

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")