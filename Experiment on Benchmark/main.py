from tqdm import tqdm
import random
import torch
from collections import deque
from DQN import DQN
from DDQN import DDQN
from DUE import DUE
from MaxminDQN import MaxminDQN
from AveragedDQN import AveragedDQN
from V2 import V2
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
from utils_env import MyEnv
#  'Seaquest-v0'   'Asterix-v0':   'BreakoutNoFrameskip-v4'  'SpaceInvadersNoFrameskip-v0': 6  是不是能直接改 我不确定...
from utils_env import  AtarEnv


def getTimeNow():
    timenow = str(datetime.now())[0:-10]
    timenow =   timenow[0:13] + '_' + timenow[-2::]
    return timenow
    



SAVE_PREFIX = "./models"

if os.path.exists(SAVE_PREFIX): shutil.rmtree(SAVE_PREFIX)
'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--env_name',
                    type=str,
                    default="BreakoutNoFrameskip-v4",
                    help='Env name')

parser.add_argument('--agent',
                    type=str,
                    default="DQN",
                    help='choose from DQN DDQN AverageQ MaxminQ', choices=["DQN", "V2","DDQN", "AverageQ", "MaxminQ","DUE"])

parser.add_argument('--write',
                    type=bool,
                    default=True,
                    help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=bool, default=False, help='Render or Not')
parser.add_argument('--load_model',
                    type=bool,
                    default=False,
                    help='Load pretrained model or Not')
parser.add_argument('--model_index',
                    type=int,
                    default=5000,
                    help='which model to load')

parser.add_argument('--gamma',
                    type=float,
                    default=0.99,
                    help='Discounted Factor')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--eps_start', type=float, default=1, help='epsilon start')
parser.add_argument('--eps_final', type=float, default=0.1, help='epsilon end')
parser.add_argument('--eps_decay',
                    type=float,
                    default=100_000,
                    help='epsilon decay')
#todo lr modify
parser.add_argument('--lr',
                    type=float,
                    default=0.00025,
                    help='Learning rate')
parser.add_argument('--net_num',
                    type=int,
                    default=2,
                    help='number of networks')
parser.add_argument('--set_num',
                    type=int,
                    default=1,
                    help='number of subset_networks')

parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='mempry sample batch size')
parser.add_argument('--memory_size',
                    type=int,
                    default=100_000,
                    help='mempry size')
parser.add_argument('--stack_size', type=int, default=4, help='stack size')

parser.add_argument('--warmup_step',
                    type=int,
                    default=50_00,
                    help='warm up steps')
parser.add_argument('--target_update',
                    type=int,
                    default=1000,
                    help='target network update interval')
parser.add_argument('--policy_update',
                    type=int,
                    default=4,
                    help='policy network update interval')
parser.add_argument('--evaluate_freq',
                    type=int,
                    default=10_000,
                    help='evaluate policy frequency')
parser.add_argument('--max_step',
                    type=int,
                    default=5_000_000,
                    help='max training steps')
#Todo do  
parser.add_argument('--no_op_max',
                    type=int,
                    default=30,
                    help='"do nothing" actions to be performed by the agent at the start of an episode')

parser.add_argument('--eval_epoch',
                    type=int,
                    default=100,
                    help='return is averaged over the last N episodes')
parser.add_argument('--device',
                    type=str,
                    default="cuda",
                    help='if more than one gpu, choose one cuda:1,cuda:2...')
parser.add_argument('--fix',
                    type=str,
                    default="",
                    help='some desription')
opt = parser.parse_args()
print(opt)
device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
print(getTimeNow() )
fname =  opt.env_name[:2] +"_" + opt.agent + "net" + str(opt.net_num) +"set" +str(opt.set_num)+ "avg_"+ str(opt.eval_epoch) +getTimeNow() + ".txt"

 
torch.cuda.empty_cache()


def main():
    rand = random.Random()
    rand.seed(opt.seed)
    new_seed = lambda: rand.randint(0, 1000_000)
    torch.manual_seed(new_seed())
    #init env
    env = AtarEnv(opt.env_name, device)
    obs_queue: deque = deque(maxlen=5)
    done = True
    #init agent
    if opt.agent  == "DQN":
        agent = DQN(
            action_dim=env.get_action_dim(),
            device=device,
            gamma=opt.gamma,
            seed=opt.seed,
            eps_start=opt.eps_start,
            eps_final=opt.eps_final,
            eps_decay=opt.eps_decay,
            lr=opt.lr,
            batch_size=opt.batch_size,
            memory_size=opt.memory_size,
            stack_size=opt.stack_size,
            load_model=opt.load_model,
            model_index=opt.model_index,
        )
    elif opt.agent  == "DDQN":
        agent = DDQN(
            action_dim=env.get_action_dim(),
            device=device,
            gamma=opt.gamma,
            seed=opt.seed,
            eps_start=opt.eps_start,
            eps_final=opt.eps_final,
            eps_decay=opt.eps_decay,
            lr=opt.lr,
            batch_size=opt.batch_size,
            memory_size=opt.memory_size,
            stack_size=opt.stack_size,
            load_model=opt.load_model,
            model_index=opt.model_index,
        )
    elif opt.agent  == "DUE":
        agent = DUE(
            action_dim=env.get_action_dim(),
            device=device,
            gamma=opt.gamma,
            seed=opt.seed,
            eps_start=opt.eps_start,
            eps_final=opt.eps_final,
            eps_decay=opt.eps_decay,
            lr=opt.lr,
            batch_size=opt.batch_size,
            memory_size=opt.memory_size,
            stack_size=opt.stack_size,
            load_model=opt.load_model,
            model_index=opt.model_index,
        )
    elif opt.agent  == "AverageQ":
        agent = AveragedDQN(
            action_dim=env.get_action_dim(),
            device=device,
            gamma=opt.gamma,
            seed=opt.seed,
            eps_start=opt.eps_start,
            eps_final=opt.eps_final,
            eps_decay=opt.eps_decay,
            lr=opt.lr,
            batch_size=opt.batch_size,
            memory_size=opt.memory_size,
            stack_size=opt.stack_size,
            load_model=opt.load_model,
            model_index=opt.model_index,
            network_number=opt.net_num,
        )
    elif opt.agent  == "MaxminQ":
        agent = MaxminDQN(
            action_dim=env.get_action_dim(),
            device=device,
            gamma=opt.gamma,
            seed=opt.seed,
            eps_start=opt.eps_start,
            eps_final=opt.eps_final,
            eps_decay=opt.eps_decay,
            lr=opt.lr,
            batch_size=opt.batch_size,
            memory_size=opt.memory_size,
            stack_size=opt.stack_size,
            load_model=opt.load_model,
            model_index=opt.model_index,
            network_number=opt.net_num,
            subset_size=opt.set_num,
        )
    elif opt.agent  == "V2":
        agent = V2(
            action_dim=env.get_action_dim(),
            device=device,
            gamma=opt.gamma,
            seed=opt.seed,
            eps_start=opt.eps_start,
            eps_final=opt.eps_final,
            eps_decay=opt.eps_decay,
            lr=opt.lr,
            batch_size=opt.batch_size,
            memory_size=opt.memory_size,
            stack_size=opt.stack_size,
            load_model=opt.load_model,
            model_index=opt.model_index,
            network_number=opt.net_num,
            subset_size=opt.set_num,
        )
    else:
        print("wrong agent name,choose from DQN DDQN AveragedQ MaxminQ")
        
        return

    progressive = tqdm(range(opt.max_step),
                       total=opt.max_step,
                       ncols=50,
                       leave=False,
                       unit="b")

    for step in progressive:
        if done:
            observations, _, _ = env.reset()
            epoch_steps = 0
            for obs in observations:
                obs_queue.append(obs)
        epoch_steps+=1
        training = len(agent.memory) > opt.warmup_step
        state = env.make_state(obs_queue).to(device).float()
        # '"do nothing" actions to be performed by the agent at the start of an episode'
        action = agent.choose_action(state, training, testing= False)
         
            
        obs, reward, done = env.step(action)
        obs_queue.append(obs)
        agent.memory.push(env.make_folded_state(obs_queue), action, reward,
                          done)

        if step % opt.policy_update == 0 and training:
            agent.learn()

        if step % opt.target_update == 0:
            agent.hard_copy()

        if step % opt.evaluate_freq == 0:
            avg_reward, frames = env.evaluate(obs_queue,
                                              agent,
                                              num_episode=opt.eval_epoch, #100
                                              render=opt.render)
            with open(  fname  , "a") as fp:
                fp.write(
                    f"{step//opt.evaluate_freq:3d} {step:8d} {avg_reward:.1f}\n"
                )
            done = True
#             print(avg_reward)
        # if step % 1000_000==0:
        #     avg_reward, frames = env.evaluate(obs_queue,
        #                                       agent,
        #                                       num_episode=100,
        #                                       render=opt.render)
        #     with open(opt.agent_name +"avg100" + ".txt", "a") as fp:
        #         fp.write(
        #             f"{step//opt.evaluate_freq:3d} {step:8d} {avg_reward:.1f}\n"
        #         )
        #     done = True
        #     print(avg_reward)

if __name__ == '__main__':
    main()