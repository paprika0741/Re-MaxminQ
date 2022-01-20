# Import common libraries
import sys
import gym
import numpy as np
from agent import QLearningAgent, DoubleQ, AverageQ, MaxminQ,DoubleQ1
import argparse
from datetime import datetime

def getTimeNow():
    timenow = str(datetime.now())[0:-10]
    timenow =   timenow[0:13] + '_' + timenow[-2::]
    return timenow

parser = argparse.ArgumentParser()

parser.add_argument('--agent',
                    type=str,
                    default="Q",
                    choices=["Q", "DQ" , "AverageQ", "MaxminQ"],
                    help='choose from Q DQ  AverageQ MaxminQ')
parser.add_argument('--sigma', type=int, default=10, help='reward variance')
parser.add_argument('--step_size',
                    type=float,
                    default=0.005,
                    choices=[0.005, 0.01, 0.02, 0.04, 0.08],
                    help='function approximator step-size')
parser.add_argument('--gamma', type=float, default=1, help='discount factor')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--eps_start',
                    type=float,
                    default= 1,
                    help='epsilon start')
parser.add_argument('--eps_final', type=float, default=0.01, help='epsilon end')
parser.add_argument('--eps_decay',
                    type=float,
                    default=50000,
                    help='epsilon decay')
parser.add_argument('--memory_size', type=int, default=100, help='mempry size')
parser.add_argument('--batch_size',
                    type=int,
                    default=8,
                    help='memory sample batch size')
parser.add_argument('--net_num', type=int, default=2, help='number of table')
parser.add_argument('--set_num',
                    type=int,
                    default=1,
                    help='number of subset_networks')
parser.add_argument('--run', type=int, default=100, help='experiment runs')
parser.add_argument('--update_fre', type=int, default=20, help='update frequency')

opt = parser.parse_args()
print(opt)
env = gym.make("MountainCarMyEasyVersion-v0")
env.seed(opt.seed)
n_bins = 8
bins = tuple([n_bins] * env.observation_space.shape[0])
offset_pos = (env.observation_space.high - env.observation_space.low) / (8)

tiling_specs = [(bins, -offset_pos),
                (bins, tuple([0.0] * env.observation_space.shape[0])),
                (bins, offset_pos)]

# tq = TiledQTable(env.observation_space.low,
#                 env.observation_space.high,
#                 tiling_specs,
#                 env.action_space.n)


def initagent():
    if opt.agent == "Q":
        return QLearningAgent(env, tiling_specs, opt.step_size, opt.gamma,
                              opt.eps_start, opt.eps_decay, opt.eps_final,
                              opt.seed, opt.batch_size, opt.memory_size)
    elif opt.agent == "DoubleQ":
        return DoubleQ(env, tiling_specs, opt.step_size, opt.gamma,
                       opt.eps_start, opt.eps_decay, opt.eps_final, opt.seed,
                       opt.batch_size, opt.memory_size)
    elif opt.agent == "DQ":
        return DoubleQ1(env, tiling_specs, opt.step_size, opt.gamma,
                       opt.eps_start, opt.eps_decay, opt.eps_final, opt.seed,
                       opt.batch_size, opt.memory_size, opt.update_fre)
    elif opt.agent == "AverageQ":
        return AverageQ(env, tiling_specs, opt.step_size, opt.gamma,
                        opt.eps_start, opt.eps_decay, opt.eps_final, opt.seed,
                        opt.batch_size, opt.memory_size, opt.net_num)

    elif opt.agent == "MaxminQ":
        return MaxminQ(env, tiling_specs, opt.step_size, opt.gamma,
                       opt.eps_start, opt.eps_decay, opt.eps_final, opt.seed,
                       opt.batch_size, opt.memory_size, opt.net_num,
                       opt.set_num)
    else:
        print("wrong agent name, choose from Q DoubleQ AverageQ MaxminQ")

fname =  "run_" + str(opt.run) + "_" + opt.agent + "sigma_" + \
                str(opt.sigma) + "_batch" + str(opt.batch_size) + "_net" +\
                str(opt.net_num) + "_set" + str(opt.set_num) +getTimeNow() +  ".txt"

avg = 0
for i in range(opt.run):
    agent = initagent()
    for j in range( 1000 ):
        epoch_step = agent.train(opt.sigma)
        if j % 100 == 0:
            print("\r" + opt.agent + "  Run{}  Episode {} | Epoch Steps: {}".format(
                i, j, epoch_step))
        with open( fname,
                "a") as fp:
            fp.write(f"{ j:4d} {epoch_step:4d} {epoch_step:.1f}\n")
    avg += epoch_step
    print(epoch_step)

with open(fname,"a") as fp:
    fp.write("==============================================\n")
    fp.write("total steps for runs \n")
    fp.write(f"{ avg  :4f}")

print(avg / opt.run)
