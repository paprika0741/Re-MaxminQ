### env

 make a new environment `MountainCarMyEasyVersion-v0` with different parameters by adapting one of the calls to `register` found in gym/gym/envs/__init__.py:

```
register(
    id="MountainCarMyEasyVersion-v0",
    entry_point="gym.envs.classic_control:MountainCarEnv",
    max_episode_steps=600,
    reward_threshold=-110.0,
)
```

```
eg:
python main.py --agent=DQ --sigma=10 --step_size=0.005 --gamma=1 --seed=0 --memory_size=100 --batch_size=8
```

```python
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

```

