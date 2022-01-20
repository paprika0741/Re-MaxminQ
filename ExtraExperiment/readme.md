```
run：
python main.py --agent=MaxminQ  
```



```python
parser.add_argument('--env_name',
                    type=str,
                    default="BreakoutNoFrameskip-v4",
                    help='Env name')

parser.add_argument('--agent',
                    type=str,
                    default="DQN",
                    help='choose from DQN DUE DDQN AverageQ MaxminQ', choices=["DQN", "DUE","DDQN", "AverageQ", "MaxminQ","M2"])

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
                    default=1_000_000,
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
#except for mem
parser.add_argument('--memory_size',
                    type=int,
                    default=1000_00,
                    help='mempry size')
parser.add_argument('--stack_size', type=int, default=4, help='stack size')

parser.add_argument('--warmup_step',
                    type=int,
                    default=50_000,
                    help='warm up steps')
parser.add_argument('--target_update',
                    type=int,
                    default=10_000,
                    help='target network update interval')
parser.add_argument('--policy_update',
                    type=int,
                    default=4,
                    help='policy network update interval')
parser.add_argument('--evaluate_freq',
                    type=int,
                    default=10_000,
                    help='evaluate policy frequency')
#Todo do  
parser.add_argument('--no_op_max',
                    type=int,
                    default=30,
                    help='"do nothing" actions to be performed by the agent at the start of an episode')

parser.add_argument('--eval_epoch',
                    type=int,
                    default=3,
                    help='return is averaged over the last N episodes')

parser.add_argument('--device',
                    type=str,
                    default="cuda",
                    help='if more than one gpu, choose one cuda:1,cuda:2...')

parser.add_argument('--max_step',
                    type=int,
                    default=5_000_000,
                    help='max training steps')
parser.add_argument('--fix',
                    type=str,
                    default="",
                    help='some desription')

```

