# 创建带有 LiDAR 数据的数据集

## 研究 leaderboard 部分代码

`leaderboard_evaluator.py`
```py
StatisticsManager.save_sensors(sensors, endpoint)
                                 ⬇️
                               sensors
LeaderboardEvaluator._load_and_run_scenario(args, config):
    self.statistics_manager.save_sensors(self.sensor_icon, self.checkpoint)

    self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]

    self.sensors = self.agent_instance.sensors()

    self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config)

    1️⃣ self.module_agent
    self.module_agent = importlib.import_module(module_name)
    module_name = os.path.basename(args.agent).split('.')[0]

    2️⃣ agent_class_name
    agent_class_name = getattr(self.module_agent, 'get_entry_point')()

    3️⃣ args

    ➡️ args.agent

    def __init__(self, args, statistics_manager):

    leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
    arguments = parser.parse_args()
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True)
```

`data_collection.sh`
```sh
    --agent=${TEAM_AGENT}
    export TEAM_AGENT=team_code/roach_ap_agent.py
```

所以原数据集产生的配置依赖于 `roach_ap_agent.py`  
我们需要修改 `leaderboard/team_code/roach_ap_agent.py` 来支持 LiDAR 数据
