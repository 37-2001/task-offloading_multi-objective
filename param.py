import argparse

parser = argparse.ArgumentParser(description='DAG_ML')
#
 # -- Basic --
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--eps', type=float, default=1e-6,
                    help='epsilon (default: 1e-6)')
parser.add_argument('--num_proc', type=int, default=1,
                    help='number of processors (default: 1)')
parser.add_argument('--num_exp', type=int, default=5,
                    help='number of experiments (default: 10)')
parser.add_argument('--job_folder', type=str, default='./spark_env/4-6/',
                    help='job folder path (default: ./spark_env/7-9/)')
parser.add_argument('--result_folder', type=str, default='./results/',
                    help='Result folder path (default: ./results)')
parser.add_argument('--model_folder1', type=str, default='./models1/',
                    help='Model folder path (default: ./models)')
parser.add_argument('--model_folder2', type=str, default='./models2/',
                    help='Model folder path (default: ./models)')
parser.add_argument('--model_folder3', type=str, default='./models3/',
                    help='Model folder path (default: ./models)')

# -- MECEnvironment --
parser.add_argument('--num_mecs', type=int, default=3,
                    help='Number of total servers (default: 3)')  # 服务器的个数
parser.add_argument('--bandwidth', type=float, default=2e6,
                    help='带宽 (default: 2MHZ')  # 带宽
parser.add_argument('--u', type=float, default=1e-27,
                    help='带宽 (default: 2MHZ')  # 带宽
parser.add_argument('--noise', type=float, default=1e-13,
                    help='噪声的功率 (default: 7.94e-15w)')  # 噪声
parser.add_argument('--transmission_power', type=float, default=0.2,
                    help='发送功率 (default: 0.2W)')  # 发送功率
parser.add_argument('--mec_power', type=float, default=0.2,
                    help='发送功率 (default: 0.2W)')
parser.add_argument('--mec_capacity', type=float,
                    default=[4e9, 6e9, 8e9, 1.2e10], nargs='+',
                    help='各个服务器的频率,单位为HZ (default:[4e9,6e9,8e9,1e10])')

parser.add_argument('--trans_rate', type=float, default=20,
                    help='带宽 (default: 20Mbps')  # 服务器间的传输速率
parser.add_argument('--cloud_rate', type=float, default=5,
                    help='带宽 (default: 10Mbps')  # 服务器间的传输速率
parser.add_argument('--cloud_cap', type=float, default=1.2e10,
                    help='带宽 (default: 10GHZ')  # 服务器间的传输速率

parser.add_argument('--mec_radius', type=float, default=100,
                    help='半径 (default: 100)')  # 半径
parser.add_argument('--mec_x', type=float,
                    default=[250,400,100,100,400,250,100], nargs='+',
                    help='mec的横坐标 (default: [100.0,150.0,50.0])')
#[50,50,150,150,100,200,0]
parser.add_argument('--mec_y', type=float,
                    default=[250,400,100,400,100,400,250], nargs='+',
                    help='mec的纵坐标 (default:[50,20,30])')

# -- job --
parser.add_argument('--num_init_dags', type=int, default=0,
                    help='Number of initial DAGs in system (default:0)')
parser.add_argument('--num_stream_dags', type=int, default=10,
                    help='number of streaming DAGs (default:3)')
parser.add_argument('--diff_reward_enabled', type=int, default=0,
                    help='Enable differential reward (default: 0)')
parser.add_argument('--bili', type=float, default=0.5,
                    help='Enable differential reward (default: 0)')

# -- Evaluation --
parser.add_argument('--test_schemes', type=str,
                    default=['learn'], nargs='+',
                    help='Schemes for testing the performance')

# -- Visualization --
parser.add_argument('--canvs_visualization', type=int, default=1,
                    help='Enable canvs visualization (default: 1)')
parser.add_argument('--canvas_base', type=int, default=-10,
                    help='Canvas color scale bottom (default: -10)')

# -- Learning --
parser.add_argument('--node_input_dim', type=int, default=5,
                    help='node input dimensions to graph embedding (default: 4)')
parser.add_argument('--job_input_dim', type=int, default=5,
                    help='mec input dimensions to graph embedding (default: 4)')
#32，64，32
parser.add_argument('--hid_dims', type=int, default=[16, 8], nargs='+',
                    help='hidden dimensions throughout graph embedding (default: [16, 8])')
parser.add_argument('--output_dim', type=int, default=8,
                    help='output dimensions throughout graph embedding (default: 8)')
parser.add_argument('--max_depth', type=int, default=6,
                    help='Maximum depth of root-leaf message passing (default: 6)')
#基线ema----------------------------------------------------------------------------
parser.add_argument('--alpha_start', type=float, default=0.4,
                    help='alpha_start (default: 0.4)')
parser.add_argument('--alpha_end', type=float, default=0.05,
                    help='alpha_start (default: 0.4)')
parser.add_argument('--decay_episodes', type=float, default=3000,
                    help='decay_episodes (default: 3000)')
parser.add_argument('--window_size', type=float, default=30,
                    help='window_size (default: 30)')
#----------------------------------------------------------------------------------
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate (default: 0.001)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size (default: 64)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor (default: 1)')
parser.add_argument('--entropy_weight_init', type=float, default=1,
                    help='Initial exploration entropy weight (default: 1)')
parser.add_argument('--entropy_weight_min', type=float, default=0.00001,
                    help='Final minimum entropy weight (default: 0.0001)')
parser.add_argument('--entropy_weight_decay', type=float, default=3e-4,
                    help='Entropy weight decay rate (default: 1e-3)')
parser.add_argument('--worker_num_gpu', type=int, default=0,
                    help='Number of GPU cores used in worker (default: 0)')
parser.add_argument('--master_gpu_fraction', type=float, default=0.5,
                    help='Fraction of memory master uses in GPU (default: 0.5)')
parser.add_argument('--average_reward_storage_size', type=int, default=100000,
                    help='Storage size for computing average reward (default: 100000)')
parser.add_argument('--reset_prob', type=float, default=0,
                    help='Probability for episode to reset (after x seconds) (default: 0)')
parser.add_argument('--reset_prob_decay', type=float, default=0,
                    help='Decay rate of reset probability (default: 0)')
parser.add_argument('--reset_prob_min', type=float, default=0,
                    help='Minimum of decay probability (default: 0)')
parser.add_argument('--num_agents', type=int, default=1,
                    help='Number of parallel agents (default: 16)')
parser.add_argument('--num_ep', type=int, default=50000,
                    help='Number of training epochs (default: 100000)')
parser.add_argument('--saved_model', type=str, default=None,
                    help='Path to the saved tf model (default: None)')
parser.add_argument('--check_interval', type=float, default=0.01,
                    help='interval for master to check gradient report (default: 10ms)')
parser.add_argument('--model_save_interval', type=int, default=1000,
                    help='Interval for saving Tensorflow model (default: 1000)')
parser.add_argument('--num_saved_models', type=int, default=1000,
                    help='Number of models to keep (default: 1000)')
parser.add_argument('--exec_cap', type=int, default=1,
                    help='Number of total executors (default: 1)')
# -- Spark interface --
parser.add_argument('--scheduler_type', type=str, default='dynamic_partition',
                    help='type of scheduling algorithm (default: dynamic_partition)')

args = parser.parse_args()
