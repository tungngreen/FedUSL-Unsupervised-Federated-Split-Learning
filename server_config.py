import argparse
import time

args = argparse.ArgumentParser(fromfile_prefix_chars='@')

args.add_argument(
    '--num_clients',
    type = int,
    default = 5
)
args.add_argument(
    '--num_active_clients',
    type = int,
    default = 5
)
args.add_argument(
    '--run_idx',
    type = int,
    default = 0,
)
args.add_argument(
    '--server_avg',
    type = int,
    default = 0
)
args.add_argument(
    '--client_avg',
    type = int,
    default = 0
)
args.add_argument(
    '--seed',
    type = lambda s: [int(item) for item in s.split(',')],
    default = '325,665,576'
)
args.add_argument(
    '--test_size',
    type = int,
    default = 2000
)
args.add_argument(
    '--generated_label',
    type = str,
    default = '/home/hihi/FedUSL/data/resnet56_generated.txt'
)
args.add_argument(
    '--data_dir',
    type = str,
    default = '/home/hihi/FedUSL/data'
)
args.add_argument(
    '--ckpt_save_dir',
    type = str,
    default = '/home/hihi/FedUSL/saved_ckpts'
)
args.add_argument(
    '--ckpt_interval',
    type = int,
    default = 5
)
args.add_argument(
    '--output_dir',
    type = str,
    default = '/home/hihi/FedUSL/saved_results'
)
args.add_argument(
    '--device',
    type = str,
    default = 'cuda:1'
)
args.add_argument(
    '--num_workers',
    type = int,
    default = 16
)
args.add_argument(
    '--model_name',
    type = str,
    default = 'resnet18'
)
args.add_argument(
    '--pretrained_model',
    type = int,
    default = 1
)
# Training hyper parameters
args.add_argument(
    '--num_epochs',
    type = int,
    default = 50,
)
args.add_argument(
    '--optimizer',
    type = str,
    default = 'sgd'
)
args.add_argument(
    '--batch_size',
    type = int,
    default = 32
)
args.add_argument(
    '--learning_rate',
    type = float,
    default = 0.1
)
args.add_argument(
    '--client_lr_scaled',
    type = int,
    help = 'This decides if the learning rate for clients should be scaled.',
    default = 0
)
args.add_argument(
    '--momentum',
    type = float,
    default = 0.9
)
args.add_argument(
    '--weight_decay',
    type = float,
    default = 0.001
)
args.add_argument(
    '--is_training',
    type = int,
    default = 1
)
args.add_argument(
    '--is_testing',
    type = int,
    default = 1
)
args.add_argument(
    '--resume',
    type = str,
    default = None
)
config, _ = args.parse_known_args()