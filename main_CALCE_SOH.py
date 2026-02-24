import os
import torch
import math
from exp.exp_main import exp_main
import numpy as np
import random

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


train_battery = '35'
start = 0
name = 'CALCE'
end = 700
task = 'SOH'
battery = 'CALCE'
class Args:
    def __init__(self):
        self.model = 'S_Mamba'
        # self.model = 'CNN_LSTM'
        self.model_id = '0_uncertain_S_Mamba_LSTM_{}#{}'.format(name,train_battery)
        self.task = task
        self.battery = battery
        self.results_path = './results/{}/'.format(battery)
        self.checkpoints = './checkpoints/'
        self.root_path = './datasets/{}/'.format(battery)
        self.train_battery = train_battery
        self.train_battery_now = train_battery
        self.data_path = 'battery_data_frames[].csv'
        self.start = start
        self.end = end
        self.norm = True

        self.epochs = 40  # 60
        self.patience = 40
        self.optim = 'adam'

        self.warmup_epochs = 10
        self.min_lr = 0
        self.smoothing_learning_rate = 0
        self.damping_learning_rate = 0
        self.lradj = 'exponential_with_warmup'

        self.learning_rate = 0.0001  # 0.0001
        self.batch_size = 64  # 16
        self.d_ff = 512  # 4
        self.d_mlp = 256  # 32
        self.d_model = 512   # 16 384
        self.n_heads = 4  # 1
        self.e_layers = 1  # 6
        self.d_layers = 1
        self.dropout = 0.8
        self.d_state = 16

        self.activation = 'relu'
        self.output_attention = False
        self.pred_len = 1
        self.seq_len = 1  # 128 67
        self.label_len = self.seq_len
        self.c_out = 1
        self.use_norm = False

        self.grad_clip = 1.0  # 1 
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'

args = Args()
exp = exp_main(args)
seed_everything(11)
print('<<<<<<<<<<<<<<<<<<<<<<<<< start training >>>>>>>>>>>>>>>>>>>>>>>>>')
exp.train()
# exp.test(Time_record=False)

torch.cuda.empty_cache()