import argparse
import pickle
import collections
import logging
import math
import os,sys,time
import random
from sys import maxsize
import pickle
import numpy as np
import torch
import torch.nn as nn
from utils.general import init_logger, make_sure_path_exists
sys.path.append('./water_quality/')

from aegan import AeGAN
from metrics.visualization_metrics import visualization

DEBUG_SCALE = 512
 
task_name=time.strftime("%Y-%m-%d-%H-%M-%S")
root_dir=r'set_your_root_dir'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ===-----------------------------------------------------------------------===
# Set up logging
# ===-----------------------------------------------------------------------===
logger = init_logger(root_dir)

# ===-----------------------------------------------------------------------===
# Log some stuff about this run
# ===-----------------------------------------------------------------------===
logger.info(' '.join(sys.argv))
logger.info('')
# logger.info(options)

python_seed=random.randrange(maxsize)
random.seed(python_seed)
np.random.seed(python_seed % (2 ** 32 - 1))
logger.info('Python random seed: {}'.format(python_seed))

# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dir_dataset=r'data\DATA_FOR_GAN_WITH_DO_TRAIN.pkl'
dataset = pickle.load(open(dir_dataset, "rb"))
train_set=dataset["train_set"]
dynamic_processor=dataset["dynamic_processor"]
static_processor=dataset["static_processor"]
train_set.set_input("sta","dyn","seq_len")

    
# ===-----------------------------------------------------------------------===
# Build model and trainer
# ===-----------------------------------------------------------------------===

# params=vars(options)
params = {
    'dataset': dir_dataset,
    'devi': '0',
    'epochs': 1000,
    'iterations': 15000,
    'd_update': 5,
    'log_dir': '../stanice_result',
    'task_name': time.strftime("%Y-%m-%d-%H-%M-%S"),
    'python_seed':random.randrange(maxsize),
    'debug': False,
    'eval_ae': False,
    'fix_ae': None,
    'fix_gan': None,
    'ae_batch_size': 128,
    'gan_batch_size': 512,
    'embed_dim':96,
    'hidden_dim':24,
    'layers': 3,
    'ae_lr':0.001,
    'weight_decay':0,
    'scale': 1,
    'dropout': 0.0,
    'gan_lr': 0.0001,
    'gan_alpha': 0.99,
    'noise_dim':96,
    'static_processor': static_processor,
    'dynamic_processor': dynamic_processor,
    'root_dir':root_dir,
    'logger': logger,
    'device': device
    
}
print(params.keys())

syn = AeGAN((static_processor, dynamic_processor), params)

if params.eval_ae:
    logger.info("\n")
    logger.info("evaluate ae!")
    syn.load_ae(params.fix_ae)
    res, h = syn.eval_ae(train_set)
    with open("{}/data".format(root_dir), "wb") as f:
        pickle.dump(res, f)
    with open("{}/hidden".format(root_dir), "wb") as f:
        pickle.dump(h, f)
    exit()
    
if params.fix_ae is not None:
    syn.load_ae(params.fix_ae)
else:
    syn.train_ae(train_set, params.epochs)
    res, h = syn.eval_ae(train_set)
    with open("{}/hidden".format(root_dir), "wb") as f:
        pickle.dump(h, f)

if params.fix_gan is not None:
    syn.load_generator(params.fix_gan)
else:
    syn.train_gan(train_set, params.iterations, params.d_update)

logger.info("\n")
logger.info("Generating data!")
result = syn.synthesize(len(train_set))

with open("{}/data".format(root_dir), "wb") as f:
    pickle.dump(result, f)