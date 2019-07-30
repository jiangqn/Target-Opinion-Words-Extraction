import yaml
import os

# load config
config = yaml.load(open('config.yml'))

# build paths
base_path = os.path.join('./data/', config['dataset'])
train_src_path = os.path.join(base_path, 'train.tsv')
test_src_path = os.path.join(base_path, 'test.tsv')
train_trg_path = os.path.join(base_path, 'train.npz')
val_trg_path = os.path.join(base_path, 'val.npz')
test_trg_path = os.path.join(base_path, 'test.npz')

# load training and test data
train_data = open(train_src_path, 'r', encoding='utf-8').readlines()
test_data = open(test_src_path, 'r', encoding='utf-8').readlines()

# split training and validation datasets
num = len(train_data)
val_data = train_data[(1 - config['val_rate']) * num: num]
train_data = train_data[0: (1 - config['val_rate']) * num]

