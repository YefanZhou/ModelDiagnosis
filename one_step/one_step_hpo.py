import numpy as np
import random
from one_step_hpo_utils import *
import sys
import argparse
import json
from pathlib import Path
sys.path.append('../')
from utils.data import preprocess

parser = argparse.ArgumentParser(description='')
parser.add_argument('--base-path',      type=str,     default='.')
parser.add_argument('--seed-lst',       type=int,   nargs='+', default=[42, 90, 38, 18, 72], help='random seeds')
parser.add_argument('--trainset-samplefactor', type=int,   nargs='+', default=[]) 
parser.add_argument('--feature-list',        type=str,   nargs='+', default=['log_data_amount', 'log_para_amount'], help='list of model features')
parser.add_argument('--save-path',      type=str,   default='', help='path to save the results')
parser.add_argument('--class-label',    type=str,   default='batch_vs_batch', help='diagnosis task label')
parser.add_argument('--train-path',     type=str,   default='../model_diagnosis_dataset/dictionary/dict_13_13_10_normal.npy', help='path to the training set')
parser.add_argument('--test-path',      type=str,   default='../model_diagnosis_dataset/dictionary/dict_13_13_10_noise.npy', help='path to the test set')
parser.add_argument('--method',         type=str,   default='decision_tree', help='method name')
parser.add_argument('--subtree-type',   type=str,   default='MDtree_temp', help='subtree of the MD tree')
parser.add_argument('--sample-type',    type=str,   default='random', help='how to sample the training set')
parser.add_argument('--tree-depth',     type=int,   default=4)
args = parser.parse_args()

train_frame, test_frame, test_dictionary = preprocess('..')

if args.class_label == 'bs_failure':
    train_frame = train_frame[(train_frame['bs_failure'] != 'optimal')]
    test_frame = test_frame[(test_frame['bs_failure'] != 'optimal')]
elif args.class_label == 'width_vs_batch':
    train_frame = train_frame[ (train_frame['width_vs_batch'] != 'optimal') & (train_frame['bs_failure'] != 'optimal') ]
    test_frame = test_frame[ (test_frame['width_vs_batch'] != 'optimal') & (test_frame['bs_failure'] != 'optimal') ]

data_list = list(set(train_frame['data']))
width_list = list(set(train_frame['width']))
data_list.sort()
width_list.sort()
dw_combined_list = [(x, y) for x in data_list for y in width_list]

if args.method in ['random', 'optimal']:
    hpo_result = baseline_diagnoise(args.method, 
                                    args.class_label, 
                                    args.seed_lst, 
                                    train_frame, 
                                    test_frame, 
                                    dw_combined_list, 
                                    test_dictionary)
else:
    hpo_result = diagnoise(args, train_frame, test_frame, dw_combined_list, data_list, width_list, test_dictionary)


params_str = f"{args.method}_label_{args.class_label}.npy"
Path(args.save_path).mkdir(parents=True, exist_ok=True)
np.save(args.save_path + params_str, hpo_result)

print(f"Diagnosis method: {args.method}, Avg Test Accuracy Improvement")
for key in hpo_result:
    print(f"Strategy of tuning step is {key}: mean: {hpo_result[key][0]:.3f} std: {hpo_result[key][1]:.3f}")

