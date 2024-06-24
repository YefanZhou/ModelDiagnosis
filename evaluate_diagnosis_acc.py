from utils.data import preprocess
from method.baseline_dt import evaluate_basic_decision_tree
from method.md_tree import evaluate_mdtree
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description='')
parser.add_argument('--feature-list', nargs='+', type=str, default=['log_data_amount', 'log_para_amount'], help='list of model features')
parser.add_argument('--seed-list', nargs='+', type=int, default=[42, 90, 38, 18, 72], help='random seeds')
parser.add_argument('--save-path', type=str, default='results/dataset_transfer/hyper_tree_random.npy', help='path to save the results')
parser.add_argument('--method', type=str, default='md_tree', help='method name')
parser.add_argument('--label', type=str, default='bs_failure', help='diagnosis task label')
parser.add_argument('--subtree-type', type=str, default='MDtree_temp', help='subtree of the MD tree')
parser.add_argument('--trainset-sampletype', type=str, default='random', help='how to sample the training set')
parser.add_argument('--trainset-samplefactor', nargs='+', type=int, default=list(range(1, 25, 1)))
parser.add_argument('--model-min-width', type=int, default=None, help='minimum width of the models studied')
args = parser.parse_args()

# convert the dictionary to the dataframe
labels_dict_frame, labels_dict_frame_noise, _ = preprocess(base_path = '.')
labels_dict_frame_run = labels_dict_frame[(labels_dict_frame[args.label] != 'optimal')]
labels_dict_frame_noise_run = labels_dict_frame_noise[(labels_dict_frame_noise[args.label] != 'optimal') ]

if args.model_min_width is not None:
    labels_dict_frame_run = labels_dict_frame_run[labels_dict_frame_run['width'] >= args.model_min_width]
    labels_dict_frame_noise_run = labels_dict_frame_noise_run[labels_dict_frame_noise_run['width'] >= args.model_min_width]


data_list = list(set(labels_dict_frame_run['data']))
width_list = list(set(labels_dict_frame_run['width']))
data_list.sort()
width_list.sort()

data_width_combined_list = [(x, y) for x in data_list for y in width_list]

if args.method == 'base_decision_tree':
    test_acc_lst, model_count_lst, _ = evaluate_basic_decision_tree(
                                                    train_frame=labels_dict_frame_run, 
                                                    test_frame=labels_dict_frame_noise_run, 
                                                    dw_combined_list=data_width_combined_list, 
                                                    feature=args.feature_list, 
                                                    trainset_samplefactor=args.trainset_samplefactor, 
                                                    seed_lst=args.seed_list, 
                                                    label=args.label,
                                                    sample_type=args.trainset_sampletype,
                                                    data_list=data_list,
                                                    width_list=width_list,
                                                    tree_depth=4
                                                    )
    
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.save_path, 
            {'test_acc_lst': test_acc_lst, 
            'model_count_lst': model_count_lst} 
            )
    
elif args.method == 'md_tree':
    test_acc_lst, model_count_lst, md_threshold, _ = evaluate_mdtree(
                                                    train_frame=labels_dict_frame_run, 
                                                    test_frame=labels_dict_frame_noise_run, 
                                                    dw_combined_list=data_width_combined_list,
                                                    feature=args.feature_list,
                                                    trainset_samplefactor=args.trainset_samplefactor, 
                                                    seed_lst=args.seed_list, 
                                                    label=args.label,
                                                    sample_type=args.trainset_sampletype, 
                                                    data_list=data_list,
                                                    width_list=width_list,
                                                    subtree_type=args.subtree_type
                                                    )

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.save_path, 
            {'test_acc_lst': test_acc_lst, 
            'model_count_lst': model_count_lst,
            'md_threshold': md_threshold} 
            )