import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import random
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import sys
import os

utils_path = '../utils'
method_path = '../method'
if utils_path not in sys.path:
    sys.path.append(utils_path)
if method_path not in sys.path:
    sys.path.append(method_path)

from baseline_dt import evaluate_basic_decision_tree
from md_tree import evaluate_mdtree



def baseline_diagnoise(method, class_label, seed_lst, train_frame, test_frame, dw_combined_list, test_dictionary):
    """baseline diagnosis methods such as random diagnosis and optimal diagnosis

    Args:
        method (_type_): _description_
        class_label (_type_): _description_
        seed_lst (_type_): _description_
        train_frame (_type_): _description_
        test_frame (_type_): _description_
        dw_combined_list (_type_): _description_
        test_dictionary (_type_): _description_

    Returns:
        _type_: _description_
    """
    total_improve = {'one':[], 'random':[], 'optimal':[]}
    for seed in seed_lst:
        np.random.seed(seed)
        count = 0
        seed_improve = {'one':0, 'random':0, 'optimal':0}
        for i in range(len(test_frame)):
            w, bs, d = test_frame['width'].iloc[i], test_frame['bs'].iloc[i], test_frame['data'].iloc[i]
            if d == 10:
                key = f"w_{w}_t_{bs}_d_{d}"
            else:
                key = f"w_{w}_t_{bs}_d_0{d}"
            
            # random diagnosis
            if method == 'random':
                # determine how to change the hyperparameter
                if class_label == 'bs_failure':
                    policy = np.random.choice(['inc-temp', 'dec-temp'])
                elif class_label == "width_vs_batch":
                    policy = np.random.choice(['inc-width', 'else'])
                    if policy != 'inc-width':
                        policy = np.random.choice(['inc-temp', 'dec-temp'])
                
                curr_acc = test_dictionary[key]['metrics_dict']['test_acc']
                
                if policy == 'inc-temp':
                    if len(test_dictionary[key]['possible_improve'][policy]['all']) == 1:
                        # the current model is at the ends of hyperparameter spectrum
                        one_step_acc = curr_acc
                        random_step_acc = curr_acc
                        optimal_step_acc = curr_acc
                    else:
                        # three methods for choosing the tuning steps
                        one_step_acc = test_dictionary[key]['possible_improve'][policy]['all'][-2]
                        random_step_acc = np.random.choice(test_dictionary[key]['possible_improve'][policy]['all'][:-1])
                        optimal_step_acc = test_dictionary[key]['possible_improve'][policy]['max']
                
                elif policy == 'dec-temp':
                    if len(test_dictionary[key]['possible_improve'][policy]['all']) == 1:
                        one_step_acc = curr_acc
                        random_step_acc = curr_acc
                        optimal_step_acc = curr_acc
                    else:
                        one_step_acc = test_dictionary[key]['possible_improve'][policy]['all'][1]
                        random_step_acc = np.random.choice(test_dictionary[key]['possible_improve'][policy]['all'][1:])
                        optimal_step_acc = test_dictionary[key]['possible_improve'][policy]['max']
                
                elif policy == 'inc-width':
                    if len(test_dictionary[key]['possible_improve'][policy]['all']) == 1:
                        one_step_acc = curr_acc
                        random_step_acc = curr_acc
                        optimal_step_acc = curr_acc
                    else:
                        one_step_acc = test_dictionary[key]['possible_improve'][policy]['all'][1]
                        random_step_acc = np.random.choice(test_dictionary[key]['possible_improve'][policy]['all'][1:])
                        optimal_step_acc = test_dictionary[key]['possible_improve'][policy]['max']
                        
            # optimal diagnosis     
            elif method == 'optimal':
                curr_acc = test_dictionary[key]['metrics_dict']['test_acc']

                if class_label == 'bs_failure':
                    policy_list = ['inc-temp', 'dec-temp']
                elif class_label == 'width_vs_batch':
                    policy_list = ['inc-temp', 'dec-temp', 'inc-width']
                    
                max_improvement = float('-inf')
                optimal_policy = None
                for policy in policy_list:
                    improvement = test_dictionary[key]['possible_improve'][policy]['improvement']
                    if improvement > max_improvement:
                        max_improvement = improvement
                        optimal_policy = policy

                if len(test_dictionary[key]['possible_improve'][optimal_policy]['all']) == 1:
                    one_step_acc = curr_acc
                    random_step_acc = curr_acc
                elif 'inc-temp' in optimal_policy:
                    one_step_acc = max( curr_acc, \
                                        test_dictionary[key]['possible_improve'][optimal_policy]['all'][-2])
                    random_step_acc = max( curr_acc, \
                                        np.random.choice(test_dictionary[key]['possible_improve'][optimal_policy]['all'][:-1]))
                elif 'dec-temp' or 'inc-width' in optimal_policy:
                    one_step_acc = max( curr_acc, \
                                        test_dictionary[key]['possible_improve'][optimal_policy]['all'][1])
                    random_step_acc = max( curr_acc, \
                                        np.random.choice(test_dictionary[key]['possible_improve'][optimal_policy]['all'][1:]))
                    
                optimal_step_acc = max( curr_acc, \
                                            test_dictionary[key]['possible_improve'][optimal_policy]['max'])
            
            # calculate the improvement
            one_step_improve = max(one_step_acc - curr_acc, 0)
            random_step_improve = max(random_step_acc - curr_acc, 0)
            optimal_step_improve = max(optimal_step_acc - curr_acc, 0)

            seed_improve['one'] += one_step_improve
            seed_improve['random'] += random_step_improve
            seed_improve['optimal'] += optimal_step_improve
            count += 1
        
        # average over configurations
        total_improve['one'].append(seed_improve['one'] / count)
        total_improve['optimal'].append(seed_improve['optimal'] / count)
        total_improve['random'].append(seed_improve['random'] / count)

    # average over seeds
    return_result = {}
    return_result['one'] = (np.mean(total_improve['one']), np.std(total_improve['one']))
    return_result['random'] = (np.mean(total_improve['random']), np.std(total_improve['random']))
    return_result['optimal'] = (np.mean(total_improve['optimal']), np.std(total_improve['optimal']))
    
    return return_result



def diagnoise(args, train_frame, test_frame, dw_combined_list, data_list, width_list, test_data):
    """few-shot training based diagnosis methods

    Args:
        args (_type_): _description_
        train_frame (_type_): _description_
        test_frame (_type_): _description_
        dw_combined_list (_type_): _description_
        data_list (_type_): _description_
        width_list (_type_): _description_
        test_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    total_improve = {'one':[], 'random':[], 'optimal':[]}
    for seed in args.seed_lst:
        np.random.seed(seed)

        # baseline methods
        if args.method == "decision_tree":
            _, _, test_pred = evaluate_basic_decision_tree(train_frame=train_frame,
                                         test_frame=test_frame,
                                         dw_combined_list=dw_combined_list,
                                         feature=args.feature_list, 
                                         trainset_samplefactor=args.trainset_samplefactor, 
                                         seed_lst=[seed], 
                                         label=args.class_label,
                                         sample_type=args.sample_type,
                                         data_list=data_list,
                                         width_list=width_list,
                                         tree_depth=args.tree_depth,
                                         dt_seed=42
                                         )
            
            # if class label == 'width_vs_batch', need to make another prediction on Q1 to determine optimizer hyperparameter is large or small
            if args.class_label == "width_vs_batch":
                _, _, batch_test_pred = evaluate_basic_decision_tree(
                                                    train_frame=train_frame, 
                                                    test_frame=test_frame, 
                                                    dw_combined_list=dw_combined_list, 
                                                    feature=args.feature_list, 
                                                    trainset_samplefactor=args.trainset_samplefactor, 
                                                    seed_lst=[seed], 
                                                    label="bs_failure",
                                                    sample_type=args.sample_type, 
                                                    data_list=data_list,
                                                    width_list=width_list,
                                                    tree_depth=args.tree_depth,
                                                    dt_seed=42
                                                    )
        
        # our method MD tree
        elif args.method == "md_tree":    
            _, _, _, test_pred = evaluate_mdtree(
                                        train_frame=train_frame, 
                                        test_frame=test_frame, 
                                        dw_combined_list=dw_combined_list,
                                        feature=args.feature_list, 
                                        trainset_samplefactor=args.trainset_samplefactor, 
                                        seed_lst=[seed], 
                                        label=args.class_label,
                                        sample_type=args.sample_type, 
                                        data_list=data_list,
                                        width_list=width_list,
                                        subtree_type=args.subtree_type
                                        )

            # if class label == 'width_vs_batch', need to make another prediction on Q1 to determine optimizer hyperparameter is large or small
            if args.class_label == "width_vs_batch":
                temp_feature_lst = ['train_error', 'mode_connectivity_peak', 'hessian_t']
                _, _, _, batch_test_pred = evaluate_mdtree(
                                        train_frame=train_frame, 
                                        test_frame=test_frame, 
                                        dw_combined_list=dw_combined_list,
                                        feature=temp_feature_lst, 
                                        trainset_samplefactor=args.trainset_samplefactor, 
                                        seed_lst=[seed], 
                                        label="bs_failure",
                                        sample_type=args.sample_type, 
                                        data_list=data_list,
                                        width_list=width_list,
                                        subtree_type="MDtree_temp"
                                        )
        results = test_pred
        count = 0
        seed_improve = {'one':0, 'random':0, 'optimal':0}
        for i in range(len(test_frame)):
            w, bs, d = test_frame['width'].iloc[i], test_frame['bs'].iloc[i], test_frame['data'].iloc[i]
            
            # convert the prediction to actual tuning policy, temperature is large -> decrease temperature, which is equivalent to batch size (optimizer hyper.) is small -> increasae batch size
            if d == 10:
                key = f"w_{w}_t_{bs}_d_{d}"
            else:
                key = f"w_{w}_t_{bs}_d_0{d}"
            if args.class_label == 'bs_failure':
                if results[i] == 'large':
                    policy = 'dec-temp'
                else:
                    policy = 'inc-temp'
            elif args.class_label == "width_vs_batch":
                if results[i] == 'width':
                    policy = 'inc-width'
                elif batch_test_pred[i] == 'large':
                    policy = 'dec-temp'
                else:
                    policy = 'inc-temp'
            
            # determine the performance after taking three types of tuning steps 
            curr_acc = test_data[key]['metrics_dict']['test_acc']
            if policy == 'inc-temp':
                if len(test_data[key]['possible_improve'][policy]['all']) == 1:
                    one_step_acc = curr_acc
                    random_step_acc = curr_acc
                    optimal_step_acc = curr_acc
                else:
                    one_step_acc = test_data[key]['possible_improve'][policy]['all'][-2]
                    random_step_acc = np.random.choice(test_data[key]['possible_improve'][policy]['all'][:-1])
                    optimal_step_acc = test_data[key]['possible_improve'][policy]['max']
            
            elif policy == 'dec-temp':
                if len(test_data[key]['possible_improve'][policy]['all']) == 1:
                    one_step_acc = curr_acc
                    random_step_acc = curr_acc
                    optimal_step_acc = curr_acc
                else:
                    one_step_acc = test_data[key]['possible_improve'][policy]['all'][1]
                    random_step_acc = np.random.choice(test_data[key]['possible_improve'][policy]['all'][1:])
                    optimal_step_acc = test_data[key]['possible_improve'][policy]['max']
            
            elif policy == 'inc-width':
                if len(test_data[key]['possible_improve'][policy]['all']) == 1:
                    one_step_acc = curr_acc
                    random_step_acc = curr_acc
                    optimal_step_acc = curr_acc
                else:
                    one_step_acc = test_data[key]['possible_improve'][policy]['all'][1]
                    random_step_acc = np.random.choice(test_data[key]['possible_improve'][policy]['all'][1:])
                    optimal_step_acc = test_data[key]['possible_improve'][policy]['max']

            # compute the test accuracy improvement
            one_step_improve = max(one_step_acc - curr_acc, 0)
            random_step_improve = max(random_step_acc - curr_acc, 0)
            optimal_step_improve = max(optimal_step_acc - curr_acc, 0)

            seed_improve['one'] += one_step_improve
            seed_improve['random'] += random_step_improve
            seed_improve['optimal'] += optimal_step_improve
            count += 1
        
        # average over samples
        total_improve['one'].append(seed_improve['one'] / count)
        total_improve['optimal'].append(seed_improve['optimal'] / count)
        total_improve['random'].append(seed_improve['random'] / count)

    # average over seeds
    return_result = {}
    return_result['one'] = (np.mean(total_improve['one']), np.std(total_improve['one']))
    return_result['random'] = (np.mean(total_improve['random']), np.std(total_improve['random']))
    return_result['optimal'] = (np.mean(total_improve['optimal']), np.std(total_improve['optimal']))
    return return_result

