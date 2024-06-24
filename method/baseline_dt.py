import os
import sys
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
sys.path.append('../method')
sys.path.append('method')
from method_utils import sample_train_set



def evaluate_basic_decision_tree(
                           train_frame, 
                           test_frame, 
                           dw_combined_list, 
                           feature, 
                           trainset_samplefactor, 
                           seed_lst, 
                           label,
                           sample_type='random',
                           data_list=None,
                           width_list=None,
                           tree_depth=4,
                           dt_seed=None,
                           ):
    """baseline diagnosis methods using unfixed decision tree

    Args:
        train_frame (Pandas.DataFrame): training set
        test_frame (Pandas.DataFrame): test set 
        dw_combined_list (a list of tuple): all combinations of data and model width
        feature (list): model features
        trainset_samplefactor (list): factors used to sample the training set
        seed_lst (list): random seeds
        label (str): diagnosis label
        sample_type (str, optional): types of sampling method to enable different transfer scenarios. Defaults to 'random'.
        data_list (list, optional): data percentage. Defaults to None.
        width_list (list, optional): model width scaling factors. Defaults to None.
        tree_depth (int, optional): depth of decision tree. Defaults to 4.
        dt_seed (int, optional): the seed. Defaults to None.

    Returns:
        vary_samples_test_lst: nested list of diagnosis accuracy
        vary_samples_count_lst: nested list of number of trained models used
        test_pred: predictions
    """

    vary_samples_test_lst = []
    vary_samples_count_lst = []
    # iterate through different sampling threshold on the training set
    for trainset_s in trainset_samplefactor:
        vary_seed_test_lst = []
        vary_seed_count_lst = []
        # iterate through random seeds
        for seed in seed_lst:
            
            # sample the training set
            filtered_df = sample_train_set(train_frame, trainset_s, dw_combined_list, data_list, width_list, seed, sample_type)

            base_hyper_train = filtered_df[feature].to_numpy()
            labels_train = filtered_df[label].to_numpy()

            base_hyper_test = test_frame[feature].to_numpy()
            labels_test = test_frame[[label]].to_numpy().ravel()

            if dt_seed is None:
                dt_seed = seed
            
            # model training
            clf = DecisionTreeClassifier(criterion='gini', max_depth=tree_depth, random_state=dt_seed).fit(base_hyper_train, labels_train)

            # model prediction and accuracy evaluation
            train_pred = clf.predict(base_hyper_train)
            test_pred = clf.predict(base_hyper_test)
            train_acc = sum(train_pred == labels_train) / len(labels_train)
            test_acc = sum(test_pred == labels_test) / len(labels_test)
            
            vary_seed_test_lst.append(test_acc)
            vary_seed_count_lst.append(len(filtered_df))

        vary_samples_test_lst.append(vary_seed_test_lst)
        vary_samples_count_lst.append(vary_seed_count_lst)
        
    vary_samples_test_lst = np.array(vary_samples_test_lst)
    vary_samples_count_lst = np.array(vary_samples_count_lst)

    return vary_samples_test_lst, vary_samples_count_lst, test_pred