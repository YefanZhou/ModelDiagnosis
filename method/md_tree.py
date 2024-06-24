import sys
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score
from collections import defaultdict
import copy 
sys.path.append('../method/')
sys.path.append('method/')
from method_utils import sample_train_set


## Four objective functions
############################################################
def objective_temp(threshold, feature_index, thresholds, X, y):
    thresholds[feature_index] = threshold
    predictions = predictor_temp(X, thresholds)
    return -accuracy_score(y, predictions)

def objective_width(threshold, feature_index, thresholds, X, y):
    thresholds[feature_index] = threshold
    predictions = predictor_width_temp(X, thresholds)
    return -accuracy_score(y, predictions)

def objective_data(threshold, feature_index, thresholds, X, y):
    thresholds[feature_index] = threshold
    predictions = predictor_data_temp(X, thresholds)
    return -accuracy_score(y, predictions)

def objective_temp_cka(threshold, feature_index, thresholds, X, y):
    thresholds[feature_index] = threshold
    predictions = predictor_temp_w_cka(X, thresholds)
    return -accuracy_score(y, predictions)
############################################################

## Four predictor function
############################################################
def predictor_temp(X, thresholds):
    predictions = []
    for i in range(X.shape[0]):
        if X[i, 0] <= thresholds[0]:
            predictions.append('small')
        else:
            # poorly-connected,  smaller than mode connectivity 
            if X[i, 1] <= thresholds[1]:
                #  less sharp temperature is too large                     cka 
                if X[i, 2] <= thresholds[2]: #and X[i, 2] > thresholds[2]:
                    predictions.append('large')
                else:
                #  more sharp temperature is too small
                    predictions.append('small')
            # well-connected,  larger than mode connectivity 
            else:
                predictions.append('large')

    predictions = np.array(predictions)
    return predictions

def predictor_temp_w_cka(X, thresholds):
    predictions = []
    for i in range(X.shape[0]):
        if X[i, 0] <= thresholds[0]:
            predictions.append('small')
        else:
            # poorly-connected,  smaller than mode connectivity 
            if X[i, 1] <= thresholds[1]:
                #  less sharp  temperature is too large                     cka 
                if X[i, 2] <= thresholds[2]: #and X[i, 2] > thresholds[2]:
                    predictions.append('small')
                else:
                #  more sharp  temperature is too small
                    predictions.append('large')
            # well-connected,  larger than mode connectivity 
            else:
                predictions.append('large')

    predictions = np.array(predictions)
    return predictions

def predictor_width_temp(X, thresholds):
    predictions = []
    for i in range(X.shape[0]):
        if X[i, 0] <= thresholds[0]: # interpolating
            if X[i, 1] <= thresholds[1]:     # poorly-connected
                predictions.append('width')
            else:                            # connected
                predictions.append('batch')
        else:
            if X[i, 2] <= thresholds[2]:
                predictions.append('width')  # poorly-connected
            else:
                predictions.append('batch')  # connected
                
    predictions = np.array(predictions)
    return predictions

def predictor_data_temp(X, thresholds):
    predictions = []
    for i in range(X.shape[0]):
        if X[i, 0] <= thresholds[0]:  #interpolating
            if X[i, 1] <= thresholds[1]: # poorly-connected
                #predictions.append('data')
                if X[i, 4] <= thresholds[4]:
                    predictions.append('data')
                else:
                    predictions.append('batch')
            else:                           # well-connected
                if X[i, 2] <= thresholds[2]:   # small hessian 
                    predictions.append('data')
                else:
                    predictions.append('batch')
        else:
            if X[i, 3] <= thresholds[3]:     # poorly-connected
                predictions.append('batch')
            else:                            # well-connected
                if X[i, 5] <= thresholds[5]:   # small hessian 
                    predictions.append('data')
                else:
                    predictions.append('batch')
                # predictions.append('data')
    predictions = np.array(predictions)
    return predictions
############################################################


MDTree_Hyper={
    # for thresholds, bounds, from left to right:  training error, connectivity, hessian_t
    'MDtree_temp':          {'objective': objective_temp,  'predictor': predictor_temp,   
                             'thresholds': [0.5, -10,  7], 'bounds': [[0, 1], [-30, 0], [4, 9]]},
    
    # for thresholds, bounds, from left to right:  training error, connectivity, connectivity
    'MDtree_width':         {'objective': objective_width,  'predictor': predictor_width_temp,
                             'thresholds': [0.5, -10, -10], 'bounds': [[0, 1], [-30, 0], [-30, 0]]},
    
    # for thresholds, bounds, from left to right:  training error, connectivity, hessian_t, connectivity, hessian_t, hessian_t
    'MDtree_data':          {'objective': objective_data,  'predictor':predictor_data_temp,
                             'thresholds': [0.5, -10,  7, -10, 7, 7], 'bounds': [[0, 1], [-30, 0], [4, 9], [-30, 0], [4, 9], [4, 9]] },
    
    'MDtree_temp_ood_data': {'objective': objective_temp,  'predictor': predictor_temp,
                             'thresholds': [0.5, -10,  5], 'bounds': [[0, 1], [-30, 0], [4, 9]]}, 
    
    'MDtree_temp_cka':      {'objective': objective_temp_cka, 'predictor': predictor_temp_w_cka,
                             'thresholds': [0.5, -10, 0.5], 'bounds': [[0, 1], [-30, 0], [0.2, 0.8]]}
}


def evaluate_mdtree(train_frame, 
                           test_frame, 
                           dw_combined_list, 
                           feature, 
                           trainset_samplefactor, 
                           seed_lst, 
                           label,
                           sample_type='random',
                           data_list=list(range(1, 11, 1)),
                           width_list=[2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128],
                           subtree_type = 'MDtree_temp'
                           ):
    """MD tree diagnosis method
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
        subtree_type (str, optional): subtree used for different diagnosis tasks, Defaults to 'MDtree_temp'.
        
    Returns:
        vary_samples_test_lst:  nested list of diagnosis accuracy
        vary_samples_count_lst:  nested list of number of trained models used
        vary_samples_md_threshold_lst: nested list of learned threshold on loss landscape metrics
        test_pred: predictions
    """

    vary_samples_test_lst = []
    vary_samples_count_lst = []
    vary_samples_md_threshold_lst = []
    for trainset_s in trainset_samplefactor:
        vary_seed_test_lst = []
        vary_seed_count_lst = []
        vary_seed_md_threshold_lst = []
        for seed in seed_lst:
            
            filtered_df = sample_train_set(train_frame, trainset_s, dw_combined_list, data_list, width_list, seed, sample_type)
            base_hyper_train = filtered_df[feature].to_numpy()
            labels_train = filtered_df[label].to_numpy()

            base_hyper_test = test_frame[feature].to_numpy()
            labels_test = test_frame[[label]].to_numpy().ravel()
            
            
            objective, predictor, thresholds, bounds = MDTree_Hyper[subtree_type]['objective'],  MDTree_Hyper[subtree_type]['predictor'], \
                                                MDTree_Hyper[subtree_type]['thresholds'], MDTree_Hyper[subtree_type]['bounds']
            
            # deep copy to make sure to use a new initialization 
            
            thresholds = copy.copy(thresholds)
            
            for feature_index in range(len(thresholds)):
                result = minimize_scalar(objective, 
                                args=(feature_index, thresholds, base_hyper_train, labels_train), 
                                method='bounded', 
                                bounds=( bounds[feature_index][0], bounds[feature_index][1]  )  )  
                thresholds[feature_index] = result.x
            
            train_pred = predictor(base_hyper_train, thresholds)
            test_pred = predictor(base_hyper_test, thresholds)
                 
            train_acc = sum(train_pred == labels_train) / len(labels_train)
            test_acc = sum(test_pred == labels_test) / len(labels_test)
            
            vary_seed_test_lst.append(test_acc)
            vary_seed_count_lst.append(len(filtered_df))
            vary_seed_md_threshold_lst.append(copy.copy(thresholds)) 

        vary_samples_test_lst.append(vary_seed_test_lst)
        vary_samples_count_lst.append(vary_seed_count_lst)
        vary_samples_md_threshold_lst.append(vary_seed_md_threshold_lst)

    return vary_samples_test_lst, vary_samples_count_lst, vary_samples_md_threshold_lst, test_pred