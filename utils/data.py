import os
import pickle
import numpy as np
from collections import defaultdict
import pandas as pd

def preprocess(base_path = '.'):

    path_to_clean = f'{base_path}/model_diagnosis_dataset/dictionary/dict_13_13_10_normal_public.npy'
    path_to_noise = f'{base_path}/model_diagnosis_dataset/dictionary/dict_13_13_10_noise_public.npy'
    
    data_percent_to_number_data = np.load(f"{base_path}/model_diagnosis_dataset/helper/data_percent_to_number_data.npy", allow_pickle=True).item()
    width_to_parameters_count = np.load(f"{base_path}/model_diagnosis_dataset/helper/width_to_parameters_count.npy", allow_pickle=True).item()

    dict_normal = np.load(path_to_clean, allow_pickle=True).item()
    dict_noise  =  np.load(path_to_noise, allow_pickle=True).item()

    labels_dict_frame = convert_data_to_frame(dict_normal, data_percent_to_number_data, width_to_parameters_count)
    labels_dict_frame_noise = convert_data_to_frame(dict_noise, data_percent_to_number_data, width_to_parameters_count)

    class_label = 'bs_failure'
    labels_dict_frame[class_label] = labels_dict_frame.apply(assign_temp_value, axis=1)
    labels_dict_frame_noise[class_label] = labels_dict_frame_noise.apply(assign_temp_value, axis=1)
    
    class_label = 'width_vs_batch'
    labels_dict_frame[class_label] = labels_dict_frame.apply(assign_width_batch_label, axis=1)
    labels_dict_frame_noise[class_label] = labels_dict_frame_noise.apply(assign_width_batch_label, axis=1)
    
    class_label = 'data_vs_batch'
    labels_dict_frame[class_label] = labels_dict_frame.apply(assign_data_batch_label, axis=1)
    labels_dict_frame_noise[class_label] = labels_dict_frame_noise.apply(assign_data_batch_label, axis=1)

    return labels_dict_frame, labels_dict_frame_noise, dict_noise

    

def get_metric_file(ckpt_folder, metric):
    if metric == 'test_acc':
        file_name = os.path.join(ckpt_folder, 'metrics/', 'loss_acc_test.pkl')
    elif metric == 'test_loss':
        file_name = os.path.join(ckpt_folder, 'metrics/', 'loss_acc_test.pkl')
    elif metric == 'train_acc':
        file_name = os.path.join(ckpt_folder, 'metrics/', 'loss_acc.pkl')
    elif metric == 'train_loss':
        file_name = os.path.join(ckpt_folder, 'metrics/', 'loss_acc.pkl')
    elif 'hessian' in metric:
        file_name = os.path.join(ckpt_folder, 'metrics/', 'hessian.pkl')
    elif 'mode_connectivity' in metric:
        file_name = os.path.join(ckpt_folder, 'metrics/', 'curve_test.npz')
    elif metric == 'L2':
        file_name = os.path.join(ckpt_folder, 'metrics/', 'model_dist.pkl')
    elif metric == 'CKA':
        file_name = os.path.join(ckpt_folder, 'metrics/', 'CKA_mixup_alpha_16.0.pkl')
    elif metric == 'CKA_test':
        file_name = os.path.join(ckpt_folder, 'metrics/', 'CKA_test.pkl')
    elif metric == 'agreement':
        file_name = os.path.join(ckpt_folder, 'metrics/', 'agreement_mixup_alpha_16.0.pkl')
    elif metric == 'agreement_test':
        file_name = os.path.join(ckpt_folder, 'metrics/', 'agreement_test.pkl')
        
    if os.path.exists(file_name):
        return file_name
    else:
        return False






def get_metric_val(metric_file, metric):
    if metric == 'test_acc':
        results = pickle.load(open(metric_file, 'rb'))
        return np.mean([results[n]['accuracy'] for n in range(5)])

    if metric == 'train_acc':
        results = pickle.load(open(metric_file, 'rb'))
        return np.mean([results[n]['accuracy'] for n in range(5)])
    
    if metric == 'test_loss':
        results = pickle.load(open(metric_file, 'rb'))
        return np.mean([results[n]['loss'] for n in range(5)])
    
    if metric == 'train_loss':
        results = pickle.load(open(metric_file, 'rb'))
        return np.mean([results[n]['loss'] for n in range(5)])
    
    elif 'CKA' in metric:
        results = pickle.load(open(metric_file, "rb"))
        CKA_all = []
        is_nan = False
        for exp_ind1 in range(5):
            for exp_ind2 in range(5):
                if exp_ind1 != exp_ind2:
                    value = results['representation_similarity'][exp_ind1][exp_ind2][-1]
                    if np.isnan(value):
                        print(metric_file, metric)
                        is_nan = True
                    else:
                        CKA_all.append(value)
        if is_nan:
            print(results['representation_similarity'])
        return np.mean(CKA_all)

    elif 'agreement' in metric:
        results = pickle.load(open(metric_file, "rb"))
        agreement_all = []
        is_nan = False
        for exp_ind1 in range(5):
            for exp_ind2 in range(5):
                if exp_ind1 != exp_ind2:
                    value = results['classification_similarity'][exp_ind1][exp_ind2][0]
                    if np.isnan(value):
                        print(metric_file, metric)
                        is_nan = True
                    else:
                        agreement_all.append(value)
        if is_nan:
            print(results['classification_similarity'])
        return np.mean(agreement_all)
    
    elif 'hessian' in metric:
        results = pickle.load(open(metric_file, "rb"))
        if '_e' in metric:
            return np.log(np.mean([results[n]['top_eigenvalue'][0] for n in range(5)]))
        elif '_t' in metric:
            return np.log(np.mean([results[n]['trace'] for n in range(5)]))
        
    elif metric == 'mode_connectivity':
        result = np.load(metric_file)['tr_err']
        u = np.argmax(np.abs(result - (result[0] + result[4])/2))
        return (result[0] + result[4])/2 - result[u]
    
    elif metric == 'mode_connectivity_remove_end':
        result = np.load(metric_file)['tr_err']
        result = result[1:-1]
        u = np.argmax(np.abs(result - (result[0] + result[-1])/2))
        return (result[0] + result[-1])/2 - result[u]
    
    elif metric == 'mode_connectivity_peak':
        result = np.load(metric_file)['tr_err']
        return -1 * np.max(result[1:-1])
    
    elif metric == 'L2':
        results = pickle.load(open(metric_file, "rb"))
        dist_all = []
        for exp_ind1 in range(5):
            for exp_ind2 in range(5):
                if exp_ind1 != exp_ind2:
                    dist_all.append(results['model_distance'][exp_ind1][exp_ind2]['dist'])
        return np.mean(dist_all)
    
    
    
    
def convert_data_to_frame(input_dict, data_dict, para_dict, error_factor=1.0, has_data=True):
    labels_dict = defaultdict(list)
    
    for key in input_dict:
        width, bs, data = int(key.split('_')[1]), int(key.split('_')[3]), int(key.split('_')[5])
        labels_dict[f'data_amount'].append(data_dict[data])
        labels_dict[f'para_amount'].append(para_dict[width])
        labels_dict[f'log_data_amount'].append(np.log(data_dict[data]))
        labels_dict[f'log_para_amount'].append(np.log(para_dict[width]))
        labels_dict[f'log_bs'].append(np.log(bs))

        labels_dict[f'width'].append(width)
        labels_dict[f'data'].append(data)
        labels_dict[f'bs'].append(bs)

        # loss metrics and performance metrics
        for metric in input_dict[key]['metrics_dict']:
            if metric == 'loss':
                labels_dict['train_loss'].append(input_dict[key]['metrics_dict'][metric])
            else:
                labels_dict[metric].append(input_dict[key]['metrics_dict'][metric])

        # train/test error
        test_error =  100 - input_dict[key]['metrics_dict']['test_acc']
        train_error = 100 - input_dict[key]['metrics_dict']['train_acc']
        labels_dict['test_error'].append(error_factor * test_error)
        labels_dict['train_error'].append(error_factor * train_error)

        labels_dict['rfi_width'].append(input_dict[key]['possible_improve']['inc-width']['improvement'])
        if has_data:
            labels_dict['rfi_data'].append(input_dict[key]['possible_improve']['inc-data']['improvement'])
        labels_dict['rfi_inc_temp'].append(input_dict[key]['possible_improve']['inc-temp']['improvement'])
        labels_dict['rfi_dec_temp'].append(input_dict[key]['possible_improve']['dec-temp']['improvement'])


    labels_dict_frame = pd.DataFrame(labels_dict)
    return labels_dict_frame


def assign_temp_value(row):
    if row['rfi_inc_temp'] > row['rfi_dec_temp']:
        return 'small'
    elif row['rfi_inc_temp'] < row['rfi_dec_temp']:
        return 'large'
    else:
        return 'optimal'
    
    
def assign_width_batch_label(row):
    if row['rfi_width'] > max(row['rfi_inc_temp'], row['rfi_dec_temp']):
        return 'width'
    elif row['rfi_width'] < max(row['rfi_inc_temp'], row['rfi_dec_temp']):
        return 'batch'
    else:
        return 'optimal'
    
    
def assign_data_batch_label(row):
    if row['rfi_data'] > max(row['rfi_inc_temp'], row['rfi_dec_temp']):
        return 'data'
    elif row['rfi_data'] < max(row['rfi_inc_temp'], row['rfi_dec_temp']):
        return 'batch'
    else:
        return 'optimal'