from collections import defaultdict
import numpy as np
import pandas as pd



def assign_temp_value(row):
    if row['rfi_inc_temp'] > row['rfi_dec_temp']:
        return 'small'
    elif row['rfi_inc_temp'] < row['rfi_dec_temp']:
        return 'large'
    else:
        return 'optimal'

# Define the function to apply to each row
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
    
def assign_data_width_label(row):
    if row['rfi_data'] > row['rfi_width']:
        return 'data'
    elif row['rfi_data'] < row['rfi_width']:
        return 'width'
    else:
        return 'optimal'


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
                labels_dict[metric].append( input_dict[key]['metrics_dict'][metric])

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