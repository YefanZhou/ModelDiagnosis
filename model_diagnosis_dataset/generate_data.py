import os
import numpy as np
import pickle
import argparse
from utils.data import get_metric_file, get_metric_val, convert_data_to_frame, assign_temp_value, assign_width_batch_label, assign_data_batch_label


parser = argparse.ArgumentParser()
parser.add_argument("--model-w-label-noise", action='store_true', default=False)
parser.add_argument("--generate-grid", action='store_true', default=False)
parser.add_argument("--generate-dictionary", action='store_true', default=False)
parser.add_argument("--generate-dataframe", action='store_true', default=False)


args = parser.parse_args()
width_list=[2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128]
bs_list=[16, 24, 32, 44, 64, 92, 128, 180, 256, 364, 512, 724, 1024]
data_list=['10', '09', '08', '07', '06', '05', '04', '03', '02', '01']


if args.model_w_label_noise:
    src_path = "./pretrained_model_collections/noisy_collection"
else:
    src_path = "./pretrained_model_collections/basic_collection"


metric_list = ['test_acc', 'test_loss', 'train_acc',  'train_loss', 
                'hessian_e', 'hessian_t', 'mode_connectivity', 
                'mode_connectivity_peak', 'CKA']
metric_title_list = ['Test accuracy', 'Test Loss', 'Train accuracy', 'Training loss', 
                      'Log Hessian eigenvalue', 'Hessian trace', 'Mode connectivity', 
                      'Mode connectivity peak',  'CKA similarity']

class_label_list = ['bs_failure', 'width_vs_batch', 'data_vs_batch']

if args.model_w_label_noise:
    name_of_grid = "dictionary/noisy_metric_grid_13_13_10_public.npy"
    name_of_ckpt = "dictionary/dict_13_13_10_noise_public.npy"
    name_of_dataframe = "dataframe/dataframe_noise_public.csv"
else:
    name_of_grid = "dictionary/metric_grid_13_13_10_public.npy"
    name_of_ckpt = "dictionary/dict_13_13_10_normal_public.npy"
    name_of_dataframe = "dataframe/dataframe_normal_public.csv"


if args.generate_grid:
    lenx = len(bs_list)
    leny = len(width_list)
    lenz = len(data_list)

    metric_grid_dictionary = {}
    metric_grid_dictionary['temp'] = bs_list
    metric_grid_dictionary['width'] = width_list
    metric_grid_dictionary['data'] = data_list


    for metric, metric_title in zip(metric_list, metric_title_list):
        phase3D = np.zeros((lenx, leny, lenz))
        for i, bs in enumerate(bs_list):
            for j, width in enumerate(width_list):
                for k, data in enumerate(data_list):
                    
                    if args.model_w_label_noise:
                        ckpt_folder = os.path.join(src_path, f'different_knobs_subset_noisy_{data}/bs_{bs}/normal/ResNet18_w{width}/')
                    else:
                        ckpt_folder = os.path.join(src_path, f'different_knobs_subset_{data}/bs_{bs}/normal/ResNet18_w{width}/')
                        
                    metric_file = get_metric_file(ckpt_folder, metric)
                    if metric_file:
                        phase3D[i][j][k] = get_metric_val(metric_file, metric)
                    else:
                        print(f"w={width} bs={bs}  data={data} ckpt_exist False, input np.nan")
                        
        metric_grid_dictionary[metric] = phase3D
        print(f'{metric_title} done')

    metric_grid_dictionary['generalization_gap'] = metric_grid_dictionary['train_acc'] - metric_grid_dictionary['test_acc']

    np.save(f'{name_of_grid}', metric_grid_dictionary)



if args.generate_dictionary:
    metric_grid_dictionary = np.load(f'{name_of_grid}', allow_pickle=True).item()
    operations = \
    {'inc-temp': 0, 
    'dec-temp': 1, 
    'inc-width': 2, 
    'dec-width': 3, 
    'inc-data': 4, 
    'dec-data': 5}

    checkpoints_lst = {}
    for j, width in enumerate(width_list):
        for i, bs in enumerate(bs_list):
            for k, data in enumerate(data_list):
                metric_to_value = {}
                for key in ['test_acc', 'train_acc',
                            'generalization_gap', 'test_loss', 'train_loss', 
                            'CKA', 'hessian_e', 'hessian_t', 
                            'mode_connectivity', 'mode_connectivity_peak']:
                    metric_to_value[key] = metric_grid_dictionary[key][i, j, k]
                    
                hyperparam_sweeping = {}
                for oper in operations:
                    # bs_list=[16, 24, 32, 44, 64, 92, 128, 180, 256, 364, 512, 724, 1024]
                    if oper == 'inc-temp': # inc-temp == dec batch size
                        impro_lst = metric_grid_dictionary['test_acc'][:i+1, j, k]
                    elif oper == 'dec-temp':  # dec-temp == inc batch size
                        impro_lst = metric_grid_dictionary['test_acc'][i:, j, k]
                    # width_list=[2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128]
                    elif oper == 'inc-width':
                        impro_lst = metric_grid_dictionary['test_acc'][i, j:, k]
                    elif oper == 'dec-width':
                        impro_lst = metric_grid_dictionary['test_acc'][i, :j+1, k]  
                    #data_list=['10', '09', '08', '07', '06', '05', '04', '03', '02', '01']
                    elif oper == 'inc-data':
                        impro_lst = metric_grid_dictionary['test_acc'][i, j, :k+1]
                    elif oper == 'dec-data':
                        impro_lst = metric_grid_dictionary['test_acc'][i, j, k:]
                        
                    hyperparam_sweeping[oper] = {'all': impro_lst.tolist()}
                    hyperparam_sweeping[oper]['max'] =  max(hyperparam_sweeping[oper]['all'])  #None if len(hyperparam_sweeping[oper]['all']) == 0 else
                    hyperparam_sweeping[oper]['improvement'] = hyperparam_sweeping[oper]['max'] - metric_to_value['test_acc']  #None if hyperparam_sweeping[oper]['max'] == None else 
                    
                oper_imp_lst = [(oper, -10000) if hyperparam_sweeping[oper]['improvement'] is None else (oper, hyperparam_sweeping[oper]['improvement']) for oper in hyperparam_sweeping]
                hyperparam_sweeping['summary_improvement'] = oper_imp_lst
                hyperparam_sweeping['summary_improvement'] = dict(hyperparam_sweeping['summary_improvement'])
                hyperparam_sweeping['operations'] = operations
                best_label = max(oper_imp_lst, key=lambda x: x[1])
                
                instance = { 'architecture': 'ResNet18', 
                            'dataset':      'cifar10', 
                            'method':        'normal', 
                            'temperature':   {'bs': bs}, 
                            'width':         width, 
                            'noise':          0, 
                            'data_amount':    int(data), 
                            'initialization': None, 
                            'epochs':         None, 
                            'metrics':None,
                            'metrics_dict':metric_to_value, 
                            'bs_range':bs_list, 
                            'width_range':width_list, 
                            'data_amount_range':data_list, 
                            'noise_range':None, 
                            'possible_improve':hyperparam_sweeping,
                            'label':best_label, 
                            }
                
                checkpoints_lst[f'w_{width}_t_{bs}_d_{data}'] = instance
                
    np.save(f'{name_of_ckpt}', checkpoints_lst)


if args.generate_dataframe:
    checkpoints_lst = np.load(f'{name_of_ckpt}', allow_pickle=True).item()
    data_percent_to_number_data = np.load("helper/data_percent_to_number_data.npy", allow_pickle=True).item()
    width_to_parameters_count = np.load("helper/width_to_parameters_count.npy", allow_pickle=True).item()

    labels_dict_frame = convert_data_to_frame(checkpoints_lst, data_percent_to_number_data, width_to_parameters_count)

    for class_label in class_label_list:
        if class_label == 'bs_failure':
            labels_dict_frame[class_label] = labels_dict_frame.apply(assign_temp_value, axis=1)
        if class_label == 'width_vs_batch':
            labels_dict_frame[class_label] = labels_dict_frame.apply(assign_width_batch_label, axis=1)
        if class_label == 'data_vs_batch':
            labels_dict_frame[class_label] = labels_dict_frame.apply(assign_data_batch_label, axis=1)
            
            
    labels_dict_frame.to_csv(f'{name_of_dataframe}', index=False)