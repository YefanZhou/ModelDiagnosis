import numpy as np



def sample_train_set(train_frame, 
                     trainset_s, 
                     dw_combined_list, 
                     data_list, 
                     width_list, 
                     seed, 
                     sample_type):
    
    np.random.seed(seed)
    
    if sample_type == 'random':
        dw_index_lst = np.random.choice(list(range(len(dw_combined_list))), size=trainset_s, replace=False)
        data_width_lst = [dw_combined_list[index] for index in dw_index_lst]
        filtered_df = \
        train_frame[train_frame.apply(lambda row: (row['data'], row['width']) in data_width_lst, axis=1)]

    elif sample_type == 'random_data':
        list_data_percent = np.random.choice(data_list, size=trainset_s, replace=False)
        filtered_df = train_frame[train_frame['data'].isin(list_data_percent)]
        
    elif sample_type == 'random_width':
        list_width_percent = np.random.choice(width_list, size=trainset_s, replace=False)
        filtered_df = train_frame[train_frame['width'].isin(list_width_percent)]

    elif sample_type == 'data_limit':
        filtered_df = train_frame[train_frame['data'] <= trainset_s]

    elif sample_type == 'width_limit':
        filtered_df = train_frame[train_frame['width'] <= trainset_s]
        
    return filtered_df