## Predict temperature w/ data limit
sample_type='data_limit'
trainset_factors=($(seq 1 10))
label='bs_failure'
subtree_type='MDtree_temp'

python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'train_loss' 'test_loss' 'test_error' \
    --save-path results/${label}/scale_transfer/validation_metric_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}

python evaluate_diagnosis_acc.py \
    --feature-list 'log_data_amount' 'log_para_amount' \
    --save-path results/${label}/scale_transfer/hyperparameter_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}

python evaluate_diagnosis_acc.py \
    --feature-list 'log_data_amount' 'log_para_amount' 'train_error' 'train_loss' 'test_loss' 'test_error' \
    --save-path results/${label}/scale_transfer/hyperparameter_validation_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}

python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'mode_connectivity_peak' 'hessian_t' \
    --save-path results/${label}/scale_transfer/md_tree_${sample_type}.npy \
    --method 'md_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}

python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'mode_connectivity_peak' 'CKA' \
    --save-path results/${label}/scale_transfer/md_tree_cka_${sample_type}.npy \
    --method 'md_tree' \
    --label ${label} \
    --subtree-type 'MDtree_temp_cka' \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}



## Predict temperature w/ width limit
sample_type='width_limit'
trainset_factors=(2 3 4 6 8 11 16 23 32 45 64 91 128)
label='bs_failure'
subtree_type='MDtree_temp'

python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'train_loss' 'test_loss' 'test_error' \
    --save-path results/${label}/scale_transfer/validation_metric_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}

python evaluate_diagnosis_acc.py \
    --feature-list 'log_data_amount' 'log_para_amount' \
    --save-path results/${label}/scale_transfer/hyperparameter_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}

python evaluate_diagnosis_acc.py \
    --feature-list 'log_data_amount' 'log_para_amount' 'train_error' 'train_loss' 'test_loss' 'test_error' \
    --save-path results/${label}/scale_transfer/hyperparameter_validation_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}

python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'mode_connectivity_peak' 'hessian_t' 'CKA' \
    --save-path results/${label}/scale_transfer/loss_landscape_metric_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}

python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'mode_connectivity_peak' 'hessian_t' \
    --save-path results/${label}/scale_transfer/md_tree_${sample_type}.npy \
    --method 'md_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}

python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'mode_connectivity_peak' 'CKA' \
    --save-path results/${label}/scale_transfer/md_tree_cka_${sample_type}.npy \
    --method 'md_tree' \
    --label ${label} \
    --subtree-type 'MDtree_temp_cka' \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}