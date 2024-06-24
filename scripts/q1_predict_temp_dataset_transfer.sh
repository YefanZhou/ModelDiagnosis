# Predict temperature random sample
trainset_factors=($(seq 1 24))
sample_type='random'
label='bs_failure'
subtree_type='MDtree_temp'

##  baseline: Hyperparameter 
python evaluate_diagnosis_acc.py \
    --feature-list 'log_data_amount' 'log_para_amount' \
    --save-path results/${label}/dataset_transfer/hyperparameter_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}

##  baseline: Validation metrics 
python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'train_loss' 'test_loss' 'test_error' \
    --save-path results/${label}/dataset_transfer/validation_metric_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}

##  our method: MD tree
python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'mode_connectivity_peak' 'hessian_t' \
    --save-path results/${label}/dataset_transfer/md_tree_${sample_type}.npy \
    --method 'md_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}

##  baseline: losslandscape metrics + decision tree
python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'mode_connectivity_peak' 'CKA' \
    --save-path results/${label}/dataset_transfer/md_tree_cka_${sample_type}.npy \
    --method 'md_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]} \
    --subtree-type MDtree_temp_cka

##  baseline: Validation metrics + Hyperparameter
python evaluate_diagnosis_acc.py \
    --feature-list 'log_data_amount' 'log_para_amount' 'train_error' 'train_loss' 'test_loss' 'test_error' \
    --save-path results/${label}/dataset_transfer/hyperparameter_validation_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}

##  our method: MD tree (using similarity instead of sharpness)
python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'mode_connectivity_peak' 'CKA' 'hessian_t' \
    --save-path results/${label}/dataset_transfer/loss_landscape_metric_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}


