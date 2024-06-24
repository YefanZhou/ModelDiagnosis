sample_type='random_data'
trainset_factors=($(seq 1 10))
label='width_vs_batch'
subtree_type='MDtree_width'

python evaluate_diagnosis_acc.py \
    --feature-list 'log_data_amount' \
    --save-path results/${label}/dataset_transfer/hyperparameter_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}


python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'train_loss' 'test_loss' 'test_error' \
    --save-path results/${label}/dataset_transfer/validation_metric_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}


python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'mode_connectivity_peak' 'mode_connectivity_peak' \
    --save-path results/${label}/dataset_transfer/md_tree_${sample_type}.npy \
    --method 'md_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}


python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'mode_connectivity_peak' 'CKA' 'hessian_t' \
    --save-path results/${label}/dataset_transfer/loss_landscape_metric_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}


sample_type='data_limit'
trainset_factors=($(seq 1 10))
label='width_vs_batch'
subtree_type='MDtree_width'


python evaluate_diagnosis_acc.py \
    --feature-list 'log_data_amount' \
    --save-path results/${label}/scale_transfer/hyperparameter_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}



python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'train_loss' 'test_loss' 'test_error' \
    --save-path results/${label}/scale_transfer/validation_metric_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}


python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'mode_connectivity_peak' 'mode_connectivity_peak' \
    --save-path results/${label}/scale_transfer/md_tree_${sample_type}.npy \
    --method 'md_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]}

