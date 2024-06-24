sample_type='random_width'
trainset_factors=($(seq 1 5))
label='data_vs_batch'
subtree_type='MDtree_data'

python evaluate_diagnosis_acc.py \
    --feature-list 'log_para_amount' \
    --save-path results/${label}/dataset_transfer/hyperparameter_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]} \
    --model-min-width 32

python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'train_loss' 'test_loss' 'test_error' \
    --save-path results/${label}/dataset_transfer/validation_metric_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]} \
    --model-min-width 32

python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'mode_connectivity_peak' 'hessian_t' 'mode_connectivity_peak' 'hessian_t' 'hessian_t' \
    --save-path results/${label}/dataset_transfer/md_tree_${sample_type}.npy \
    --method 'md_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]} \
    --model-min-width 32


sample_type='width_limit'
trainset_factors=(32 45 64 91 128)
label='data_vs_batch'
subtree_type='MDtree_data'


python evaluate_diagnosis_acc.py \
    --feature-list 'log_para_amount' \
    --save-path results/${label}/scale_transfer/hyperparameter_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]} \
    --model-min-width 32


python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'train_loss' 'test_loss' 'test_error' \
    --save-path results/${label}/scale_transfer/validation_metric_tree_${sample_type}.npy \
    --method 'base_decision_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]} \
    --model-min-width 32


python evaluate_diagnosis_acc.py \
    --feature-list 'train_error' 'mode_connectivity_peak' 'hessian_t' 'mode_connectivity_peak' 'hessian_t' 'hessian_t' \
    --save-path results/${label}/scale_transfer/md_tree_${sample_type}.npy \
    --method 'md_tree' \
    --label ${label} \
    --subtree-type ${subtree_type} \
    --trainset-sampletype ${sample_type} \
    --trainset-samplefactor ${trainset_factors[@]} \
    --model-min-width 32

