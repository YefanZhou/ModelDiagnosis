train_path='../model_diagnosis_dataset/dictionary/dict_13_13_10_normal_public.npy'
test_path='../model_diagnosis_dataset/dictionary/dict_13_13_10_noise_public.npy'

# MD tree
################################################################################################
# MD tree  bs_failure
python one_step_hpo.py \
    --seed-lst 42 90 38 18 72 \
    --save-path './hpo_results/' \
    --base-path '..' \
    --train-path ${train_path} \
    --test-path ${test_path} \
    --tree-depth 4 \
    --method 'md_tree' \
    --subtree-type 'MDtree_temp' \
    --sample-type 'random' \
    --trainset-samplefactor 8 \
    --class-label 'bs_failure' \
    --feature-list 'train_error' 'mode_connectivity_peak' 'hessian_t'


# MD tree  width_vs_batch
python one_step_hpo.py \
    --seed-lst 42 90 38 18 72 \
    --save-path './hpo_results/' \
    --base-path '..' \
    --train-path ${train_path} \
    --test-path ${test_path} \
    --tree-depth 4 \
    --method 'md_tree' \
    --subtree-type 'MDtree_width' \
    --sample-type 'random_data' \
    --trainset-samplefactor 1 \
    --class-label 'width_vs_batch' \
    --feature-list 'train_error' 'mode_connectivity_peak' 'mode_connectivity_peak'



