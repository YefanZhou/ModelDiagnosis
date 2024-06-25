train_path='../model_diagnosis_dataset/dictionary/dict_13_13_10_normal_public.npy'
test_path='../model_diagnosis_dataset/dictionary/dict_13_13_10_noise_public.npy'

# # hyper tree
# ################################################################################################
# # hyper tree  bs_failure
python one_step_hpo.py \
    --seed-lst 42 90 38 18 72 \
    --save-path './hpo_results/' \
    --base-path '..' \
    --train-path ${train_path} \
    --test-path ${test_path} \
    --tree-depth 4 \
    --method 'decision_tree' \
    --sample-type 'random' \
    --trainset-samplefactor 8 \
    --class-label 'bs_failure' \
    --feature-list 'log_data_amount' 'log_para_amount'

# # hyper tree  width_vs_batch
python one_step_hpo.py \
    --seed-lst 42 90 38 18 72 \
    --save-path './hpo_results/' \
    --base-path '..' \
    --train-path ${train_path} \
    --test-path ${test_path} \
    --tree-depth 4 \
    --method 'decision_tree' \
    --sample-type 'random_data' \
    --trainset-samplefactor 1 \
    --class-label 'width_vs_batch' \
    --feature-list "log_data_amount"


# # validation tree
# ################################################################################################
# # validation tree  bs_failure
python one_step_hpo.py \
    --seed-lst 42 90 38 18 72 \
    --save-path './hpo_results/' \
    --base-path '..' \
    --train-path ${train_path} \
    --test-path ${test_path} \
    --tree-depth 4 \
    --method 'decision_tree' \
    --sample-type 'random' \
    --trainset-samplefactor 8 \
    --class-label 'bs_failure' \
    --feature-list "train_loss" "train_error" "test_error" "test_loss" 


# # validation tree  width_vs_batch
python one_step_hpo.py \
    --seed-lst 42 90 38 18 72 \
    --save-path './hpo_results/' \
    --base-path '..' \
    --train-path ${train_path} \
    --test-path ${test_path} \
    --tree-depth 4 \
    --method 'decision_tree' \
    --sample-type 'random_data' \
    --trainset-samplefactor 1 \
    --class-label 'width_vs_batch' \
    --feature-list "train_loss" "train_error" "test_error" "test_loss"






