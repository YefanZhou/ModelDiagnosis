train_path='../model_diagnosis_dataset/dictionary/dict_13_13_10_normal_public.npy'
test_path='../model_diagnosis_dataset/dictionary/dict_13_13_10_noise_public.npy'

################################################################################################
# # optimal method
for task in 'bs_failure' 'width_vs_batch'
    do 
        python one_step_hpo.py \
            --seed-lst 42 90 38 18 72 \
            --save-path './hpo_results/' \
            --base-path '..' \
            --train-path ${train_path} \
            --test-path ${test_path} \
            --method 'optimal' \
            --sample-type 'random' \
            --class-label ${task} 
    done

# ################################################################################################
# # random method
for task in 'bs_failure' 'width_vs_batch'
    do 
        python one_step_hpo.py \
            --seed-lst 42 90 38 18 72 \
            --save-path './hpo_results/' \
            --base-path '..' \
            --train-path ${train_path} \
            --test-path ${test_path} \
            --method 'random' \
            --sample-type 'random' \
            --class-label ${task} 
    done







