# Model Diagnosis Dataset
The preprocessed datasets will be downloaded after `git clone` the repository. We first give a overview of the datasets, and show how we preprocess the data.

## Overview of the dataset
Here are subcollections used in our study:
* The `basic_collection` (denoted as $\mathcal{F}$ in the paper) contains 1,690 configurations, featuring various ResNet models trained on the CIFAR-10 dataset. These configurations differ in three factors: the number of model parameters is varied by changing the width of ResNet-18 among {2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128}, the optimizer hyperparameter is varied by changing the batch size among {16, 24, 32, 44, 64, 92, 128, 180, 256, 364, 512, 724, 1024}, the amount of data samples is varied via subsampling the subset from CIFAR-10 training set between 10% to 100%, with increments of 10%.
* The `noisy_collection` (denoted as $\mathcal{F}^{\prime}$ in the paper) also contains 1,690 configurations with the same varying factors, but each model is trained with 10% label noise.

The `basic_collection` is provided in *python dictionary* in `model_diagnosis_dataset/dictionary/dict_13_13_10_normal_public.npy` (with the key being `f'w_{width}_t_{bs}_d_{data}'`), and is provided in *pandas dataframe* in `model_diagnosis_dataset/dataframe/dataframe_normal_public.csv`.

The `noisy_collection` is provided in *python dictionary* in `model_diagnosis_dataset/dictionary/dict_13_13_10_noise_public.npy`, and is provided in *pandas dataframe* in `model_diagnosis_dataset/dataframe/dataframe_noise_public.csv`.


## Preprocessing
The preprocessing includes the following steps: 
1. convert the unstructured metric data to structures such as dictionary and dataframe, 
2. compute the room for improvement (RFI) metric, which quantifies the impact of a specific failure source 
3. labeling the most severe failure sources for diagnosis task.
To preprocess these collections, we need to first download the unstructured raw data from [google drive](https://drive.google.com/drive/folders/14TrdMDkKwZWVnCTSFfwO0I2gQnFw8Sa4?usp=sharing), and then run the following commands:

```bash
cd model_diagnosis_dataset
# Constructing the dictionary, computing the room for improvement, for `basic_collection`
python generate_data.py --generate-grid --generate-dictionary 

# Converting dictionary to dataframe, and labeling the failure source (This step is not necessary for preprocessing but useful for dataset exploration, it will execute in real-time during experiments.)
python generate_data.py --generate-dataframe 
```
Adding `--model-w-label-noise` on each command for repeating the same procedure for `noisy_collection`.


## Features for diagnosis
Each data sample (one configuration, one row of dataframe) has the following properties (columns of dataframe):
* `data_amount`: amount of training samples
* `para_amount`: number of model parameters
* `bs`: batch size
* `train_error`: classification error of models evaluated on training set
* `test_error`: classification error of models evaluated on test set
* `train_loss`: cross entropy loss of models evaluated on training set
* `test_loss`: cross entropy loss of models evaluated on test set
* `hessian_t`: hessian trace of models computed using training set
* `hessian_e`: top eigenvalue of Hessian computed using training set
* `mode_connectivity_peak`: training error peak of Bezier curves (connecting two models trained with different random seeds) using training set data
* `CKA`: CKA similarity between output of two models trained with different random seeds using training set data
* `rfi_inc_temp`: room for improvement of increasing temperature (meaning decreasing batch size)
* `rfi_dec_temp`: room for improvement of decreasing temperature (meaning increasing batch size)
* `rfi_width`: room for improvement of increasing width (meaning increasing number of model parameters)
* `rfi_data`: room for improvement of increasing data amount
* `bs_failure`: label of failure source, the batch size is {large or small}
* `width_vs_batch`: label of failure source, the failure source is {width or batch size}
* `data_vs_batch`: label of failure source, the failure source is {data or batch size}


The models are trained and the loss landscape metrics are computed using the code from [loss_landscape_taxonomy](https://github.com/nsfzyzz/loss_landscape_taxonomy/tree/main).
