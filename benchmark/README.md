## Benchmark 
Table A presents the evaluation setup and the corresponding figures in the paper. Table B presents a few selected setups for comparing different methods.

**Table A. The tasks and scenarios studied in the paper to evaluate the effectiveness of diagnosis methods and model features. The figure number refers to the figures in the paper. $k$ controls the different amounts/scales of pre-trained models sampled from the same training set. In the qualitative results in the paper, $k$ varies along the $x$-axis.**
|   Case           |  Task         | Scenario |  Training set |  Test set |   Qualitative results in the paper |  
|   --------     | --------      | -------- |  --------     |  -------- |    -------- | 
| Case 1|  Q1 (optimizer hyperparameter is large or small)   |  Dataset transfer   |   $k$ configs randomly sampled from `basic_collection` ($\mathcal{F}$)     | 1560 configs from `noisy_collection` ($\mathcal{F}^{\prime}$) |  Figure 3 (a)  | 
| Case 2|  Q1    |  Scale transfer    | configs with training (image) data $\leq$ $k$ from `basic_collection` ($\mathcal{F}$)     | 1560 configs from `noisy_collection` ($\mathcal{F}^{\prime}$) |   Figure 3 (b)   | 
| Case 3|  Q1    |  Scale transfer    | configs with model parameters $\leq$ $k$ from `basic_collection` ($\mathcal{F}$)     | 1560 configs from `noisy_collection` ($\mathcal{F}^{\prime}$) | Figure 3 (c)   | 
| Case 4|  Q2 (failure source is model size or optimizer hyperparameter)   |  Dataset transfer   |   $k$ configs randomly sampled from `basic_collection` ($\mathcal{F}$)     | 1560 configs from `noisy_collection` ($\mathcal{F}^{\prime}$) |   Figure 6 (a)  |  
| Case 5|  Q2    |  Scale transfer   |  configs with training (image) data $\leq$ $k$ from `basic_collection` ($\mathcal{F}$)     | 1560 configs from `noisy_collection` ($\mathcal{F}^{\prime}$) |    Figure 6 (b) | 



**Table B. Diagnosis accuracy ($\uparrow$, %) of different methods in diagnosing models. The results are averaged over 5 random seeds. The Task, Scenario, and Test set of the cases are shown in Table A. Table B provides the quantitative results when $k$ in each case of Table A is specified.**
|   Case   |  Training set   |             Hyperparameter + DT |  Validation + DT |  MD tree |
|   --------    | --------       | -------- |  --------     |  -------- | 
|   Case 1  |  96 configs randomly sampled from `basic_collection` ($\mathcal{F}$) |          49.56  |   72.82   |  **87.70**  |
|   Case 2  |  configs with training (image) data $\leq$ 5K from `basic_collection` ($\mathcal{F}$)             |  64.74  |   51.73   |  **75.89**  |
|   Case 3  |  configs with model parameters $\leq$ 0.011M from `basic_collection` ($\mathcal{F}$)            |  60.12  |   67.55   |  **82.56**  | 
|   Case 4  |  167 configs randomly sampled from `basic_collection` ($\mathcal{F}$)          |  53.54  |   55.79   |  **74.10**  |
|   Case 5  |  configs with training (image) data $\leq$ 5K from `basic_collection` ($\mathcal{F}$)        | 53.54  |   63.61   |  **78.16**  |