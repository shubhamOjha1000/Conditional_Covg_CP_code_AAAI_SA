# Beyond Marginal Coverage in Conformal Prediction for Pathological Image Classification

- #### Authors</ins>: **Siddharth Narendra,  Shubham Ojha, Aditya Narendra, Abhay Kshirsaga, Abhishek Mallick**

### Abstract


Conformal Prediction (CP) is an uncertainty estimation
framework that generates prediction sets, ensuring that the
true class is included with a user-specified probability known
as coverage. This coverage is typically marginal (averaged),
meaning that the probability of the true label being included
in the prediction sets matches the specified confidence level
across all test cases. This can lead to inconsistent coverage
across different classes, which may reduce diagnostic preci-
sion. In clinical settings, achieving class-conditional cover-
age is crucial, where coverage is assured for every ground
truth class. This study implements a Classwise CP method
applied to two cancer cell classification datasets to achieve
class-conditional coverage. Our results demonstrate the ef-
fectiveness of this approach through a significant reduction
in the average class coverage gap compared to the Baseline
CP method.



## Dataset Guide

### Datasets Used
- **Center for Recognition and Inspection of Cells (CRIC) Dataset** : [Download Link](https://database.cric.com.br/downloads)
- **Breast Cancer Histopathological Database (BreakHis) Dataset**: [Download Link](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)


### Extract Cells from the Patches 
- To extract a cell crop size of 100 x 100 centred on the nucleus from the patches.
```
python Cropping.py --dataset='path to the cell centre coordinates csv file' --img_dir='path to the cric img patches directory' --cell_img_dir='path to cell img directory'
```

### Feature Extractor
- To extract features from the image patches.
```
python Feature_Extractor.py --img_dir='path to the image directory' --csv_file='path to the csv file' --model='Model to be used for feature extraction' --batch_size='Batch size for the dataloader' --num_workers='num workers for dataloader'
```

### Train-Val-Test Split for image features 
- To split the image feature containing csv file into train, val and test.
```
python feature_diet_Train_val_test_split.py --dataset='path to the img feature file' --split='Train/Test split ratio' --folds='No of folds in K-folds'
```


- Train the model using cross entropy loss
```
python feature_main.py --loss='cross_entropy' --feat_dir='path to the feature directory' --num_epochs='Number of total training epochs' --model='Model to be used' --batch_size='Batch size for the dataloader' --num_workers='num workers for dataloader' --dataset='Dataset used :- Breast_cancer/Cervical_cancer '
```

## Evaluation Methods

### For BreakHis Dataset

- Eval CP method for baseline results 

```
python Breast_cancer_Eval_CP_method.py --Trials='Number of total trials' --softmax_output_file_path='path to the softmax_output_file' --split='Calib/test split ratio'
```

- Eval CP method for class conditional covg
```
python Breast_cancer_Class_conditional_covg_eval_CP_method.py --Trials='Number of total trials' --softmax_output_file_path='path to the softmax_output_file' --split='Calib/test split ratio'
```


### For CRIC Dataset

- Eval CP method for baseline results 
```
python Cervical_cancer_Eval_CP_method.py --Trials='Number of total trials' --softmax_output_file_path='path to the softmax_output_file' --split='Calib/test split ratio'
```

- Eval CP method for class conditional covg
```
python Cervical_cancer_Class_conditional_covg_eval_CP_method.py --Trials='Number of total trials' --softmax_output_file_path='path to the softmax_output_file' --split='Calib/test split ratio' 
```
