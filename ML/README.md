### Classification Report
#### Overview
This report summarizes a classification task. The goal was to train a machine learning model for solving a classification task and evaluate its performance using various metrics.

####  Data Preprocessing

#### Models
We experimented with several machine learning models for this classification task. Here's a list of the models we implemented:

#### Data
Перед решением задачи было сделано следующее:
##### Data Cleaning
1. Features with a significant number of missing values were removed.
2. Features with fewer than 50% missing values were imputed with either the median or the most common value.
3. Constant features were removed.
4. Some features were logarithmically transformed.
5. Highly correlated features were eliminated.

##### Data Transformation
1. Features with fewer than five unique values were one-hot encoded (OHE).
2. Features with more than five unique values were encoded using Frequency Encoding.
3. Various features were extracted from the 'sample_date' column.

   



##### Model 1: [Logistic Regression]
Performance: 
```
Metrics:   train test
ROC_AUC:   0.743 0.74
Gini:      0.485 0.479
F1_score:  0.288 0.291
Log_loss:  11.136 11.621
```
![image](https://github.com/NikitaKuznetsow/Test-Task/assets/66497711/8c3c6d13-c111-4ec3-906d-57f215d18d15)


##### Model 2: [KNN]
Performance: 
```
Metrics:   train test
ROC_AUC:   0.906 0.638
Gini:      0.812 0.276
F1_score:  0.256 0.135
Log_loss:  3.143 3.777
```
![image](https://github.com/NikitaKuznetsow/Test-Task/assets/66497711/1183888d-e197-45d7-bb57-16766c63b3e3)

##### Model 3: [LightGBM]
Для этой модели гиперпарамтеры были оптимизированы с помощью бибилотеки optuna
Performance:
```
Metrics:   train test
ROC_AUC:   0.98 0.785
Gini:      0.959 0.57
F1_score:  0.641 0.371
Log_loss:  3.773 6.659
```
![image](https://github.com/NikitaKuznetsow/Test-Task/assets/66497711/4c0ca954-e309-4de3-ba7f-b3730c48c3ba)

#### Feature importance and More
##### Logistic Regression Weights
![image](https://github.com/NikitaKuznetsow/Test-Task/assets/66497711/93e4c509-ae60-4759-987b-c390ed006374)
##### LightGBM Feature Importance
![image](https://github.com/NikitaKuznetsow/Test-Task/assets/66497711/136fdea5-4507-49f0-babe-10388d8c43ed)
##### SHAP Values
###### Histogram
![image](https://github.com/NikitaKuznetsow/Test-Task/assets/66497711/ab4c4f77-f609-4880-a036-d03b1c704744)
###### Detailed Histogram
![image](https://github.com/NikitaKuznetsow/Test-Task/assets/66497711/348d7648-587d-49ef-9040-66810672b43b)
##### Accumulated Local Effects (ALE) for Best Features
<img width="561" alt="image" src="https://github.com/NikitaKuznetsow/Test-Task/assets/66497711/b3f938af-9cb6-4586-bd4f-17982c83b7df">

<img width="555" alt="image" src="https://github.com/NikitaKuznetsow/Test-Task/assets/66497711/4e5408f5-7384-4bcf-82b0-60e828244a41">

<img width="550" alt="image" src="https://github.com/NikitaKuznetsow/Test-Task/assets/66497711/ef14c85e-a7d9-4083-bf48-9d1e2374c4f4">

##### Conclusion
In the course of solving this classification task, we experimented with three models: K-Nearest Neighbors (KNN), Logistic Regression, and LightGBM. After hyperparameter optimization, LightGBM achieved the highest ROC AUC score of approximately 0.785, outperforming the other models. These features exhibited substantial influence on the final predictions, as evidenced by the ALE (Accumulated Local Effects) plots.

