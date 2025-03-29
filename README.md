### Social Network Ads - Performance Prediction
## Project Overview
The project implements various __bagging__ and __boosting__ models to predict whether a user purchases a product after viewing its social network Ad.  
MLflow is used to handle the machine learning workflow. Various options for the python command line interface is also provided such as selecting which model to train, specifying parameters etc.

## Dataset Information
### columns
- `Age`
- `EstimatedSalary`
- `Purchased`
    - 0 : No
    - 1 : Yes

The dataset includes age and estimated salary of the user. The purchased column indicates weather the particular user with age and estimated salary have bought the product or not by viewing the social ads of the product .

## Dependencies
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `xgboost`
- `scikit-learn`

```
pip install pandas numpy seaborn matplotlib xgboost scikit-learn
```