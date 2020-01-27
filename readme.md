# Project description
</br>
Learn best Linear Regression model on housing data using RFECV feature selection. </br> 
Model 1: default features </br>
Model 2: with polynomial features (degree 2)

# Feature visualizations - examples

![alt text](https://github.com/mateusz-g94/DS-Housing-regression/blob/master/grp/price.png)

</br>

![alt text](https://github.com/mateusz-g94/DS-Housing-regression/blob/master/grp/grade.png)

</br>

# Corelation heatmap - only for numeric variables

![alt text](https://github.com/mateusz-g94/DS-Housing-regression/blob/master/grp/corr.png)

# Model 1
## RFECV result
Optimal number of features: 13
</br>

![alt text](https://github.com/mateusz-g94/DS-Housing-regression/blob/master/grp/model1_rfecv_feature_selection.png)

## Model summary 

![alt text](https://github.com/mateusz-g94/DS-Housing-regression/blob/master/grp/model1_summ.png)

## Regression diagnostics - on test set

![alt text](https://github.com/mateusz-g94/DS-Housing-regression/blob/master/grp/model1_test_residuals_hist.png)

</br>

![alt text](https://github.com/mateusz-g94/DS-Housing-regression/blob/master/grp/model1_test_y_vs_prediction.png)

</br>

![alt text](https://github.com/mateusz-g94/DS-Housing-regression/blob/master/grp/model1_test_residuals_vs_y.png)

</br>
Graph shows that error increases with increasing price. This leads to hypotesis that the form of the model isn't linear. Adding polynomial features should lead to model improvment. We can expect that price of expensive apartments depends on other factors then the price of cheap apartments.  

# Model 2
## RFECV result
Optimal number of features: 118
</br>

![alt text](https://github.com/mateusz-g94/DS-Housing-regression/blob/master/grp/model2_rfecv_feature_selection.png)

## Model summary 

![alt text](https://github.com/mateusz-g94/DS-Housing-regression/blob/master/grp/model2_summ.png)

## Regression diagnostics - on test set

![alt text](https://github.com/mateusz-g94/DS-Housing-regression/blob/master/grp/model2_test_residuals_hist.png)

</br>

![alt text](https://github.com/mateusz-g94/DS-Housing-regression/blob/master/grp/model2_test_y_vs_prediction.png)

</br>

![alt text](https://github.com/mateusz-g94/DS-Housing-regression/blob/master/grp/model2_test_residuals_vs_y.png)

</br> 
Graph shows model improvment compared to model 1 without polynomial features. 
