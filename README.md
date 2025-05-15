
# stepAIC - Stepwise and Lasso linear regression to minimimize AIC or BIC in Python and Jupyter Notebook

The stepAIC package includes two functions, **stepwise** and **lasso**, to find the optimum set of predictor variables that minimizes either the Akaike Information Criterion (AIC, default) or Bayesian Information Criterion (BIC, optional) in a linear regression model.

The choice between Lasso regression and stepwise regression depends on the specific context and requirements of the analysis. Lasso regression is generally preferred for its efficiency and ability to handle large datasets without overfitting. However, stepwise regression can be more suitable for exploratory data analysis and when the goal is to identify the most influential predictors. It is also less prone to the issues of inflated type I error rates and the complexity of overfitting that can arise with Lasso regression. Ultimately, the best choice depends on the data characteristics and the researcher's objectives.

### Stepwise

The **stepwise** function in the stepAIC package has the option to use either forward selection (default), backward selection, or all combinations for the optimum set of predictor variables as follows:

- Forward selection (default) starts with no predictors and adds predictors as long as it improves the model (reduces AIC or BIC) 
- Backward selection starts with all predictors and removes predictors as long as it improves the model (reduces AIC or BIC)
- All possible combinations of predictors to find the best of all possible models (up to 20 candidate predictors)

The stepwise algorithm also has the option (default) to remove any non-signficant predictors (p-values below a user-specified p_threshold with default p_threshold=0.05) after either a forward or backward search. 

### Lasso

The **lasso** function in the stepAIC package provides output of regression models and summary statistics using the following four methods from the sklearn.linear_model package:

- LassoCV: Lasso using Cross-Validation with coordinate descent  
- LassoLarsCV: Lasso using Cross-Validation with Least Angle Regression
- LassoLarsIC using AIC: Lasso using Least Angle Regression with Akaike Information Criterion
- LassoLarsIC using BIC: Lasso using Least Angle Regression with Bayesian Information Criterion

Lasso linear regression includes a penalty term to the standard least squares objective function. The penalty term is a sum of the absolute values of the regression coefficients multiplied by a hyperparameter, denoted as "alpha". The **lasso** function finds the optimum value of alpha for each of the four different methods listed above. The alpha determines the amount of shrinkage applied to the model coefficients. As alpha increases, the coefficients are pushed towards zero, and some may become exactly zero, effectively eliminating those features from the model. 

### AIC vs BIC

Using AIC as the criterion is the default in the stepAIC fuction. The user also has the option to use the BIC as the criterion instead. AIC is considered to be a useful critierion in stepwise regression. However, BIC is generally considered to be better than AIC for several reasons:

- Penalty for Complexity: BIC penalizes models more heavily for the number of parameters, making it more conservative and less likely to overfit, especially with larger sample sizes.
- Model Selection: BIC is particularly useful when the sample size is large, as it encourages simpler models that are less likely to capture noise.
- Model Recovery: Studies suggest that BIC tends to recover the true model more effectively than AIC, particularly in scenarios where the sample size is large.
While both criteria are useful for model selection, BIC is often preferred for its stricter criteria, which helps in avoiding overfitting and improving model interpretability

The stepAIC function requires that you have already installed numpy, pandas, scikit-learn, tabulate, and statsmodels packages. We also recommend that you have installed seaborn and matplotlib.

# Installation for Python or Jupyter Notebook

If you have not already installed stepAIC, enter the following with pip or !pip in your notebook or terminal:<br>
```
pip install git+https://github.com/gjpelletier/stepAIC.git
```

if you are upgrading from a previous installation of stepAIC, enter the following with pip pr !pip in your notebook or terminal:<br>
```
pip install git+https://github.com/gjpelletier/stepAIC.git --upgrade
```

Next import the stepAIC function as follows in your notebook or python code:<br>
```
from stepAIC import stepwise, lasso
```

# Example

The [Stepwise_and_Lasso_example.ipynb](https://github.com/gjpelletier/stepAIC/blob/main/Stepwise_and_Lasso_example.ipynb) Jupyter notebook presents examples of the use of the stepwise and Lasso linear regression functions.

