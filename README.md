
# stepAIC - Stepwise and Lasso linear regression to minimimize AIC or BIC in Python and Jupyter Notebook

The stepAIC package includes two functions, **stepwise** and **lasso**, to find the optimum set of predictor variables that minimizes either the Akaike Information Criterion (AIC, default) or Bayesian Information Criterion (BIC, optional) in a linear regression model.

The choice between Lasso regression and stepwise regression depends on the specific context and requirements of the analysis. Stepwise regression is widely used ([e.g. Murtaugh, 2009](https://doi.org/10.1111/j.1461-0248.2009.01361.x)), but often criticized ([e.g. Flom and Cassell, 2007](https://www.lexjansen.com/pnwsug/2008/DavidCassell-StoppingStepwise.pdf)). Lasso regression is generally preferred for its efficiency and ability to handle large datasets without overfitting. However, stepwise regression can be more suitable for exploratory data analysis and when the goal is to identify the most influential predictors. Ultimately, the best choice depends on the data characteristics and the researcher's objectives.

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

# Example 1. Use Lasso regression to analyze the diabetes data set provided by sklearn

Running the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)
X.head()

# Use the lasso function in the stepAIC package
from stepAIC import lasso
model_objects, model_outputs = lasso(X, y)
```

Produces the following display of output:
```
Lasso regression statistics of best models in model_outputs['stats']:

| Statistic          |         LassoCV |     LassoLarsCV |    LassoLarsAIC |    LassoLarsBIC |
|:-------------------|----------------:|----------------:|----------------:|----------------:|
| alpha              |     1.11865     |     1.10767     |     0.950407    |     0.950407    |
| r-squared          |     0.512957    |     0.512989    |     0.51341     |     0.51341     |
| adjusted r-squared |     0.503959    |     0.503991    |     0.50442     |     0.50442     |
| nobs               |   442           |   442           |   442           |   442           |
| df residuals       |   434           |   434           |   434           |   434           |
| df model           |     7           |     7           |     7           |     7           |
| F-statistic        |    65.2989      |    65.3073      |    65.4173      |    65.4173      |
| Prob (F-statistic) |     1.11022e-16 |     1.11022e-16 |     1.11022e-16 |     1.11022e-16 |
| RMSE               |    53.7411      |    53.7393      |    53.7161      |    53.7161      |
| Log-Likelihood     | -2388.18        | -2388.16        | -2387.97        | -2387.97        |
| AIC                |  4792.36        |  4792.33        |  4790           |  4790           |
| BIC                |  4825.09        |  4825.06        |  4818.64        |  4818.64        |

Coefficients of best models in model_outputs['popt']:

| Coefficient   |   LassoCV |   LassoLarsCV |   LassoLarsAIC |   LassoLarsBIC |
|:--------------|----------:|--------------:|---------------:|---------------:|
| constant      | 152.133   |     152.133   |      152.133   |      152.133   |
| age           |  -0       |       0       |        0       |        0       |
| sex           |  -9.11162 |      -9.13079 |       -9.40617 |       -9.40617 |
| bmi           |  24.8066  |      24.809   |       24.8419  |       24.8419  |
| bp            |  13.9806  |      13.9909  |       14.1342  |       14.1342  |
| s1            |  -4.58713 |      -4.61047 |       -4.94418 |       -4.94418 |
| s2            |  -0       |       0       |        0       |        0       |
| s3            | -10.5553  |     -10.5615  |      -10.651   |      -10.651   |
| s4            |   0       |       0       |        0       |        0       |
| s5            |  24.2697  |      24.2839  |       24.4841  |       24.4841  |
| s6            |   2.45869 |       2.46804 |        2.6051  |        2.6051  |
```

![Lasso_alpha_vs_coef](https://github.com/user-attachments/assets/b3bbde6a-32d5-4fee-bc51-8f260ded300e)

![LassoCV_alpha_vs_MSE](https://github.com/user-attachments/assets/5926b551-6b15-4e18-ad45-adad163d32a5)

![LassoLarsCV_alpha_vs_MSE](https://github.com/user-attachments/assets/a4ec7ccd-2ee7-4410-93c3-d22080fbe7b5)

![LassoLarsIC_alpha_vs_AIC_BIC](https://github.com/user-attachments/assets/804930e0-40ba-4480-8a51-4c80a8190940)

![LassoLarsIC_sequence_of_AIC_BIC](https://github.com/user-attachments/assets/ffbf630e-8f0f-47e1-abce-f4f007b33207)

![residuals](https://github.com/user-attachments/assets/187e1a01-573d-42d0-a3e5-e0a657e2e76e)







