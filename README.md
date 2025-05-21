
# stepAIC - Stepwise, Lasso, and Ridge linear regression to minimimize AIC or BIC in Python and Jupyter Notebook

The stepAIC package includes three functions, **stepwise**, and **lasso**, and **ridge**, to find the optimum set of predictor variables that minimizes either the Akaike Information Criterion (AIC, default) or Bayesian Information Criterion (BIC, optional) in a linear regression model.

The choice between Lasso, Ridge, or Stepwise regression depends on the specific context and requirements of the analysis. Stepwise regression is widely used ([e.g. Murtaugh, 2009](https://doi.org/10.1111/j.1461-0248.2009.01361.x)), but often criticized ([e.g. Flom and Cassell, 2007](https://www.lexjansen.com/pnwsug/2008/DavidCassell-StoppingStepwise.pdf)). Lasso and Ridge regression are generally preferred for their efficiency and ability to handle large datasets without overfitting. However, Stepwise regression can be more suitable for exploratory data analysis and when the goal is to identify the most influential predictors. Ultimately, the best choice depends on the data characteristics and the researcher's objectives.

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

Lasso (Least Absolute Shrinkage and Selection Operator) adds an L1 penalty to the loss function. This penalty encourages sparsity in the model, meaning that some coefficients will be exactly zero, effectively removing the corresponding predictors from the model. 

Lasso linear regression includes a penalty term to the standard least squares objective function. The penalty term is a sum of the absolute values of the regression coefficients multiplied by a hyperparameter, denoted as "alpha". The **lasso** function finds the optimum value of alpha for each of the four different methods listed above. The alpha determines the amount of shrinkage applied to the model coefficients. As alpha increases, the coefficients are pushed towards zero, and some may become exactly zero, effectively eliminating those features from the model. 

Lasso regression is useful for dealing with multicollinearity, where predictors are highly correlated, and when an optimal subset of the candidate features should be included in the model. 

### Ridge

The **ridge** function in the stepAIC package provides output of regression models and summary statistics using the following three methods from the sklearn.linear_model package:

- RidgeCV: RidgeCV regression with default cross-validation using the MSE as the scoring criterion to select alpha
- Ridge_AIC: Ridge regression using AIC as the scoring criterion to select alpha by trial
- Ridge_BIC: Ridge regression using BIC as the scoring criterion to select alpha by trial

Ridge regression adds an L2 penalty to the loss function, which is the product of the regularization hyperparameter and the sum of the squares of the coefficients. This penalty shrinks the coefficients towards zero but does not force them to be exactly zero. 

Ridge regression is useful for dealing with multicollinearity, where predictors are highly correlated, and when all candidate features should be included in the model. 

### Comparison of Stepwise, Lasso, and Ridge

- Feature selection: Lasso performs explicit feature selection by setting some coefficients to zero, while Ridge shrinks coefficients but retains all predictors. Stepwise regression also performs feature selection but can be less stable than Lasso. 
- Regularization: Both Lasso and Ridge are regularization techniques that prevent overfitting, but they do so differently. Lasso is more likely to produce sparse models, while Ridge is more likely to shrink coefficients smoothly. 
- Computational cost: Stepwise regression can be computationally expensive, especially for large datasets. Lasso and Ridge can be solved more efficiently using optimization algorithms. 

### AIC vs BIC

Using AIC as the criterion is the default in the stepAIC **stepwise** fuction. The user also has the option to use the BIC as the criterion instead. AIC is considered to be a useful critierion in stepwise regression. However, BIC is generally considered to be better than AIC for several reasons:

- Penalty for Complexity: BIC penalizes models more heavily for the number of parameters, making it more conservative and less likely to overfit, especially with larger sample sizes.
- Model Selection: BIC is particularly useful when the sample size is large, as it encourages simpler models that are less likely to capture noise.
- Model Recovery: Studies suggest that BIC tends to recover the true model more effectively than AIC, particularly in scenarios where the sample size is large.
While both criteria are useful for model selection, BIC is often preferred for its stricter criteria, which helps in avoiding overfitting and improving model interpretability

### Limitations of Ridge regression for feature selection and the utility of AIC and BIC

Unlike Lasso regression, Ridge regression does not have zeroing of selected coefficients as a goal. Therefore, Ridge regression generally does not select a subset of features for the final best model. Instead, Ridge regression retains all of the candiate features in the input data set and has the goal of minimizing the coefficient values as a strategy to reduce the variance inflation factors to mitigate the effects of multicollinearity.

AIC and BIC have limited value in optimizing Ridge regression. The AIC and BIC in Ridge regression is not sensitive to the alpha parameter because the AIC and BIC values are strongly affected by the number of model parameters. As the alpha parameter is adjusted, the AIC and BIC values change by a relatively small amount depending on the variance of the residuals at each value of alpha. This means that the AIC and BIC values across a wide range of alpha values do not penalize the model for having too many parameters in Ridge regression. Using AIC and BIC have the effect of choosing the lowest value of alpha, which is similar to performing ordinary linear regression without regularaization and with no mitigation of multicollinearity.

If feature selection is the goal of the analysis, then Stepwise or Lasso regression methods are generally better than Ridge regression for that purpose. If your analysis requires that all candidate features are retained in the final model, then Ridge regression is ideal for that purpose using the **ridge** results for RidgCV.

## Installation for Python or Jupyter Notebook

The stepAIC functions require that you have already installed numpy, pandas, scikit-learn, tabulate, matplotlib, and statsmodels packages. 

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

## Example 1. Use Lasso regression to analyze diabetes data

In this example we will use Lasso regression to analyze the diabetes data available from sklearn. The lasso function uses the sklearn.processing StandardScaler to standardize the X values by default. Then the lasso function uses the standardized X values to find each of the best fit models using LassoCV, LassoLarsCV, LassoLarsIC using AIC, and LassoLarsIC using BIC.

Run the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)
X.head()

# Use the lasso function in the stepAIC package
from stepAIC import lasso
model_objects, model_outputs = lasso(X, y)
```

Running the code above produces the following display of output tables with regression statistics and best-fit coefficients for each model (LassoCV, LassoLarsCV, LassoLarsIC using AIC, LassoLarsIC using BIC):
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


| Feature   |   LassoCV |   LassoLarsCV |   LassoLarsAIC |   LassoLarsBIC |
|:----------|----------:|--------------:|---------------:|---------------:|
| const     | 152.133   |     152.133   |      152.133   |      152.133   |
| age       |  -0       |       0       |        0       |        0       |
| sex       |  -9.11162 |      -9.13079 |       -9.40617 |       -9.40617 |
| bmi       |  24.8066  |      24.809   |       24.8419  |       24.8419  |
| bp        |  13.9806  |      13.9909  |       14.1342  |       14.1342  |
| s1        |  -4.58713 |      -4.61047 |       -4.94418 |       -4.94418 |
| s2        |  -0       |       0       |        0       |        0       |
| s3        | -10.5553  |     -10.5615  |      -10.651   |      -10.651   |
| s4        |   0       |       0       |        0       |        0       |
| s5        |  24.2697  |      24.2839  |       24.4841  |       24.4841  |
| s6        |   2.45869 |       2.46804 |        2.6051  |        2.6051  |


Variance Inflation Factors model_outputs['vif']:
Note: VIF>5 indicates excessive collinearity


| Feature   |   LassoCV |   LassoLarsCV |   LassoLarsAIC |   LassoLarsBIC |
|:----------|----------:|--------------:|---------------:|---------------:|
| const     |   1       |       1       |        1       |        1       |
| age       | nan       |     nan       |      nan       |      nan       |
| sex       |   1.25594 |       1.25594 |        1.25594 |        1.25594 |
| bmi       |   1.49419 |       1.49419 |        1.49419 |        1.49419 |
| bp        |   1.39388 |       1.39388 |        1.39388 |        1.39388 |
| s1        |   1.58065 |       1.58065 |        1.58065 |        1.58065 |
| s2        | nan       |     nan       |      nan       |      nan       |
| s3        |   1.66245 |       1.66245 |        1.66245 |        1.66245 |
| s4        | nan       |     nan       |      nan       |      nan       |
| s5        |   2.07745 |       2.07745 |        2.07745 |        2.07745 |
| s6        |   1.45162 |       1.45162 |        1.45162 |        1.45162 |
```

Note that the most parsimonious model was found using LassoLarsIC with BIC as the criterion. If the goal of the study is to find the most parsimonious model with the smallest subset of features, then using BIC as the criterion appears to provide the best results.

The model_objects and model_outputs returned by the lasso function also contain the best-fit sklearn model objects and many other useful outputs as described by help(lasso). All of the optional arguments for the lasso function are also explained by running help(lasso) 

## Example 2. Use Stepwise regression to analyze diabetes data

In this example we will use Stepwise regression to analyze the diabetes data available from sklearn.

We will use the following optional arguments:

- criterion='bic' to use BIC as the criterion
- direction='all' to search for the best model from all possible combinations of features
- standardize='on' to use the sklearn.processing StandardScaler to standardize the X values

Run the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)
X.head()

# Use the stepwise function in the stepAIC package
from stepAIC import stepwise
model_object, model_output = stepwise(X, y, 
    criterion='bic', direction='all', standardize='on')
```

Running the code above produces the following output display of the best fit model:
```
Best 10 subsets of features in model_outputs['step_features']:

|   Rank |     AIC |     BIC |   rsq_adj | Features                               |
|-------:|--------:|--------:|----------:|:---------------------------------------|
|      0 | 4792.26 | 4816.81 |  0.502997 | ['sex' 'bmi' 'bp' 's3' 's5']           |
|      1 | 4792.26 | 4816.81 |  0.502997 | ['sex' 'bmi' 'bp' 's3' 's5']           |
|      2 | 4788.6  | 4817.24 |  0.508193 | ['sex' 'bmi' 'bp' 's1' 's2' 's5']      |
|      3 | 4789.92 | 4818.56 |  0.506728 | ['sex' 'bmi' 'bp' 's1' 's4' 's5']      |
|      4 | 4790.12 | 4818.76 |  0.5065   | ['sex' 'bmi' 'bp' 's1' 's3' 's5']      |
|      5 | 4791.09 | 4819.73 |  0.505419 | ['sex' 'bmi' 'bp' 's2' 's3' 's5']      |
|      6 | 4793.18 | 4821.82 |  0.503078 | ['sex' 'bmi' 'bp' 's3' 's4' 's5']      |
|      7 | 4789.32 | 4822.05 |  0.508488 | ['sex' 'bmi' 'bp' 's1' 's2' 's4' 's5'] |
|      8 | 4789.37 | 4822.1  |  0.508429 | ['sex' 'bmi' 'bp' 's1' 's2' 's5' 's6'] |
|      9 | 4793.56 | 4822.2  |  0.502647 | ['sex' 'bmi' 'bp' 's3' 's5' 's6']      |

Best of all possible models after removing insignficant features if any:
Best features:  ['sex', 'bmi', 'bp', 's3', 's5'] 

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 target   R-squared:                       0.509
Model:                            OLS   Adj. R-squared:                  0.503
Method:                 Least Squares   F-statistic:                     90.26
Date:                Sun, 18 May 2025   Prob (F-statistic):           4.75e-65
Time:                        20:17:19   Log-Likelihood:                -2390.1
No. Observations:                 442   AIC:                             4792.
Df Residuals:                     436   BIC:                             4817.
Df Model:                           5                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        152.1335      2.585     58.849      0.000     147.053     157.214
sex          -11.2146      2.876     -3.899      0.000     -16.868      -5.562
bmi           24.9036      3.106      8.019      0.000      18.800      31.008
bp            15.5172      3.001      5.171      0.000       9.620      21.415
s3           -13.7518      3.122     -4.404      0.000     -19.889      -7.615
s5            22.5597      3.124      7.221      0.000      16.419      28.700
==============================================================================
Omnibus:                        2.465   Durbin-Watson:                   1.990
Prob(Omnibus):                  0.291   Jarque-Bera (JB):                2.099
Skew:                           0.051   Prob(JB):                        0.350
Kurtosis:                       2.678   Cond. No.                         2.33
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

Variance Inflation Factors of selected_features:
Note: VIF>5 indicates excessive collinearity

| Feature   |     VIF |
|:----------|--------:|
| const     | 1       |
| sex       | 1.23788 |
| bmi       | 1.44328 |
| bp        | 1.34724 |
| s3        | 1.45888 |
| s5        | 1.46057 |
```

Note that Example 2 using stepwise regression found a more parsimonious model with 5 features (compared with 7 features using Lasso regression), and similar skill compared with using Lasso regression for the same diabetes data set.

Also note that the model_object and model_output returned by the stepwise function also provide the best-fitted statsmodels model object and a dictionary of many useful regression results and summary statistics as explained by running help(stepwise).

## Example 3. Use Ridge regression to analyze diabetes data

In this example we will use Ridge regression to analyze the diabetes data available from sklearn. The ridge function uses the sklearn.processing StandardScaler to standardize the X values by default. Then the ridge function uses the standardized X values to find each of the best fit models using RidgeCV (default using MSE for optimization), and Ridge using AIC and BIC for optimization.

Run the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)
X.head()

# Use the lasso function in the stepAIC package
from stepAIC import lasso
model_objects, model_outputs = lasso(X, y)
```

Running the code above produces the following output display of the best fit model:
```
Ridge regression statistics of best models in model_outputs['stats']:


| Statistic          |         RidgeCV |       Ridge_AIC |       Ridge_BIC |
|:-------------------|----------------:|----------------:|----------------:|
| alpha              |     1.81242     |     1e-06       |     1e-06       |
| r-squared          |     0.517348    |     0.517748    |     0.517748    |
| adjusted r-squared |     0.505001    |     0.505412    |     0.505412    |
| nobs               |   442           |   442           |   442           |
| df residuals       |   431           |   431           |   431           |
| df model           |    10           |    10           |    10           |
| F-statistic        |    46.1983      |    46.2724      |    46.2724      |
| Prob (F-statistic) |     1.11022e-16 |     1.11022e-16 |     1.11022e-16 |
| RMSE               |    53.4983      |    53.4761      |    53.4761      |
| Log-Likelihood     | -2386.18        | -2385.99        | -2385.99        |
| AIC                |  4794.35        |  4793.99        |  4793.99        |
| BIC                |  4839.36        |  4838.99        |  4838.99        |


Coefficients of best models in model_outputs['popt']:


| Feature   |    RidgeCV |   Ridge_AIC |   Ridge_BIC |
|:----------|-----------:|------------:|------------:|
| const     | 152.133    |  152.133    |  152.133    |
| age       |  -0.403896 |   -0.476121 |   -0.476121 |
| sex       | -11.2846   |  -11.4069   |  -11.4069   |
| bmi       |  24.7844   |   24.7265   |   24.7265   |
| bp        |  15.3371   |   15.4294   |   15.4294   |
| s1        | -25.917    |  -37.6799   |  -37.6799   |
| s2        |  13.345    |   22.6762   |   22.6762   |
| s3        |  -0.363515 |    4.80613  |    4.80613  |
| s4        |   7.04127  |    8.42204  |    8.42204  |
| s5        |  31.24     |   35.7344   |   35.7344   |
| s6        |   3.29929  |    3.21667  |    3.21667  |


Variance Inflation Factors model_outputs['vif']:
Note: VIF>5 indicates excessive collinearity


| Feature   |    RidgeCV |   Ridge_AIC |   Ridge_BIC |
|:----------|-----------:|------------:|------------:|
| age       | 0.00582757 |     1.21727 |     1.21727 |
| sex       | 0.00585865 |     1.27803 |     1.27803 |
| bmi       | 0.00550864 |     1.50936 |     1.50936 |
| bp        | 0.00560622 |     1.45937 |     1.45937 |
| s1        | 0.00493996 |    59.0691  |    59.0691  |
| s2        | 0.00498344 |    39.109   |    39.109   |
| s3        | 0.00532967 |    15.3747  |    15.3747  |
| s4        | 0.00460323 |     8.88691 |     8.88691 |
| s5        | 0.00510341 |    10.0573  |    10.0573  |
| s6        | 0.00544229 |     1.48457 |     1.48457 |
```

Note that the VIF results for Ridge_AIC and Ridge_BIC indicate excessive multicollinearity. This is because AIC and BIC have limited value in optimizing Ridge regression. The lowest AIC and BIC values occur at the lowest vales of alpha, which is similar to performing ordinary linear regression with no regularization and no reduction in VIF. The AIC and BIC in Ridge regression is not sensitive to the alpha parameter because the AIC and BIC values are strongly affected by the number of model parameters. As the alpha parameter is adjusted, the AIC and BIC values change by a relatively small amount depending on the variance of the residuals at each value of alpha. This means that the AIC and BIC values across a wide range of alpha values do not penalize the model for having too many parameters in Ridge regression.



