
# Stepwise, Lasso, Ridge, Elastic Net, and Stacking linear regression to minimize MSE, AIC, BIC, or VIF in Python and Jupyter Notebook

## Note: This repository has been replaced by the PyMLR repository that is available at this link: https://github.com/gjpelletier/PyMLR

The stepAIC module began as a project to provide a Python alternative to the R stepAIC function. It grew to include five main functions, **stepwise**, **lasso**, **ridge**, **elastic**, and **stacking** to find the set of predictor variables that minimizes either the Mean Squared Error (MSE), Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC), adjusted rsquared, or Variance Inflation Factors (VIF) in a multiple linear regression model.

The choice between Lasso, Ridge, Elastic Net, Stepwise, or Stacking regression depends on the specific context and requirements of the analysis. Stepwise regression is widely used ([e.g. Murtaugh, 2009](https://doi.org/10.1111/j.1461-0248.2009.01361.x)), but often criticized ([e.g. Flom and Cassell, 2007](https://www.lexjansen.com/pnwsug/2008/DavidCassell-StoppingStepwise.pdf)). Lasso, Ridge, and Elastic Net regression are generally preferred for their efficiency and ability to handle large datasets without overfitting. However, Stepwise regression can be more suitable for exploratory data analysis and when the goal is to identify the most influential predictors. Ultimately, the best choice depends on the data characteristics and the researcher's objectives. Stacking regression is an ensemble machine learning technique that improves predictive accuracy by combining multiple base regression models.

The functions in the stepAIC module fit the linear regression models using solvers from statsmodels and sklearn, and add functionality to make it easier to apply those methods. 

In one line of python code the user can display and save the output of the following:

- regression summary statistics of the best fit model (e.g. r-squared, adjusted r-squared, RMSE of residuals, p-value of the F-statistic, AIC, BIC, alpha, etc.)
- best fit intercept and model coefficients
- variance  inflation factors
- residual plots
- figures showing other diagnostic descriptions of the analysis (e.g. coefficients vs alpha, MSE vs alpha, AIC and BIC vs alpha) 

### Stepwise

The **stepwise** function in the stepAIC module has the option to use either forward selection (default), backward selection, or all subsets of possible combinations for the optimum set of predictor variables as follows using statsmodels OLS:

- Forward selection (default) starts with no predictors and adds predictors as long as it improves the model (reduces AIC or BIC, or increases adjusted rsquared) 
- Backward selection starts with all predictors and removes predictors as long as it improves the model (reduces AIC or BIC, or increases adjusted rsquared)
- All subsets of possible combinations of predictor features to find the best of all possible models (up to 20 candidate predictors)

Either the AIC, BIC, or adjusted rsquared may be used as the criterion with forward, backward, or all subsets. In addition, there is an option to find all features with p-values less than a signficance threshold through backward elimination based only on the p-values of the coefficients. The stepwise algorithm also has the option (default) to remove any non-signficant predictors after either a forward, backward, or all subsets search using the AIC, BIC, or adjusted rsquared criterion. 

### Lasso

The **lasso** function in the stepAIC module provides output of regression models and summary statistics using the following methods using sklearn.linear_model:

- LassoCV: Lasso using Cross-Validation with coordinate descent to optimize alpha
- LassoLarsCV: Lasso using Cross-Validation with Least Angle Regression
- LassoLarsIC using AIC: Lasso using Least Angle Regression with Akaike Information Criterion
- LassoLarsIC using BIC: Lasso using Least Angle Regression with Bayesian Information Criterion

Lasso (Least Absolute Shrinkage and Selection Operator) adds a penalty to the loss function. This penalty encourages sparsity in the model, meaning that some coefficients will be exactly zero, effectively removing the corresponding predictors from the model. 

Lasso linear regression includes an L1 penalty term to the standard least squares objective function. The penalty term is a sum of the absolute values of the regression coefficients multiplied by a hyperparameter, denoted as "alpha". The **lasso** function finds the optimum value of alpha for each of the methods listed above. The alpha determines the amount of shrinkage applied to the model coefficients. As alpha increases, the coefficients are pushed towards zero, and some may become exactly zero, effectively eliminating those features from the model. 

Lasso regression is useful for dealing with multicollinearity, where predictors are highly correlated, and when an optimal subset of the candidate features should be included in the model. 

### Ridge

The **ridge** function in the stepAIC module provides output of regression models and summary statistics using the following methods using sklearn.linear_model:

- RidgeCV: RidgeCV regression with default cross-validation using the MSE as the scoring criterion to optimize alpha
- RidgeAIC: Ridge regression using AIC as the scoring criterion to optimize alpha by trial
- RidgeBIC: Ridge regression using BIC as the scoring criterion to optimize alpha by trial
- RidgeVIF: Ridge regression using target VIF to optimize alpha by trial

Ridge regression adds an L2 penalty to the loss function, which is the product of the regularization hyperparameter and the sum of the squares of the coefficients. This penalty shrinks the coefficients towards zero but does not force them to be exactly zero. 

Ridge regression is useful for dealing with multicollinearity, where predictors are highly correlated, and when all candidate features should be included in the model. 

### Elastic Net

The **elastic** function in the stepAIC module provides output of the fitted regression model and summary statistics using the following method using sklearn.linear_model:

- ElasticNetCV: Elastic regression with cross-validation using the MSE as the scoring criterion to optimize alpha and the L1-ratio that balances between L1 and L2 regularization.

Elastic regression, also know as Elastic Net, is a regularization technique that combines the strengths of Lasso (L1) and Ridge (L2) regression methods. It is particularly useful for handling datasets with high-dimensional features and multicollinearity (correlated features). By blending the penalties of L1 and L2, Elastic Net provides a balance between feature selection (Lasso) and coefficient shrinkage (Ridge).

### Stacking

The **stacking** function in the stepAIC module provides output of a fitted regression model and summary statistics using the sklearn StackingRegressor function for ensemble modeling with any combination of sklearn base regressors which can be turned 'on' or 'off' with the following optional keyword arguments:

- lasso= 'on' (default) or 'off'        uses LassoCV
- ridge= 'on' (default) or 'off'        uses RidgeCV
- elastic= 'on' (default) or 'off'      uses ElasticNetCV
- sgd= 'on' (default) or 'off'          uses SGDRegressor
- knr= 'on' (default) or 'off'          uses KNeighborsRegressor
- svr= 'on' (default) or 'off'          uses SVR(kernel='rbf')
- mlp= 'on' or 'off' (default)          uses MLPRegressor
- gbr= 'on' (default) or 'off'          uses GradientBoostingRegressor
- tree= 'on' (default) or 'off'         uses DecisionTreeRegressor
- forest= 'on' (default) or 'off'       uses RandomForestRegressor

The meta-model may be specifed using the optional keyword argument meta:

- meta= 'linear', 'lasso', 'ridge' (default), or 'elastic' 

where 'linear' uses LinearRegression, 'lasso' uses LassoCV,  'ridge' uses RidgeCV (default), and 'elastic' uses ElasticNetCV as the meta-model for the final estimator.

Stacking regression is an **ensemble** machine learning technique that improves predictive accuracy by combining multiple base regression models. Instead of selecting a single best model, stacking leverages multiple models to generate a more robust final prediction.

**How Stacking Regression Works**

Base Regressors (Level 0 Models):  
- Several regression models (e.g., Lasso, Ridge, Elastic, DecisionTree, RandomForest, etc.) are trained independently on the dataset.
- Each model learns different aspects of the data.

Meta-Model (Level 1 Model):  
- A separate model (e.g. linear regression, Lasso, or Elastic) is trained to **learn from the outputs of base models**.
- It assigns strength weights to each base model’s predictions, determining which models contribute the most to final accuracy.

Final Prediction: 
- The meta-model makes a final prediction based on the base models' combined outputs.

**Advantages of Stacking Regression**

– Utilizes the strengths of multiple models  
– Helps mitigate overfitting or underfitting 
– Works well for datasets with nonlinear relationships

### Comparison of Stepwise, Lasso, Ridge, and Elastic Net

- Feature selection: Lasso performs explicit feature selection by setting some coefficients to zero, while Ridge shrinks coefficients but retains all predictors. Elastic Net balances the regularization methods of Lasso and Ridge and is able to do feature selection. Stepwise regression also performs feature selection but can be less stable than Lasso. 
- Regularization: Lasso, Ridge, and Elastic Net are regularization techniques that prevent overfitting, but they do so differently. Lasso is more likely to produce sparse models, while Ridge is more likely to shrink coefficients smoothly. Elastic Net balances the capabilities of Lasso and Ridge.
- Computational cost: Stepwise regression can be computationally expensive, especially for large datasets. Lasso, Ridge, and Elastic Net can be solved more efficiently using optimization algorithms. 

### AIC vs BIC

Using AIC as the criterion is the default in the **stepwise** fuction. The user also has the option to use the BIC as the criterion instead. AIC is considered to be a useful critierion in stepwise regression. However, BIC is generally considered to be better than AIC for several reasons:

- Penalty for Complexity: BIC penalizes models more heavily for the number of parameters, making it more conservative and less likely to overfit, especially with larger sample sizes.
- Model Selection: BIC is particularly useful when the sample size is large, as it encourages simpler models that are less likely to capture noise.
- Model Recovery: Studies suggest that BIC tends to recover the true model more effectively than AIC, particularly in scenarios where the sample size is large.
While both criteria are useful for model selection, BIC is often preferred for its stricter criteria, which helps in avoiding overfitting and improving model interpretability

### Limitations of Ridge regression for feature selection and the utility of AIC and BIC

Unlike Lasso regression, Ridge regression does not have zeroing of selected coefficients as a goal. Therefore, Ridge regression generally does not select a subset of features for the final best model. Instead, Ridge regression retains all of the candiate features in the input data set and has the goal of minimizing the coefficient values as a strategy to reduce the variance inflation factors to mitigate the effects of multicollinearity.

AIC and BIC have limited value in optimizing Ridge regression. The AIC and BIC in Ridge regression is not sensitive to the alpha parameter because the AIC and BIC values are strongly affected by the number of model parameters. As the alpha parameter is adjusted, the AIC and BIC values change by a relatively small amount depending on the variance of the residuals at each value of alpha. This means that the AIC and BIC values across a wide range of alpha values do not penalize the model for having too many parameters in Ridge regression. Using AIC and BIC have the effect of choosing the lowest value of alpha, which is similar to performing ordinary linear regression without regularaization and with no mitigation of multicollinearity.

If feature selection is the goal of the analysis, then Stepwise or Lasso regression methods are generally better than Ridge regression for that purpose. If your analysis requires that all candidate features are retained in the final model, then Ridge regression is ideal for that purpose using the **ridge** results for RidgCV.

### Acceptable VIF as the target for Ridge regression

Ridge regression reduces the Variance Inflation Factors of the features by adding a penalty term to the ordinary least squares regression. The magnitude of the penalty term is related to the regularization paramter (alpha) and the sum of the squared coefficients. At very low values of alpha there is negligible penalty and the Ridge regression results are practically the same as OLS. As the alpha value is increased, the penalty increases and the VIF values of the features decreases, which decreases the magnitude of the coefficients to mitigates the problem of multicollinearity.

Cross-validated ridge regression (e.g. using RidgeCV) does not always result in acceptable multicollinearity as indicated by VIF. While cross-validation helps in fine-tuning the regression coefficients, it does not always result in VIF values close to 1. Ideally the VIF of all features should be as close as possibe to VIF=1. This can be achieved using a trial and error method of evaluating the VIF values of the model features over a range of alpha values. 

The **ridge** function in stepAIC includes an algorithm (RidgeVIF) to find the model with the optimum value of alpha that will result in VIF values as close as possible to a user-specified target VIF (default target VIF=1.0). This assures that there will be acceptable multicollinearity for all features. The trade-off is that this algorithm reduces the model coefficients such that the target VIF will be achieved. The user has the option to specify any target for VIF to explore the balance between VIF and coefficient values.  


## Installation for Python or Jupyter Notebook

The stepAIC module require that you have already installed numpy, pandas, scikit-learn, tabulate, matplotlib, and statsmodels packages. 

If you have not already installed stepAIC, enter the following with pip or !pip in your notebook or terminal:<br>
```
pip install git+https://github.com/gjpelletier/stepAIC.git
```

if you are upgrading from a previous installation of stepAIC, enter the following with pip pr !pip in your notebook or terminal:<br>
```
pip install git+https://github.com/gjpelletier/stepAIC.git --upgrade
```

Next import the **stepwise**, **lasso**, **ridge**, and **elastic** functions from the stepAIC module as follows in your notebook or python code:<br>
```
from stepAIC import stepwise, lasso, ridge, elastic
```

## Example 1. Use Lasso regression to analyze diabetes data

In this example we will use Lasso regression to analyze the diabetes data available from sklearn. The lasso function uses the sklearn.processing StandardScaler to standardize the X values by default. Then the lasso function uses the standardized X values to find each of the best fit models using LassoCV, LassoLarsCV, LassoLarsIC using AIC, and LassoLarsIC using BIC.

Run the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Use the lasso function in the stepAIC module
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

# Use the stepwise function in the stepAIC module
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
Date:                Tue, 27 May 2025   Prob (F-statistic):           4.75e-65
Time:                        17:53:07   Log-Likelihood:                -2390.1
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

In this example we will use Ridge regression to analyze the diabetes data available from sklearn. The ridge function uses the sklearn.processing StandardScaler to standardize the X values by default. Then the ridge function uses the standardized X values to find each of the best fit models using RidgeCV, and Ridge using AIC, BIC, and VIF for optimization.

Run the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Use the ridge function in the stepAIC module
from stepAIC import ridge
model_objects, model_outputs = ridge(X, y)
```

Running the code above produces the following output display of the best fit model:
```
Ridge regression statistics of best models in model_outputs['stats']:


| Statistic          |         RidgeCV |        RidgeAIC |        RidgeBIC |        RidgeVIF |
|:-------------------|----------------:|----------------:|----------------:|----------------:|
| alpha              |     1.87382     |     0.001       |     0.001       |    53.367       |
| r-squared          |     0.51733     |     0.517748    |     0.517748    |     0.511683    |
| adjusted r-squared |     0.504982    |     0.505412    |     0.505412    |     0.499191    |
| nobs               |   442           |   442           |   442           |   442           |
| df residuals       |   431           |   431           |   431           |   431           |
| df model           |    10           |    10           |    10           |    10           |
| F-statistic        |    46.1949      |    46.2724      |    46.2724      |    45.1624      |
| Prob (F-statistic) |     1.11022e-16 |     1.11022e-16 |     1.11022e-16 |     1.11022e-16 |
| RMSE               |    53.4993      |    53.4761      |    53.4761      |    53.8114      |
| Log-Likelihood     | -2386.18        | -2385.99        | -2385.99        | -2388.76        |
| AIC                |  4794.37        |  4793.99        |  4793.99        |  4799.51        |
| BIC                |  4839.37        |  4838.99        |  4838.99        |  4844.51        |


Coefficients of best models in model_outputs['popt']:


| Feature   |    RidgeCV |   RidgeAIC |   RidgeBIC |   RidgeVIF |
|:----------|-----------:|-----------:|-----------:|-----------:|
| const     | 152.133    | 152.133    | 152.133    | 152.133    |
| age       |  -0.402061 |  -0.476067 |  -0.476067 |   0.132284 |
| sex       | -11.2811   | -11.4068   | -11.4068   |  -9.59744  |
| bmi       |  24.7848   |  24.7266   |  24.7266   |  22.9527   |
| bp        |  15.3345   |  15.4293   |  15.4293   |  14.173    |
| s1        | -25.6503   | -37.6704   | -37.6704   |  -3.44889  |
| s2        |  13.1337   |  22.6685   |  22.6685   |  -3.5854   |
| s3        |  -0.47979  |   4.8019   |   4.8019   |  -9.04181  |
| s4        |   7.01112  |   8.42088  |   8.42088  |   5.55365  |
| s5        |  31.1369   |  35.7308   |  35.7308   |  20.5852   |
| s6        |   3.30159  |   3.21673  |   3.21673  |   4.25116  |


Variance Inflation Factors model_outputs['vif']:
Note: VIF>5 indicates excessive collinearity


| Feature   |   RidgeCV |   RidgeAIC |   RidgeBIC |   RidgeVIF |
|:----------|----------:|-----------:|-----------:|-----------:|
| age       |   1.20293 |    1.2173  |    1.2173  |   0.904903 |
| sex       |   1.26072 |    1.27806 |    1.27806 |   0.918176 |
| bmi       |   1.48134 |    1.50942 |    1.50942 |   1.02315  |
| bp        |   1.43679 |    1.45942 |    1.45942 |   1.00172  |
| s1        |  26.672   |   59.1714  |   59.1714  |   0.532883 |
| s2        |  18.5444  |   39.1737  |   39.1737  |   0.7917   |
| s3        |   8.59212 |   15.3958  |   15.3958  |   1.00853  |
| s4        |   7.59308 |    8.89004 |    8.89004 |   1.40875  |
| s5        |   5.49131 |   10.0716  |   10.0716  |   1.06848  |
| s6        |   1.46335 |    1.48461 |    1.48461 |   1.0328   |
```

The VIF results using RidgeCV show substantially reduced multicollinearity (2 features with VIF>10) compared with results of Ridge_AIC and Ridge_BIC (4 features with VIF>10). However, the multicollinearity for RidgeCV, RidgeAIC, and RidgeBIC is excessive. Using AIC and BIC to optimize the ridge regression is especially problematic. This is because AIC and BIC have limited value in optimizing Ridge regression. The lowest AIC and BIC values occur at the lowest vales of alpha, which is similar to performing ordinary linear regression with no regularization and no reduction in VIF. The AIC and BIC in Ridge regression is not sensitive to the alpha parameter because the AIC and BIC values are strongly affected by the number of model parameters. As the alpha parameter is adjusted, the AIC and BIC values change by a relatively small amount depending on the variance of the residuals at each value of alpha. This means that the AIC and BIC values across a wide range of alpha values do not penalize the model for having too many parameters in Ridge regression.

The RidgeVIF method is able to find a model where all features have acceptable VIF<5, with all VIF values as close as possible to the target VIF (default target VIF=1.0). The model skill for RidgeVIF is similar to the other methods.

## Example 4. Use Elastic Net regression to analyze diabetes data

In this example we will use Elastic Net regression to analyze the diabetes data available from sklearn. The elastic function uses the sklearn.processing StandardScaler to standardize the X values by default. Then the elastic function uses the standardized X values to find the best fit model using ElasticNetCV, and Ridge using MSE as the scoring criterion.

Run the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Use the elastic function in the stepAIC module
from stepAIC import elastic
model_objects, model_outputs = elastic(X, y)
```

Running the code above produces the following output display of the best fit model:

```
ElasticNetCV regression statistics of best model in model_outputs['stats']:


| Statistic          |    ElasticNetCV |
|:-------------------|----------------:|
| alpha              |     1.11865     |
| r-squared          |     0.512957    |
| adjusted r-squared |     0.503959    |
| nobs               |   442           |
| df residuals       |   434           |
| df model           |     7           |
| F-statistic        |    65.2989      |
| Prob (F-statistic) |     1.11022e-16 |
| RMSE               |    53.7411      |
| Log-Likelihood     | -2388.18        |
| AIC                |  4792.36        |
| BIC                |  4825.09        |
| L1-ratio           |     1           |


Coefficients of best models in model_outputs['popt']:


| Feature   |   ElasticNetCV |
|:----------|---------------:|
| const     |      152.133   |
| age       |       -0       |
| sex       |       -9.11162 |
| bmi       |       24.8066  |
| bp        |       13.9806  |
| s1        |       -4.58713 |
| s2        |       -0       |
| s3        |      -10.5553  |
| s4        |        0       |
| s5        |       24.2697  |
| s6        |        2.45869 |
```

Note that for the data in the example, the best fit model from ElasticNetCV was found using L1-ratio=1, which is equivalent to Lasso regression using LassoCV in the lasso function of stepAIC.

## Example 5. Use Stacking regression to analyze diabetes data

In this example we will use Stacking regression to analyze the diabetes data available from sklearn. The **stacking** function uses the sklearn StackingRegressor with an ensemble of models. The **stacking** function standardizes the X values by default. 

Run the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Use the stacking function in the stepAIC module
from stepAIC import stacking
model_objects, model_outputs = stacking(X, y)
```

Running the code above produces the following output display of the best fit model:
```
StackingRegressor statistics of fitted ensemble model in model_outputs['stats']:


| Statistic          |   StackingRegressor |
|:-------------------|--------------------:|
| r-squared          |         0.572372    |
| adjusted r-squared |         0.56245     |
| n_samples          |       442           |
| df residuals       |       432           |
| df model           |         9           |
| F-statistic        |        64.2471      |
| Prob (F-statistic) |         1.11022e-16 |
| RMSE               |        50.3566      |
| Log-Likelihood     |     -2359.43        |
| AIC                |      4738.85        |
| BIC                |      4779.77        |


Meta-model coefficients of base_regressors in model_outputs['meta_params']:


- positive intercept suggests base models under-predict target
- negative intercept suggests base models over-predict target
- positive coefficients have high importance
- coefficients near zero have low importance
- negative coefficients have counteracting importance


| Coefficient               |   StackingRegressor |
|:--------------------------|--------------------:|
| Intercept                 |         -79.4182    |
| LassoCV                   |           1.66162   |
| RidgeCV                   |           0.0481335 |
| ElasticNetCV              |          -1.83486   |
| SGDRegressor              |           0.731691  |
| KNeighborsRegressor       |           0.112592  |
| SVR                       |           0.762235  |
| GradientBoostingRegressor |           0.0743225 |
| DecisionTreeRegressor     |          -0.0582058 |
| RandomForestRegressor     |           0.096149  |
```

![StackingRegressor_residuals](https://github.com/user-attachments/assets/4bba57ff-142f-4585-9d41-dcdcbc36f012)

Note that the stacking regression was able to find a final model with greater skill than the methods used in Examples 1-4.

