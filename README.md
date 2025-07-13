
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
