
# stepAIC - Python function for stepwise regression to minimize either AIC or BIC and eliminate non-significant predictors

The stepAIC function uses an algorithm to find the optimum set of predictor variables that minimizes either the Akaike Information Criterion (AIC, default) or Bayesian Information Criterion (BIC, optional) in a regression model. 

The stepAIC algorothim has the option to use either a forward selection (default), backward selection, or all combinations for the optimum set of predictor variables as follows:

- Forward selection (default) starts with no predictors and adds predictors as long as it improves the model (reduces AIC or BIC) 
- Backward selection starts with all predictors and removes predictors as long as it improves the model (reduces AIC or BIC)
- All combinations evaluates all possible combinations of predictors to find the best of all possible models (requires 20 or fewer candiate predictors)

The stepAIC algorithm also has the option (default) to remove any non-signficant predictors (p-values below a user-specified p_threshold with default p_threshold=0.05) after either a forward or backward search. 

Using AIC as the criterion is the default in the stepAIC fuction. The user also has the option to use the BIC as the criterion instead. AIC is considered to be a useful critierion in stepwise regression. However, BIC is generally considered to be better than AIC for several reasons:

- Penalty for Complexity: BIC penalizes models more heavily for the number of parameters, making it more conservative and less likely to overfit, especially with larger sample sizes.
- Model Selection: BIC is particularly useful when the sample size is large, as it encourages simpler models that are less likely to capture noise.
- Model Recovery: Studies suggest that BIC tends to recover the true model more effectively than AIC, particularly in scenarios where the sample size is large.
While both criteria are useful for model selection, BIC is often preferred for its stricter criteria, which helps in avoiding overfitting and improving model interpretability

The stepAIC function requires that you have already installed numpy, pandas, statsmodels, itertools packages. We also recommend that you have installed seaborn and matplotlib.

# Installation for Python or Jupyter Notebook

If you have not already installed PyOAE, enter the following with pip or !pip in your notebook or terminal:<br>
```
pip install git+https://github.com/gjpelletier/stepAIC.git
```

if you are upgrading from a previous installation of PyOAE, enter the following with pip pr !pip in your notebook or terminal:<br>
```
pip install git+https://github.com/gjpelletier/stepAIC.git --upgrade
```

Next import the stepAIC function as follows in your notebook or python code:<br>
```
from stepAIC import stepAIC
```

# Example

The [stepAIC_example_concrete.ipynb](https://github.com/gjpelletier/stepAIC/blob/main/stepAIC_example_concrete.ipynb) Jupyter notebook presents examples of the use of stepAIC with AIC and BIC as the criterion with forward and backward stepping.

