
# stepAIC - Python functions for stepwise regression to minimize AIC or BIC and eliminate non-significant predictors

The stepAIC and stepBIC functions use an algorithm to find the optimum set of predictor variables that minimizes the Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC) in a regression model. The stepAIC and stepBIC algorothims have the option to use either a forward (default) or backward stepwise search for the optimum set of predictor variables. The forward direction (default) starts with no predictors and adds predictors as long as it improves the model (reduces AIC or BIC). The optional backward direction starts with all predictors and removes predictors as long as it improves the model (reduces AIC or BIC). The stepAIC algorithm also has the option (default) to remove any non-signficant predictors (p-values below a user-specified p_threshold with default p_threshold=0.05) after either a forward or backward search. 

BIC is generally considered better than AIC for evaluation of stepwise regression for several reasons:

- Penalty for Complexity: BIC penalizes models more heavily for the number of parameters, making it more conservative and less likely to overfit, especially with larger sample sizes.
- Model Selection: BIC is particularly useful when the sample size is large, as it encourages simpler models that are less likely to capture noise.
- Model Recovery: Studies suggest that BIC tends to recover the true model more effectively than AIC, particularly in scenarios where the sample size is large.
While both criteria are useful for model selection, BIC is often preferred for its stricter criteria, which helps in avoiding overfitting and improving model interpretability

The stepAIC and stepBIC functions require that you have already installed numpy, pandas, and statsmodels packages. We also recommend that you have installed seaborn and matplotlib.

# Installation for Python or Jupyter Notebook

If you have not already installed PyOAE, enter the following with pip or !pip in your notebook or terminal:<br>
```
pip install git+https://github.com/gjpelletier/stepAIC.git
```

if you are upgrading from a previous installation of PyOAE, enter the following with pip pr !pip in your notebook or terminal:<br>
```
pip install git+https://github.com/gjpelletier/stepAIC.git --upgrade
```

Next import the stepAIC and stepBIC functions as follows in your notebook or python code:<br>
```
from stepAIC import stepAIC, stepBIC
```

# Example

The [stepAIC_example_concrete.ipynb](https://github.com/gjpelletier/stepAIC/blob/main/stepAIC_example_concrete.ipynb) Jupyter notebook presents an example of the use of stepAIC and stepBIC.

