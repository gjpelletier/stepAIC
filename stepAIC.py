# -*- coding: utf-8 -*-

__version__ = "1.0.30"

def stepwise_AIC_BIC(X, y, **kwargs):

    """
    Python function for stepwise linear regression to minimize AIC or BIC
    and eliminate non-signficant predictors

    by
    Greg Pelletier
    gjpelletier@gmail.com
    09-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = pandas dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = pandas dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        criterion= 'AIC' (default) or 'BIC' where
            'AIC': use the Akaike Information Criterion to score the model
            'BIC': use the Bayesian Information Criterion to score the model
        verbose= 'on' (default) or 'off' where
            'on': provide model summary at each step
            'off': provide model summary for only the final selected model
        direction= 'forward' (default), 'backward', or 'all' where
            'forward' (default): 
                1) Start with no predictors in the model
                2) Add the predictor that results in the lowest AIC
                3) Keep adding predictors as long as it reduces AIC
            'backward':
                1) Fit a model with all predictors.
                2) Remove the predictor that results in the lowest AIC
                3) Keep removing predictors as long as it reduces AIC
            'all': find the best model of all possibe combinations of predictors
                Note: 'all' requires no more than 20 columns in X
        drop_insig= 'on' (default) or 'off'
            'on': drop predictors with p-values below threshold p-value (default) 
            'off': keep all predictors regardless of p-value
        p_threshold= threshold p-value to eliminate predictors (default 0.05)                

    RETURNS
        selected_features, model
            selected_features are the final selected features
            model is the final model returned by statsmodels.api OLS

    NOTE
    Do any necessary/optional cleaning and standardizing of data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique 
    column names for each column.

    EXAMPLE 1 - use the default AIC as the criterion with forward stepping:
    best_features, best_model = stepwise_AIC_BIC(X, y)

    EXAMPLE 2 - use the option of BIC as the criterion with forward stepping:
    best_features, best_model = stepwise_AIC_BIC(X, y, criterion='BIC')

    EXAMPLE 3 - use the option of BIC as the criterion with backward stepping:
    best_features, best_model = stepwise_AIC_BIC(X, y, criterion='BIC', direction='backward')

    EXAMPLE 4 - use the option of BIC as the criterion and search all possible models:
    best_features, best_model = stepwise_AIC_BIC(X, y, criterion='BIC', direction='all')

    """

    import statsmodels.api as sm
    from itertools import combinations
    import pandas as pd
    import numpy as np
    import sys
    
    # Define default values of input data arguments
    defaults = {
        'criterion': 'AIC',
        'verbose': 'on',
        'direction': 'forward',
        'drop_insig': 'on',
        'p_threshold': 0.05
        }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}
    p_threshold = data['p_threshold']
    if data['criterion'] == 'aic':
        data['criterion'] = 'AIC'
    if data['criterion'] == 'bic':
        data['criterion'] = 'BIC'
    if data['criterion'] == 'AIC':
        crit = 'AIC'
    elif data['criterion'] == 'BIC':
        crit = 'BIC'

    # check for input errors
    ctrl = isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame)
    if not ctrl:
        print('Check X and y: they need to be pandas dataframes!','\n')
        sys.exit()
    ctrl = (X.index == y.index).all()
    if not ctrl:
        print('Check X and y: they need to have the same index values!','\n')
        sys.exit()
    ctrl = np.isreal(X).all() and X.isna().sum().sum()==0 and X.ndim==2
    if not ctrl:
        print('Check X: it needs be a 2-D dataframe of real numbers with no nan values!','\n')
        sys.exit()
    ctrl = np.isreal(y).all() and y.isna().sum().sum()==0 and y.ndim==1
    if not ctrl:
        print('Check X: it needs be a 1-D dataframe of real numbers with no nan values!','\n')
        sys.exit()
    ctrl = X.shape[0] == y.shape[0]
    if not ctrl:
        print('Check X and y: X and y need to have the same number of rows!','\n')
        sys.exit()
    ctrl = X.columns.is_unique
    if not ctrl:
        print('Check X: X needs to have unique column names for every column!','\n')
        sys.exit()
    if data['direction'] == 'all':
        ctrl = X.shape[1]<=20
        if not ctrl:
            print('X needs to have <= 20 columns to use all directions! Try forward or backward stepping instead!','\n')
            sys.exit()
        
    if data['direction'] == 'forward':
    
        # Forward selection to minimize AIC or BIC
        selected_features = []
        remaining_features = list(X.columns)
        best_score = float('inf')
        istep = 0
        while remaining_features:
            score_with_candidates = []        
            for candidate in remaining_features:
                model = sm.OLS(y, sm.add_constant(X[selected_features + [candidate]])).fit()
                if data['criterion'] == 'AIC':
                    score_with_candidates.append((model.aic, candidate))
                elif data['criterion'] == 'BIC':
                    score_with_candidates.append((model.bic, candidate))
            score_with_candidates.sort()  # Sort by criterion
            best_new_score, best_candidate = score_with_candidates[0]        
            if best_new_score < best_score:
                best_score = best_new_score
                selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
                istep += 1
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                if data['criterion'] == 'AIC':
                    score = model.aic
                elif data['criterion'] == 'BIC':
                    score = model.bic
                if (data['verbose'] == 'on' or
                        (remaining_features == [] and data['drop_insig'] == 'off')):
                    print('\nFORWARD STEP '+str(istep)+", "+crit+"= {:.2f}".format(score))
                    print('Features added: ', selected_features,'\n')
                    print(model.summary())        
            else:            
                remaining_features.remove(best_candidate)
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                if (data['verbose'] == 'on' or 
                        (data['verbose'] == 'off' and data['drop_insig'] == 'off')):
                    print('\nFINAL FORWARD MODEL BEFORE REMOVING INSIGNIFICANT PREDICTORS')
                    print('Best features: ', selected_features,'\n')
                    print(model.summary())
                break            

        if data['drop_insig'] == 'on':
    
            # Backward elimination of features with p < p_threshold
            while selected_features:
    
                # Backward elimination of non-signficant predictors
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                p_values = model.pvalues.iloc[1:]  # Ignore intercept
                max_p_value = p_values.max()
        
                if max_p_value > p_threshold:
                    worst_feature = p_values.idxmax()
                    selected_features.remove(worst_feature)
                else:
                    print('\nFINAL FORWARD MODEL AFTER REMOVING INSIGNIFICANT PREDICTORS')
                    print('Best features: ', selected_features,'\n')
                    print(model.summary())
                    break
    
    if data['direction'] == 'backward':

        # Backward selection to minimize AIC or BIC
        selected_features = list(X.columns)
        remaining_features = []
        istep = 0
        # while remaining_features:
        while len(selected_features) > 0:
            score_with_candidates = []        
            model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
            if data['criterion'] == 'AIC':
                best_score = model.aic
            elif data['criterion'] == 'BIC':
                best_score = model.bic
            # for candidate in remaining_features:
            for candidate in selected_features:
                # model = sm.OLS(y, sm.add_constant(X[selected_features - [candidate]])).fit()
                test_features = selected_features.copy()
                test_features.remove(candidate)
                model = sm.OLS(y, sm.add_constant(X[test_features])).fit()
                if data['criterion'] == 'AIC':
                    score_with_candidates.append((model.aic, candidate))
                elif data['criterion'] == 'BIC':
                    score_with_candidates.append((model.bic, candidate))
            score_with_candidates.sort()  # Sort by criterion
            best_new_score, best_candidate = score_with_candidates[0]        
            if best_new_score < best_score:
                best_score = best_new_score
                remaining_features.append(best_candidate)
                selected_features.remove(best_candidate)
                istep += 1
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                if data['criterion'] == 'AIC':
                    score = model.aic
                elif data['criterion'] == 'BIC':
                    score = model.bic
                if (data['verbose'] == 'on' or
                        (selected_features == [] and data['drop_insig'] == 'off')):
                    print('\nBACKWARD STEP '+str(istep)+", "+crit+"= {:.2f}".format(score))
                    print('Features added: ', selected_features,'\n')
                    print(model.summary())        
            else:            
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                if (data['verbose'] == 'on' or 
                        (data['verbose'] == 'off' and data['drop_insig'] == 'off')):
                    print('\nFINAL BACKWARD MODEL BEFORE REMOVING INSIGNIFICANT PREDICTORS')
                    print('Best features: ', selected_features,'\n')
                    print(model.summary())
                break            

        if data['drop_insig'] == 'on':
    
            while selected_features:
                # Backward elimination of non-signficant predictors
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                p_values = model.pvalues.iloc[1:]  # Ignore intercept
                max_p_value = p_values.max()
                if max_p_value > p_threshold:
                    worst_feature = p_values.idxmax()
                    selected_features.remove(worst_feature)
                else:
                    model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                    print('\nFINAL BACKWARD MODEL AFTER REMOVING INSIGNIFICANT PREDICTORS')
                    print('Best features: ', selected_features,'\n')
                    print(model.summary())
                    break

    if data['direction'] == 'all':

        # make a list of lists of all possible combinations of features
        list_combinations = []
        for n in range(len(list(X.columns)) + 1):
            list_combinations += list(combinations(list(X.columns), n))

        # loop through all possible combinations and sort by AIC or BIC of each combination
        score_with_candidates = []        
        for i in range(len(list_combinations)):
            selected_features = list(map(str,list_combinations[i]))
            model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
            if data['criterion'] == 'AIC':
                score_with_candidates.append((model.aic, selected_features))
            elif data['criterion'] == 'BIC':
                score_with_candidates.append((model.bic, selected_features))
        score_with_candidates.sort()  # Sort by criterion
        best_score, selected_features = score_with_candidates[0]        
        model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
        if (data['verbose'] == 'on' or 
                (data['verbose'] == 'off' and data['drop_insig'] == 'off')):
            print('\nBEST OF ALL POSSIBLE MODELS BEFORE REMOVING INSIGNIFICANT PREDICTORS')
            print('Best features: ', selected_features,'\n')
            print(model.summary())
 
        if data['drop_insig'] == 'on':
    
            while selected_features:
                # Backward elimination of non-signficant predictors
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                p_values = model.pvalues.iloc[1:]  # Ignore intercept
                max_p_value = p_values.max()
                if max_p_value > p_threshold:
                    worst_feature = p_values.idxmax()
                    selected_features.remove(worst_feature)
                else:
                    model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                    print('\nBEST OF ALL POSSIBLE MODELS AFTER REMOVING INSIGNIFICANT PREDICTORS')
                    print('Best features: ', selected_features,'\n')
                    print(model.summary())
                    break
  
    return selected_features, model


def lasso_linear_regression_stats(X,y,model):

    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.linear_model import LassoLarsIC
    from sklearn.linear_model import LassoCV
    import sys

    """
    Calculate linear regression summary statistics 
    from input and output of models that were fitted with 
    sklearn.linear_model LassoCV or LassoLarsIC

    by
    Greg Pelletier
    gjpelletier@gmail.com
    12-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = pandas dataframe of the observed independent variables 
        that were used to fit the model
    y = pandas dataframe of the observed dependent variable 
        that was used to fit the model
    model = output model object from sklearn.linear_model LassoCV or LassoLarsIC
    """

    # check for input errors
    ctrl = isinstance(X, pd.DataFrame)
    if not ctrl:
        print('Check X: it needs to be pandas dataframes!','\n')
        sys.exit()
    ctrl = (X.index == y.index).all()
    if not ctrl:
        print('Check X and y: they need to have the same index values!','\n')
        sys.exit()
    ctrl = np.isreal(X).all() and X.isna().sum().sum()==0 and X.ndim==2
    if not ctrl:
        print('Check X: it needs be a 2-D dataframe of real numbers with no nan values!','\n')
        sys.exit()
    ctrl = np.isreal(y).all() and y.isna().sum().sum()==0 and y.ndim==1
    if not ctrl:
        print('Check X: it needs be a 1-D dataframe of real numbers with no nan values!','\n')
        sys.exit()
    ctrl = X.shape[0] == y.shape[0]
    if not ctrl:
        print('Check X and y: X and y need to have the same number of rows!','\n')
        sys.exit()
    ctrl = X.columns.is_unique
    if not ctrl:
        print('Check X: X needs to have unique column names for every column!','\n')
        sys.exit()
        
    # Calculate regression summary stats
    y_pred = model.predict(X)                   # best fit of the predicted y values
    residuals = y - y_pred
    nobs = np.size(y)

    # dataframe of model parameters, intercept and coefficients, including zero coefs
    nparam = 1 + model.coef_.size               # number of parameters including intercept
    popt = [['' for i in range(nparam)], np.full(nparam,np.nan)]
    for i in range(nparam):
        if i == 0:
            popt[0][i] = 'constant'
            popt[1][i] = model.intercept_
        else:
            popt[0][i] = X.columns[i-1]
            popt[1][i] = model.coef_[i-1]
    popt = pd.DataFrame(popt).T
    popt.columns = ['name', 'coef']

    nparam = np.count_nonzero(popt['coef'])     # number of non-zero coef (incl intcpt)
    df = nobs - nparam
    SSE = np.sum(residuals ** 2)                # sum of squares (residual error)
    MSE = SSE / df                              # mean square (residual error)
    syx = np.sqrt(MSE)                          # standard error of the estimate
    RMSE = np.sqrt(SSE/nobs)                    # root mean squared error
    SST = np.sum(y **2) - np.sum(y) **2 / nobs  # sum of squares (total)
    SSR = SST - SSE                             # sum of squares (regression model)
    MSR = SSR / (nparam-1)                      # mean square (regression model)
    Fstat = MSR / MSE                           # F statistic
    dfn = nparam - 1                            # df numerator for F-test
    dfd = df                                    # df denomenator for F-test
    pvalue = 1-stats.f.cdf(Fstat, dfn, dfd)     # p-value of F-test
    rsquared = SSR / SST                                    # ordinary r-squared                                                    # ordinary rsquared
    adj_rsquared = 1-(1-rsquared)*(nobs-1)/(nobs-nparam-1)  # adjusted rsquared

    # Calculate Log-Likelihood (LL), AIC, and BIC
    sigma_squared = np.sum(residuals**2) / nobs  # Variance estimate
    sigma = np.sqrt(sigma_squared)
    log_likelihood = -0.5 * nobs * (np.log(2 * np.pi) + np.log(sigma_squared) + 1)
    aic = -2 * log_likelihood + 2 * nparam
    bic = -2 * log_likelihood + nparam * np.log(nobs)

    # Put residuals and y_pred into pandas dataframes to preserve the index of X and y
    df_y_pred = pd.DataFrame(y_pred)
    df_y_pred.index = y.index
    df_y_pred.columns = ['y_pred']    
    df_y_pred = df_y_pred['y_pred']
    df_residuals = pd.DataFrame(residuals)
    df_residuals.index = y.index
    df_residuals.columns = ['residuals']    
    df_residuals = df_residuals['residuals']
        
    # put the results into a dictionary
    result = {
            'X': X,
            'y': y,
            'y_pred': df_y_pred,
            'residuals': df_residuals,
            'model': model,
            'popt': popt,
            'nobs': nobs,
            'nparam': nparam,
            'df': df,
            'SST': SST,
            'SSR': SSR,
            'SSE': SSE,
            'MSR': MSR,
            'MSE': MSE,
            'syx': syx,
            'RMSE': RMSE,
            'Fstat': Fstat,
            'dfn': dfn,
            'dfd': dfd,
            'pvalue': pvalue,
            'rsquared': rsquared,
            'adj_rsquared': adj_rsquared,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic        
            }

    return result

def Lasso_CV_AIC_BIC(X, y, **kwargs):

    """
    Python function for Lasso linear regression 
    using k-fold cross-validation (CV) or to minimize AIC or BIC

    by
    Greg Pelletier
    gjpelletier@gmail.com
    09-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = pandas dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = pandas dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        nfolds= number of folds to use for cross-validation (CV)
            with k-fold LassoCV or LassoLarsCV (default nfolds=20)
        standardize= 'on' (default) or 'off' where
            'on': standardize X using sklearn.preprocessing StandardScaler
            'off': do not standardize X
        verbose= 'on' (default) or 'off' where
            'on': display model summary on screen 
            'off': turn off display of model summary on screen

    RETURNS
        model_objects, model_outputs
            model_objects are the output objects from 
                sklearn.linear_model LassoCV, LassoLarsCV, and LassoLarsIC
                of the final best models using the following four methods: 
                - LassoCV: k-fold CV coordinate descent
                - LassoLarsCV: k-fold CV least angle regression
                - LassoLarsAIC: LassoLarsIC using AIC
                - LassoLarsBIC: LasspLarsIC using BIC
            model_outputs is a dictionary of the following outputs 
                from the four Lasso linear regression model methods:
                - 'scaler': sklearn.preprocessing StandardScaler for X
                - 'standardize': 'on' scaler was used for X, 'off' scaler not used
                - 'alpha_vs_coef': model coefficients for each X variable
                    as a function of alpha using Lasso
                - 'alpha_vs_AIC_BIC': AIC and BIC as a function of alpha 
                    using LassoLarsIC
                - 'stats': Regression statistics for each model
                - 'popt': Constant (intercept) and coefficients for the 
                    best fit models from each of the four methods
                - 'y_pred': Predicted y values for each of the four methods
                - 'residuals': Residuals (y-y_pred) for each of the four methods

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = Lasso_CV_AIC_BIC(X, y)

    """

    from stepAIC import lasso_linear_regression_stats
    import time
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LassoCV
    from sklearn.linear_model import LassoLarsCV
    from sklearn.linear_model import LassoLarsIC
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
   
    # Define default values of input data arguments
    defaults = {
        'nfolds': 20,
        'standardize': 'on',
        'verbose': 'on'
        }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # check for input errors
    ctrl = isinstance(X, pd.DataFrame)
    if not ctrl:
        print('Check X: it needs to be pandas dataframes!','\n')
        sys.exit()
    ctrl = (X.index == y.index).all()
    if not ctrl:
        print('Check X and y: they need to have the same index values!','\n')
        sys.exit()
    ctrl = np.isreal(X).all() and X.isna().sum().sum()==0 and X.ndim==2
    if not ctrl:
        print('Check X: it needs be a 2-D dataframe of real numbers with no nan values!','\n')
        sys.exit()
    ctrl = np.isreal(y).all() and y.isna().sum().sum()==0 and y.ndim==1
    if not ctrl:
        print('Check X: it needs be a 1-D dataframe of real numbers with no nan values!','\n')
        sys.exit()
    ctrl = X.shape[0] == y.shape[0]
    if not ctrl:
        print('Check X and y: X and y need to have the same number of rows!','\n')
        sys.exit()
    ctrl = X.columns.is_unique
    if not ctrl:
        print('Check X: X needs to have unique column names for every column!','\n')
        sys.exit()

    # Suppress warnings
    warnings.filterwarnings('ignore')
    print('Fitting Lasso regression models, please wait ...')
    print("\n")

    # Set start time for calculating run time
    start_time = time.time()

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}
    # model_outputs['y'] = y  # echo input y
    # model_outputs['X'] = X  # echo input X

    # Standardized X (X_scaled)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    # Convert scaled arrays into pandas dataframes with same column names as X
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    # Copy index from unscaled to scaled dataframes
    X_scaled.index = X.index
    # model_outputs['X_scaled'] = X_scaled                 # standardized X
    model_outputs['scaler'] = scaler                     # scaler used to standardize X
    model_outputs['standardize'] = data['standardize']   # 'on': X_scaled was used to fit, 'off': X was used

    # Specify X_fit to be used for fitting the models 
    if data['standardize'] == 'on':
        X_fit = X_scaled
    elif data['standardize'] == 'off':
        X_fit = X

    # Calculate the role of alpha vs coefficient values
    alphas = 10**np.linspace(-3,3,100)
    lasso = Lasso(max_iter=10000)
    coefs = []
    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(X_fit, y)
        coefs.append(lasso.coef_)
    alpha_vs_coef = pd.DataFrame({
        'alpha': alphas,
        'coef': coefs
        })
    model_outputs['alpha_vs_coef'] = alpha_vs_coef

    # LassoCV k-fold cross validation via coordinate descent
    model_cv = LassoCV(cv=nfolds, random_state=0, max_iter=10000).fit(X_fit, y)
    model_objects['LassoCV'] = model_cv
    alpha_cv = model_cv.alpha_

    # LassoLarsCV k-fold cross validation via least angle regression
    model_lars_cv = LassoLarsCV(cv=nfolds, max_iter=10000).fit(X_fit, y)
    model_objects['LassoLarsCV'] = model_lars_cv
    alpha_lars_cv = model_lars_cv.alpha_

    # LassoLarsIC minimizing AIC
    model_aic = LassoLarsIC(criterion="aic", max_iter=10000).fit(X_fit, y)
    model_objects['LassoLarsAIC'] = model_aic
    alpha_aic = model_aic.alpha_

    # LassoLarsIC minimizing BIC
    model_bic = LassoLarsIC(criterion="bic", max_iter=10000).fit(X_fit, y)
    model_objects['LassoLarsBIC'] = model_bic
    alpha_bic = model_bic.alpha_

    # results of alphas to minimize AIC and BIC
    alpha_vs_AIC_BIC = pd.DataFrame(
        {
            "alpha": model_aic.alphas_,
            "AIC": model_aic.criterion_,
            "BIC": model_bic.criterion_,
        }
        ).set_index("alpha")
    model_outputs['alpha_vs_AIC_BIC'] = alpha_vs_AIC_BIC

    # Lasso Plot the results of lasso coef as function of alpha
    if data['verbose'] == 'on':
        ax = plt.gca()
        ax.plot(alphas, coefs)
        ax.set_xscale('log')
        plt.axis('tight')
        plt.xlabel(r"$\alpha$")
        plt.legend(X_fit.columns)
        plt.ylabel('Coefficients')
        plt.title(r'Lasso regression coefficients as a function of $\alpha$');
        plt.savefig("Lasso_alpha_vs_coef.png", dpi=300)

    # LassoCV Plot the MSE vs alpha for each fold
    if data['verbose'] == 'on':
        lasso = model_cv
        plt.figure()
        plt.semilogx(lasso.alphas_, lasso.mse_path_, linestyle=":")
        plt.plot(
            lasso.alphas_,
            lasso.mse_path_.mean(axis=-1),
            color="black",
            label="Average across the folds",
            linewidth=2,
        )
        plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha: CV estimate")        
        # ymin, ymax = 2300, 3800
        # plt.ylim(ymin, ymax)
        plt.xlabel(r"$\alpha$")
        plt.ylabel("Mean Square Error")
        plt.legend()
        _ = plt.title(
            "LassoCV - Mean Square Error on each fold: coordinate descent"
        )
        plt.savefig("LassoCV_alpha_vs_MSE.png", dpi=300)

    # LassoLarsCV Plot the MSE vs alpha for each fold
    if data['verbose'] == 'on':
        lasso = model_lars_cv
        plt.figure()
        plt.semilogx(lasso.cv_alphas_, lasso.mse_path_, ":")
        plt.semilogx(
            lasso.cv_alphas_,
            lasso.mse_path_.mean(axis=-1),
            color="black",
            label="Average across the folds",
            linewidth=2,
        )
        plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha: CV estimate")

        # plt.ylim(ymin, ymax)
        plt.xlabel(r"$\alpha$")
        plt.ylabel("Mean Square Error")
        plt.legend()
        _ = plt.title(f"LassoLarsCV - Mean Square Error on each fold: LARS")
        plt.savefig("LassoLarsCV_alpha_vs_MSE.png", dpi=300)

    # LassoLarsIC Plot of alphas to minimize AIC and BIC
    if data['verbose'] == 'on':
        results = alpha_vs_AIC_BIC
        ax = results.plot()
        ax.vlines(
            alpha_aic,
            results["AIC"].min(),
            results["AIC"].max(),
            label="alpha: AIC estimate",
            linestyles="--",
            color="tab:blue",
        )
        ax.vlines(
            alpha_bic,
            results["BIC"].min(),
            results["BIC"].max(),
            label="alpha: BIC estimate",
            linestyle="--",
            color="tab:orange",
        )
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Information Criterion (AIC or BIC)")
        ax.set_xscale("log")
        ax.legend()
        _ = ax.set_title(
            "LassoLarsIC - Information Criterion for model selection"
        )
        plt.savefig("LassoLarsIC_alpha_vs_AIC_BIC.png", dpi=300)

    # Calculate regression stats
    stats_cv = lasso_linear_regression_stats(X_fit, y, model_cv)
    stats_lars_cv = lasso_linear_regression_stats(X_fit, y, model_lars_cv)
    stats_aic = lasso_linear_regression_stats(X_fit, y, model_aic)
    stats_bic = lasso_linear_regression_stats(X_fit, y, model_bic)

    # residual plot for training error
    if data['verbose'] == 'on':
        res_cv = stats_cv['residuals']
        res_lars_cv = stats_lars_cv['residuals']
        res_aic = stats_aic['residuals']
        res_bic = stats_bic['residuals']
        rmse_cv = stats_cv['RMSE']
        rmse_lars_cv = stats_lars_cv['RMSE']
        rmse_aic = stats_aic['RMSE']
        rmse_bic = stats_bic['RMSE']
        plt.figure()
        plt.scatter(y, (res_cv), s=30, label=('LassoCV (RMSE={:.2f})'.format(rmse_cv)))
        plt.scatter(y, (res_lars_cv), s=20, label=('LassoLarsCV (RMSE={:.2f})'.format(rmse_lars_cv)))
        plt.scatter(y, (res_aic), s=10, label=('LassoLarsAIC (RMSE={:.2f})'.format(rmse_aic)))
        plt.scatter(y, (res_bic), s=5, label=('LassoLarsBIC (RMSE={:.2f})'.format(rmse_bic)))
        rmse_cv = np.sqrt(np.mean((res_cv)**2))
        plt.hlines(y=0, xmin=min(y), xmax=max(y), color='k')
        plt.title("Residual plot for training error")
        plt.legend();
        plt.xlabel('y')
        plt.ylabel('residual')
        plt.savefig("residuals.png", dpi=300)

    # Find the AIC and BIC of the LassoLarsAIC and LassoLarsBIC models
    min_index_aic = model_outputs['alpha_vs_AIC_BIC']['AIC'].idxmin()
    min_index_bic = model_outputs['alpha_vs_AIC_BIC']['BIC'].idxmin()
    AIC_for_LassoLarsAIC = model_outputs['alpha_vs_AIC_BIC']['AIC'][min_index_aic]
    BIC_for_LassoLarsAIC = model_outputs['alpha_vs_AIC_BIC']['BIC'][min_index_aic]
    AIC_for_LassoLarsBIC = model_outputs['alpha_vs_AIC_BIC']['AIC'][min_index_bic]
    BIC_for_LassoLarsBIC = model_outputs['alpha_vs_AIC_BIC']['BIC'][min_index_bic]

    # Make the model_outputs dataframes
    list1_name = ['alpha','r-squared','adjusted r-squared','nobs','df residuals','df model','F-statistic',
                        'Prob (F-statistic)','Log-Likelihood','AIC','BIC']
    list2_name = list(stats_cv['popt']['name'])
    list3_name = list1_name + list2_name

    list1_cv = [model_cv.alpha_, stats_cv["rsquared"], stats_cv["adj_rsquared"], 
                       stats_cv["nobs"], stats_cv["df"], stats_cv["dfn"], 
                       stats_cv["Fstat"], stats_cv["pvalue"], 
                       stats_cv["log_likelihood"],stats_cv["aic"],stats_cv["bic"]]
    list2_cv = list(stats_cv['popt']['coef'])
    list3_cv = list1_cv + list2_cv

    list1_lars_cv = [model_lars_cv.alpha_, stats_lars_cv["rsquared"], stats_lars_cv["adj_rsquared"], 
                       stats_lars_cv["nobs"], stats_lars_cv["df"], stats_lars_cv["dfn"], 
                       stats_lars_cv["Fstat"], stats_lars_cv["pvalue"], 
                       stats_lars_cv["log_likelihood"],stats_lars_cv["aic"],stats_lars_cv["bic"]]
    list2_lars_cv = list(stats_lars_cv['popt']['coef'])
    list3_lars_cv = list1_lars_cv + list2_lars_cv

    list1_aic = [model_aic.alpha_, stats_aic["rsquared"], stats_aic["adj_rsquared"], 
                       stats_aic["nobs"], stats_aic["df"], stats_aic["dfn"], 
                       stats_aic["Fstat"], stats_aic["pvalue"], 
                       stats_aic["log_likelihood"],AIC_for_LassoLarsAIC,BIC_for_LassoLarsAIC]
    list2_aic = list(stats_aic['popt']['coef'])
    list3_aic = list1_aic + list2_aic

    list1_bic = [model_bic.alpha_, stats_bic["rsquared"], stats_bic["adj_rsquared"], 
                       stats_bic["nobs"], stats_bic["df"], stats_bic["dfn"], 
                       stats_bic["Fstat"], stats_bic["pvalue"], 
                       stats_bic["log_likelihood"],AIC_for_LassoLarsBIC,BIC_for_LassoLarsBIC]
    list2_bic = list(stats_bic['popt']['coef'])
    list3_bic = list1_bic + list2_bic

    summary = pd.DataFrame(
        {
            "Coefficient": list3_name,
            "LassoCV": list3_cv,
            "LassoLarsCV": list3_lars_cv,
            "LassoLarsAIC": list3_aic,
            "LassoLarsBIC": list3_bic
        }
        )
    # model_outputs['summary'] = summary

    stats = pd.DataFrame(
        {
            "Statistic": list1_name,
            "LassoCV": list1_cv,
            "LassoLarsCV": list1_lars_cv,
            "LassoLarsAIC": list1_aic,
            "LassoLarsBIC": list1_bic
        }
        )
    model_outputs['stats'] = stats

    popt = pd.DataFrame(
        {
            "Coefficient": list2_name,
            "LassoCV": list2_cv,
            "LassoLarsCV": list2_lars_cv,
            "LassoLarsAIC": list2_aic,
            "LassoLarsBIC": list2_bic
        }
        )
    model_outputs['popt'] = popt

    y_pred = pd.DataFrame(
        {
            "LassoCV": stats_cv['y_pred'],
            "LassoLarsCV": stats_lars_cv['y_pred'],
            "LassoLarsAIC": stats_aic['y_pred'],
            "LassoLarsBIC": stats_aic['y_pred']
        }
        )
    y_pred.index = y.index
    model_outputs['y_pred'] = y_pred

    residuals = pd.DataFrame(
        {
            "LassoCV": stats_cv['residuals'],
            "LassoLarsCV": stats_lars_cv['residuals'],
            "LassoLarsAIC": stats_aic['residuals'],
            "LassoLarsBIC": stats_aic['residuals']
        }
        )
    residuals.index = y.index
    model_outputs['residuals'] = residuals

    """
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    
    X_vif = X[best_features]
    
    # Add a constant for the intercept term
    X_vif = sm.add_constant(X_vif)
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X_vif.columns
    
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i)
                        for i in range(len(X_vif.columns))]
    """

    # Print model_outputs
    if data['verbose'] == 'on':
        print("Lasso regression statistics of best models in model_outputs['stats']:")
        print("\n")
        print(model_outputs['stats'].to_markdown(index=False))
        print("\n")
        print("Coefficients of best models in model_outputs['popt']:")
        print("\n")
        print(model_outputs['popt'].to_markdown(index=False))
        print("\n")

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return model_objects, model_outputs

