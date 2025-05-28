# -*- coding: utf-8 -*-

__version__ = "1.0.69"

def detect_dummy_variables(df, sep=None):
    """
    Detects dummy variables in a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        sep (str, optional): Separator used in column names if dummy variables were created with pd.get_dummies. Defaults to None.

    Returns:
        bool: True if dummy variables are likely present, False otherwise.
    """
    import pandas as pd

    for col in df.columns:
        if df[col].nunique() == 2 and df[col].isin([0, 1]).all():
            return True

    if sep is not None:
        try:
            pd.from_dummies(df, sep=sep)
            return True
        except ValueError:
            pass

    return False

def nnn(x):

    """
    PURPOSE
    Count the number of non-nan values in the numpy array x
    USAGE
    result = nnn(x)
    INPUT
    x = any numpy array of any dimension
    OUTPUT
    result = number of non-nan values in the array x
    """
    
    import numpy as np

    result = np.count_nonzero(~np.isnan(x))
    
    return result

def stepwise(X, y, **kwargs):

    """
    Python function for stepwise linear regression to minimize AIC or BIC
    and eliminate non-signficant predictors

    by
    Greg Pelletier
    gjpelletier@gmail.com
    17-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        criterion= 'aic' (default) or 'bic' where
            'aic': use the Akaike Information Criterion to score the model
            'bic': use the Bayesian Information Criterion to score the model
            'r2': use the adjusted r-squared to score the model
            'p_coef': use p-values of coefficients to select features
                using p_coef  as criterion automatically uses backward direction
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
            'all': find the best model of all possibe subsets of predictors
                Note: 'all' requires no more than 20 columns in X
        standardize= 'on' or 'off' (default) where
            'on': standardize X using sklearn.preprocessing StandardScaler
            'off': do not standardize X (default)
        drop_insig= 'on' (default) or 'off'
            'on': drop predictors with p-values below threshold p-value (default) 
            'off': keep all predictors regardless of p-value
        p_threshold= threshold p-value to eliminate predictors (default 0.05)                

    RETURNS
        model_object, model_output 
            model_object is the final model returned by statsmodels.api OLS
            model_output is a dictionary of the following outputs:    
            model_outputs is a dictionary of the following outputs 
                from the four Lasso linear regression model methods:
                - 'scaler': sklearn.preprocessing StandardScaler for X
                - 'standardize': 'on' scaler was used for X, 'off' scaler not used
                - 'selected_features' are the final selected features
                - 'step_features' are the features and fitness score at each step
                    (if 'direction'=='forward' or 'direction'=='backward'), 
                    or the best 10 subsets of features (if 'direction'=='all'),
                    including the AIC, BIC, and adjusted r-squared
                - 'y_pred': Predicted y values for each of the four methods
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'popt': Constant (intercept) and coefficients for the 
                    best fit models from each of the four methods
                - 'pcov': Covariance matrix of features 
                - 'vif': Variance Inlfation Factors of selected_features
                - 'stats': Regression statistics for each model
                - 'summary': statsmodels model.summary() of the best fitted model

    NOTE
    Do any necessary/optional cleaning of data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique 
    column names for each column.

    EXAMPLE 1 - use the default AIC as the criterion with forward stepping:
    best_features, best_model = stepwise(X, y)

    EXAMPLE 2 - use the option of BIC as the criterion with forward stepping:
    best_features, best_model = stepwise(X, y, criterion='BIC')

    EXAMPLE 3 - use the option of BIC as the criterion with backward stepping:
    best_features, best_model = stepwise(X, y, criterion='BIC', direction='backward')

    EXAMPLE 4 - use the option of BIC as the criterion and search all possible models:
    best_features, best_model = stepwise(X, y, criterion='BIC', direction='all')

    """

    from stepAIC import detect_dummy_variables
    import statsmodels.api as sm
    from itertools import combinations
    import pandas as pd
    import numpy as np
    import sys
    from sklearn.preprocessing import StandardScaler
    import time
    import matplotlib.pyplot as plt
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import warnings
    
    # Define default values of input data arguments
    defaults = {
        'criterion': 'AIC',
        'verbose': 'on',
        'direction': 'forward',
        'standardize': 'off',
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
    elif data['criterion'] == 'r2':
        crit = 'rsq_adj'
    if data['criterion'] == 'p_coef':
        data['direction'] = 'backward'
    
    # check for input errors
    ctrl = detect_dummy_variables(X)
    if ctrl:
        print('Check X: Stewpise can not handle dummies. Try using lasso if X has dummies.','\n')
        sys.exit()
    ctrl = isinstance(X, pd.DataFrame)
    if not ctrl:
        print('Check X: it needs to be pandas dataframe!','\n')
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

    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Set start time for calculating run time
    if data['direction'] == 'all':
        nsubsets = 2**X.shape[1]
        runtime = (nsubsets / (2**16)) * (120/60)   # guess runtime assuming 120 sec for 16 candidate features
        if X.shape[1] > 15:
            print("Fitting models for all "+str(nsubsets)+
                " subsets of features, this may take about {:.0f} minutes, please wait ...".format(runtime))
        else:
            print("Fitting models for all "+str(nsubsets)+
                " subsets of features, this may take up to a minute, please wait ...")
            
    else:
        print('Fitting Stepwise models, please wait ...')
    if data['verbose'] == 'on':
        print('\n')
    start_time = time.time()
    # model_outputs = {}
    step_features = {}
    
    # Option to use standardized X
    if data['standardize'] == 'on':
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        # Convert scaled arrays into pandas dataframes with same column names as X
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        # Copy index from unscaled to scaled dataframes
        X_scaled.index = X.index
        # Replace X with the standardized X for regression
        X = X.copy()
        X = X_scaled.copy()
        
    if data['direction'] == 'forward':

        # Forward selection to minimize AIC or BIC
        selected_features = []
        remaining_features = list(X.columns)

        # best_score = float('inf')

        istep = 0
        while remaining_features:
            score_with_candidates = []        
            
            # start with only a constant in the model
            if istep == 0:
                X_const = np.ones((len(y), 1))  # column of ones for the intercept
                X_const = pd.DataFrame(X_const,columns=['constant'])
                X_const.index = X.index
                model = sm.OLS(y, X_const).fit()

                # output dataframe of score at each step
                step_features = {'Step': 0, 'AIC': model.aic, 'BIC': model.bic, 
                    'rsq_adj': 0.0, 'Features': [[]]}
                step_features = pd.DataFrame(step_features)
                
                if data['criterion'] == 'AIC':
                    candidate = ['']
                    score_with_candidates.append((model.aic, candidate))
                    best_score = model.aic
                elif data['criterion'] == 'BIC':
                    candidate = ['']
                    score_with_candidates.append((model.bic, candidate))
                    best_score = model.bic
                elif data['criterion'] == 'r2':
                    candidate = ['']
                    score_with_candidates.append((1-model.rsquared_adj, candidate))
                    best_score = 1-model.rsquared_adj
                                       
            for candidate in remaining_features:
                model = sm.OLS(y, sm.add_constant(X[selected_features + [candidate]])).fit()
                if data['criterion'] == 'AIC':
                    score_with_candidates.append((model.aic, candidate))
                elif data['criterion'] == 'BIC':
                    score_with_candidates.append((model.bic, candidate))
                elif data['criterion'] == 'r2':
                    score_with_candidates.append((1-model.rsquared_adj, candidate))
            score_with_candidates.sort()  # Sort by criterion
            best_new_score, best_candidate = score_with_candidates[0]        
            if best_new_score < best_score:
                best_score = best_new_score
                selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
                istep += 1
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

                # add new row to output dataframe
                new_row = {'Step': istep, 'AIC': model.aic, 'BIC': model.bic, 
                           'rsq_adj': model.rsquared_adj, 
                           'Features': np.array(selected_features)}
                step_features = pd.concat([step_features, 
                        pd.DataFrame([new_row])], ignore_index=True)
                
                if data['criterion'] == 'AIC':
                    score = model.aic
                elif data['criterion'] == 'BIC':
                    score = model.bic
                elif data['criterion'] == 'r2':
                    score = model.rsquared_adj
                if (data['verbose'] == 'on' and
                        (remaining_features == [] and data['drop_insig'] == 'off')):
                    print("Model skill and features at each step in model_outputs['step_features']:\n")
                    print(step_features.to_markdown(index=False))
                    print('\nForward step '+str(istep)+", "+crit+"= {:.2f}".format(score))
                    print('Features added: ', selected_features,'\n')
                    print(model.summary())        

            else:            
                remaining_features.remove(best_candidate)
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                                
                if (data['verbose'] == 'on' and 
                        (remaining_features != [] and data['drop_insig'] == 'off')):
                    print("Model skill and features at each step in model_outputs['step_features']:\n")
                    print(step_features.to_markdown(index=False))
                    print('\nFinal forward model before removing insignficant features if any:')
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
                    
                    # add new row to output dataframe
                    new_row = {'Step': istep+1, 'AIC': model.aic, 'BIC': model.bic, 
                               'rsq_adj': model.rsquared_adj, 
                               'Features': np.array(selected_features)}
                    step_features = pd.concat([step_features, 
                            pd.DataFrame([new_row])], ignore_index=True)

                    if data['verbose'] == 'on':
                        print("Model skill and features at each step in model_outputs['step_features']:\n")
                        print(step_features.to_markdown(index=False))
                        print('\nFinal forward model after removing insignficant features if any:')
                        print('Best features: ', selected_features,'\n')
                        print(model.summary())
                    break
    
    if data['direction'] == 'backward' and data['criterion'] != 'p_coef':

        # Backward selection to minimize AIC or BIC
        selected_features = list(X.columns)
        remaining_features = []
        istep = 0
        while len(selected_features) > 0:
            score_with_candidates = []        
            model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

            # start output dataframe of score at each step
            if istep == 0:
                step_features = {'Step': 0, 'AIC': model.aic, 'BIC': model.bic, 
                    'rsq_adj': model.rsquared_adj, 'Features': [np.array(selected_features)]}
                step_features = pd.DataFrame(step_features)
            
            if data['criterion'] == 'AIC':
                best_score = model.aic
            elif data['criterion'] == 'BIC':
                best_score = model.bic
            elif data['criterion'] == 'r2':
                best_score = 1-model.rsquared_adj
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
                elif data['criterion'] == 'r2':
                    score_with_candidates.append((1-model.rsquared_adj, candidate))
            score_with_candidates.sort()  # Sort by criterion
            best_new_score, best_candidate = score_with_candidates[0]        
            if best_new_score < best_score:
                best_score = best_new_score
                remaining_features.append(best_candidate)
                selected_features.remove(best_candidate)
                istep += 1
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

                # add new row to output dataframe
                new_row = {'Step': istep, 'AIC': model.aic, 'BIC': model.bic, 
                           'rsq_adj': model.rsquared_adj, 
                           'Features': np.array(selected_features)}
                step_features = pd.concat([step_features, 
                        pd.DataFrame([new_row])], ignore_index=True)
                            
                if data['criterion'] == 'AIC':
                    score = model.aic
                elif data['criterion'] == 'BIC':
                    score = model.bic
                elif data['criterion'] == 'r2':
                    score = model.rsquared_adj
                if (data['verbose'] == 'on' and
                        (selected_features == [] and data['drop_insig'] == 'off')):
                    print("Model skill and features at each step in model_outputs['step_features']:\n")
                    print(step_features.to_markdown(index=False))
                    print('\nBacksard step '+str(istep)+", "+crit+"= {:.2f}".format(score))
                    print('Features added: ', selected_features,'\n')
                    print(model.summary())        

            else:            
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
               
                if data['verbose'] == 'on' and data['drop_insig'] == 'off':
                    print("Model skill and features at each step in model_outputs['step_features']:\n")
                    print(step_features.to_markdown(index=False))
                    print('\nFinal backward model before removing insignficant features if any:')
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

                    # add new row to output dataframe
                    new_row = {'Step': istep+1, 'AIC': model.aic, 'BIC': model.bic, 
                               'rsq_adj': model.rsquared_adj, 
                               'Features': np.array(selected_features)}
                    step_features = pd.concat([step_features, 
                            pd.DataFrame([new_row])], ignore_index=True)
                    
                    print("Model skill and features at each step in model_outputs['step_features']:\n")
                    # print(model_outputs['step_features'].to_markdown(index=False))
                    print(step_features.to_markdown(index=False))
                    print('\nFinal backward model after removing insignficant features if any:')
                    print('Best features: ', selected_features,'\n')
                    print(model.summary())
                    break

    if data['direction'] == 'backward' and data['criterion'] == 'p_coef':

        # Backward selection to keep only features with p_coef <= p_threshold
        selected_features = list(X.columns)
        remaining_features = []
        istep = 0
        while selected_features:
            # Backward elimination of non-signficant predictors
            model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

            # start output dataframe of score at each step
            if istep == 0:
                step_features = {'Step': 0, 'AIC': model.aic, 'BIC': model.bic, 
                    'rsq_adj': model.rsquared_adj, 'Features': [np.array(selected_features)]}
                step_features = pd.DataFrame(step_features)
                new_row = {'Step': istep, 'AIC': model.aic, 'BIC': model.bic, 
                           'rsq_adj': model.rsquared_adj, 
                           'Features': np.array(selected_features)}
                step_features = pd.concat([step_features, 
                        pd.DataFrame([new_row])], ignore_index=True)
            else:
                new_row = {'Step': istep, 'AIC': model.aic, 'BIC': model.bic, 
                           'rsq_adj': model.rsquared_adj, 
                           'Features': np.array(selected_features)}
                step_features = pd.concat([step_features, 
                        pd.DataFrame([new_row])], ignore_index=True)
                
            p_values = model.pvalues.iloc[1:]  # Ignore intercept
            max_p_value = p_values.max()
            istep += 1
            if max_p_value > p_threshold:
                worst_feature = p_values.idxmax()
                selected_features.remove(worst_feature)
            else:
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

                # add new row to output dataframe
                new_row = {'Step': istep, 'AIC': model.aic, 'BIC': model.bic, 
                           'rsq_adj': model.rsquared_adj, 
                           'Features': np.array(selected_features)}
                step_features = pd.concat([step_features, 
                        pd.DataFrame([new_row])], ignore_index=True)
                
                print("Model skill and features at each step in model_outputs['step_features']:\n")
                print(step_features.to_markdown(index=False))
                print('\nFinal backward model after removing insignficant features if any:')
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

            if i == 0:
                # output dataframe of score at each step
                step_features = {'Rank': i, 'AIC': model.aic, 'BIC': model.bic, 
                    'rsq_adj': model.rsquared_adj, 'Features': selected_features}
                step_features = pd.DataFrame(step_features)
            else:
                # add new row to output dataframe
                new_row = {'Rank': i, 'AIC': model.aic, 'BIC': model.bic, 
                           'rsq_adj': model.rsquared_adj, 
                           'Features': np.array(selected_features)}
                step_features = pd.concat([step_features, 
                        pd.DataFrame([new_row])], ignore_index=True)
                            
            if data['criterion'] == 'AIC':
                score_with_candidates.append((model.aic, selected_features))
            elif data['criterion'] == 'BIC':
                score_with_candidates.append((model.bic, selected_features))
            elif data['criterion'] == 'r2':
                score_with_candidates.append((1-model.rsquared_adj, selected_features))
        score_with_candidates.sort()  # Sort by criterion
        best_score, selected_features = score_with_candidates[0]        
        model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

        if data['drop_insig'] == 'off':            
            # sort step_features by criterion
            if data['criterion'] == 'AIC':
                step_features = step_features.sort_values(by='AIC')
            elif data['criterion'] == 'BIC':
                step_features = step_features.sort_values(by='BIC')            
            elif data['criterion'] == 'r2':
                step_features = step_features.sort_values(by='rsq_adj', ascending=False)            
            ranks = np.arange(0, step_features.shape[0])
            step_features['Rank'] = ranks        
            # save best 10 subsets of features in step_features
            nhead = min(step_features.shape[0],10)
            step_features = step_features.head(nhead)
        
        if data['verbose'] == 'on' and data['drop_insig'] == 'off':            
            print("Best "+str(nhead)+" subsets of features in model_outputs['step_features']:\n")
            print(step_features.head(nhead).to_markdown(index=False))
            print('\nBest of all possible models before removing insignficant features if any:')
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

                    # add new row to output dataframe
                    new_row = {'Rank': i+1, 'AIC': model.aic, 'BIC': model.bic, 
                               'rsq_adj': model.rsquared_adj, 
                               'Features': np.array(selected_features)}
                    step_features = pd.concat([step_features, 
                            pd.DataFrame([new_row])], ignore_index=True)

                    # sort step_features by criterion
                    if data['criterion'] == 'AIC':
                        step_features = step_features.sort_values(by='AIC')
                    elif data['criterion'] == 'BIC':
                        step_features = step_features.sort_values(by='BIC')            
                    elif data['criterion'] == 'r2':
                        step_features = step_features.sort_values(by='rsq_adj', ascending=False)            
                    ranks = np.arange(0, step_features.shape[0])
                    step_features['Rank'] = ranks        
                    # save best 10 subsets of features in step_features
                    nhead = min(step_features.shape[0],10)
                    step_features = step_features.head(nhead)
                    
                    if data['verbose'] == 'on':
                        print("Best "+str(nhead)+" subsets of features in model_outputs['step_features']:\n")
                        print(step_features.head(nhead).to_markdown(index=False))
                        print('\nBest of all possible models after removing insignficant features if any:')
                        print('Best features: ', selected_features,'\n')
                        print(model.summary())
                    break
            
    # Variance Inflation Factors of selected_features
    # Add a constant for the intercept
    X_ = sm.add_constant(X[selected_features])    
    vif = pd.DataFrame()
    vif['Feature'] = X_.columns
    vif["VIF"] = [variance_inflation_factor(X_.values, i)
                        for i in range(len(X_.columns))]
    vif.set_index('Feature',inplace=True)
    if data['verbose'] == 'on':
        print("\nVariance Inflation Factors of selected_features:")
        print("Note: VIF>5 indicates excessive collinearity\n")
        print(vif.to_markdown(index=True))

    # dataframe of model parameters, intercept and coefficients, including zero coefs if any
    nparam = model.params.size               # number of parameters including intercept
    popt = [['' for i in range(nparam)], np.full(nparam,np.nan)]
    for i in range(nparam):
        popt[0][i] = model.model.exog_names[i]
        popt[1][i] = model.params[i]
    popt = pd.DataFrame(popt).T
    popt.columns = ['Feature', 'param']
    popt.set_index('Feature',inplace=True)
    
    model_object = model
    model_output = {}
    scaler = StandardScaler().fit(X)
    model_output['scaler'] = scaler
    model_output['standardize'] = data['standardize']
    model_output['selected_features'] = selected_features
    model_output['step_features'] = step_features
    model_output['y_pred'] = model.predict(sm.add_constant(X[selected_features]))
    model_output['residuals'] = model_output['y_pred'] - y
    model_output['popt'] = popt

    # # Get the covariance matrix of parameters including intercept
    # # results = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
    # # cov_matrix = results.cov_params()
    # cov_matrix = model.cov_params()
    # # Exclude the intercept (assumes the intercept is the first parameter)
    # cov_matrix_excluding_intercept = cov_matrix.iloc[1:, 1:]
    X_ = sm.add_constant(X[selected_features])    # Add a constant for the intercept
    pcov = pd.DataFrame(np.cov(X_, rowvar=False), index=X_.columns)
    pcov.columns = X_.columns
    
    model_output['pcov'] = pcov
    model_output['vif'] = vif

    # Summary statitistics
    list_name = ['r-squared','adjusted r-squared',
        'nobs','df residuals','df model',
        'F-statistic','Prob (F-statistic)',
        'RMSE',
        'Log-Likelihood','AIC','BIC']
    list_stats = [model.rsquared, model.rsquared_adj,
        len(y), model.df_resid, model.df_model, 
        model.fvalue, model.f_pvalue, 
        np.sqrt(np.mean(model_output['residuals']**2)),  
        model.llf,model.aic,model.bic]
    stats = pd.DataFrame(
        {
            "Statistic": list_name,
            "Value": list_stats
        }
        )
    stats.set_index('Statistic',inplace=True)
    model_output['stats'] = stats
    model_output['summary'] = model.summary()
    
    # plot residuals
    if data['verbose'] == 'on':
        y_pred = model_output['y_pred']
        res = model_output['residuals']
        rmse = np.sqrt(np.mean(res**2))
        plt.figure()
        plt.scatter(y_pred, res)
        plt.hlines(y=0, xmin=min(y), xmax=max(y), color='k')
        plt.title("Residual plot for training error, RMSE={:.2f}".format(rmse))
        plt.xlabel('y_pred')
        plt.ylabel('residual')
        plt.savefig("Stepwise_residuals.png", dpi=300)
        
    # Print the run time
    fit_time = time.time() - start_time
    if data['verbose'] == 'on':
        print('\n')
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")

    # Restore warnings to normal
    warnings.filterwarnings("default")
    
    return model_object, model_output


def linear_regression_stats(X,y,model):

    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.linear_model import LassoLarsIC
    from sklearn.linear_model import LassoCV
    import sys

    """
    Calculate linear regression summary statistics 
    from input and output of models that were fitted with 
    sklearn.linear_model Lasso, LassoCV, LassoLarsCV, or LassoLarsIC

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
    model = output model object from sklearn.linear_model 
        LassoCV, LassoLarsCV, or LassoLarsIC
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
    residuals = y_pred - y
    nobs = np.size(y)

    # dataframe of model parameters, intercept and coefficients, including zero coefs
    nparam = 1 + model.coef_.size               # number of parameters including intercept
    popt = [['' for i in range(nparam)], np.full(nparam,np.nan)]
    for i in range(nparam):
        if i == 0:
            popt[0][i] = 'const'
            popt[1][i] = model.intercept_
        else:
            popt[0][i] = X.columns[i-1]
            popt[1][i] = model.coef_[i-1]
    popt = pd.DataFrame(popt).T
    popt.columns = ['Feature', 'param']

    nparam = np.count_nonzero(popt['param'])     # number of non-zero param (incl intcpt)
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

def lasso(X, y, **kwargs):

    """
    Python function for Lasso linear regression 
    using k-fold cross-validation (CV) or to minimize AIC or BIC

    by
    Greg Pelletier
    gjpelletier@gmail.com
    17-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        nfolds= number of folds to use for cross-validation (CV)
            with k-fold LassoCV or LassoLarsCV (default nfolds=20)
        standardize= 'on' (default) or 'off' where
            'on': standardize X using sklearn.preprocessing StandardScaler
            'off': do not standardize X
        alpha_min= minimum value of range of alphas to evaluate (default=1e-3)
        alpha_max= maximum value of range of alphas to evaluate (default=1e3)
        n_alpha= number of log-spaced alphas to evaluate (default=1000)
        verbose= 'on' (default) or 'off' where
            'on': display model summary on screen 
            'off': turn off display of model summary on screen

    Standardization is generally recommended for Lasso regression.

    It is generally recommended to use a largest possible number of folds 
    for LassoCV and LassoLarsCV to ensure more accurate model selection. 
    The only disadvantage of a large number of folds is the increase 
    computational time. The lasso function allows you to specify 
    the number of folds using the nfolds argument. 
    Using a larger number can lead to better performance. 
    For optimal results, consider experimenting 
    with different fold sizes to find the best balance 
    between performance and speed.

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
                - 'y_pred': Predicted y values for each of the four methods
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'popt': Constant (intercept) and coefficients for the 
                    best fit models from each of the four methods
                - 'popt_table': Constant (intercept) and coefficients
                    of best fit of all four methods in one table
                - 'pcov': Covariance matrix of features 
                - 'vif': Variance Inlfation Factors of features of each method
                - 'vif_table': Variance Inlfation Factors of features of 
                    all four methods in one table
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = lasso(X, y)

    """

    from stepAIC import linear_regression_stats, detect_dummy_variables
    import time
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import Lasso
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
        'alpha_min': 1.0e-3,
        'alpha_max': 1.0e3,
        'n_alpha': 1000,
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
    ctrl = data['alpha_min'] > 0 
    if not ctrl:
        print('Check input of alpha_min, it must be greater than zero!','\n')
        sys.exit()
    ctrl = data['alpha_max'] > data['alpha_min'] 
    if not ctrl:
        print('Check input of alpha_max, it must be greater than alpha_min!','\n')
        sys.exit()
    ctrl = data['n_alpha'] > 1 
    if not ctrl:
        print('Check inputs of n_alpha, it must be greater than 1!','\n')
        sys.exit()

    # Suppress warnings
    warnings.filterwarnings('ignore')
    print('Fitting Lasso regression models, please wait ...')
    if data['verbose'] == 'on':
        print("\n")

    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

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
        X_fit = X.copy()

    # Calculate the role of alpha vs coefficient values
    alpha_min = np.log10(data['alpha_min'])
    alpha_max = np.log10(data['alpha_max'])    
    n_alpha = data['n_alpha']    
    alphas = 10**np.linspace(alpha_min,alpha_max,n_alpha)
    # alphas = 10**np.linspace(-3,3,100)
    lasso = Lasso(max_iter=10000)
    coefs = []
    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(X_fit, y)
        coefs.append(lasso.coef_)
    alpha_vs_coef = pd.DataFrame({
        'alpha': alphas,
        'coef': coefs
        }).set_index("alpha")
    model_outputs['alpha_vs_coef'] = alpha_vs_coef

    # LassoCV k-fold cross validation via coordinate descent
    model_cv = LassoCV(cv=data['nfolds'], random_state=0, max_iter=10000).fit(X_fit, y)
    model_objects['LassoCV'] = model_cv
    alpha_cv = model_cv.alpha_

    # LassoLarsCV k-fold cross validation via least angle regression
    model_lars_cv = LassoLarsCV(cv=data['nfolds'], max_iter=10000).fit(X_fit, y)
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
        plt.axvline(lasso.alpha_, linestyle="--", color="black", 
                    label="CV selected alpha={:.3e}".format(model_cv.alpha_))        
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
        plt.axvline(lasso.alpha_, linestyle="--", color="black", 
                    label="LarsCV selected alpha={:.3e}".format(model_lars_cv.alpha_))

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
            label="AIC selected alpha={:.3e}".format(model_aic.alpha_),
            linestyles="--",
            color="tab:blue",
        )
        ax.vlines(
            alpha_bic,
            results["BIC"].min(),
            results["BIC"].max(),
            label="BIC selected alpha={:.3e}".format(model_bic.alpha_),
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

    # LassoLarsIC Plot sequence of alphas to minimize AIC and BIC
    if data['verbose'] == 'on':
        plt.figure()
        aic_criterion = model_aic.criterion_
        bic_criterion = model_bic.criterion_
        index_alpha_path_aic = np.flatnonzero(model_aic.alphas_ == model_aic.alpha_)[0]
        index_alpha_path_bic = np.flatnonzero(model_bic.alphas_ == model_bic.alpha_)[0]
        # print('check index alpha: ',index_alpha_path_aic == index_alpha_path_bic)
        plt.plot(aic_criterion, color="tab:blue", marker="o", label="AIC criterion")
        plt.plot(bic_criterion, color="tab:orange", marker="o", label="BIC criterion")
        # vline for alpha for aic
        plt.vlines(
            index_alpha_path_aic,
            aic_criterion.min(),
            aic_criterion.max(),
            color="tab:blue",
            linestyle="--",
            label="AIC selected alpha={:.3e}".format(model_aic.alpha_),
        )
        # vline for alpha for bic
        plt.vlines(
            index_alpha_path_bic,
            aic_criterion.min(),
            aic_criterion.max(),
            color="tab:orange",
            linestyle="--",
            label="BIC selected alpha={:.3e}".format(model_bic.alpha_),
        )
        plt.legend()
        plt.ylabel("Information Criterion (AIC or BIC)")
        plt.xlabel("Lasso model sequence")
        _ = plt.title("LassoLarsIC - Model sequence of AIC and BIC")
        plt.savefig("LassoLarsIC_sequence_of_AIC_BIC.png", dpi=300)

    # Calculate regression stats
    stats_cv = linear_regression_stats(X_fit, y, model_cv)
    stats_lars_cv = linear_regression_stats(X_fit, y, model_lars_cv)
    stats_aic = linear_regression_stats(X_fit, y, model_aic)
    stats_bic = linear_regression_stats(X_fit, y, model_bic)

    # residual plot for training error
    if data['verbose'] == 'on':
        y_pred_cv = stats_cv['y_pred']
        y_pred_lars_cv = stats_lars_cv['y_pred']
        y_pred_aic = stats_aic['y_pred']
        y_pred_bic = stats_bic['y_pred']
        res_cv = stats_cv['residuals']
        res_lars_cv = stats_lars_cv['residuals']
        res_aic = stats_aic['residuals']
        res_bic = stats_bic['residuals']
        rmse_cv = stats_cv['RMSE']
        rmse_lars_cv = stats_lars_cv['RMSE']
        rmse_aic = stats_aic['RMSE']
        rmse_bic = stats_bic['RMSE']
        plt.figure()
        plt.scatter(y_pred_cv, (res_cv), s=40, label=('LassoCV (RMSE={:.2f})'.format(rmse_cv)))
        plt.scatter(y_pred_lars_cv, (res_lars_cv), s=15, label=('LassoLarsCV (RMSE={:.2f})'.format(rmse_lars_cv)))
        plt.scatter(y_pred_aic, (res_aic), s=10, label=('LassoLarsAIC (RMSE={:.2f})'.format(rmse_aic)))
        plt.scatter(y_pred_bic, (res_bic), s=5, label=('LassoLarsBIC (RMSE={:.2f})'.format(rmse_bic)))
        rmse_cv = np.sqrt(np.mean((res_cv)**2))
        plt.hlines(y=0, xmin=min(y), xmax=max(y), color='k')
        plt.title("Residual plot for training error")
        plt.legend();
        plt.xlabel('y_pred')
        plt.ylabel('residual')
        plt.savefig("Lasso_residuals.png", dpi=300)

    # Find the AIC and BIC of the LassoLarsAIC and LassoLarsBIC models
    min_index_aic = model_outputs['alpha_vs_AIC_BIC']['AIC'].idxmin()
    min_index_bic = model_outputs['alpha_vs_AIC_BIC']['BIC'].idxmin()
    AIC_for_LassoLarsAIC = model_outputs['alpha_vs_AIC_BIC']['AIC'][min_index_aic]
    BIC_for_LassoLarsAIC = model_outputs['alpha_vs_AIC_BIC']['BIC'][min_index_aic]
    AIC_for_LassoLarsBIC = model_outputs['alpha_vs_AIC_BIC']['AIC'][min_index_bic]
    BIC_for_LassoLarsBIC = model_outputs['alpha_vs_AIC_BIC']['BIC'][min_index_bic]

    # Make the model_outputs dataframes
    list1_name = ['alpha','r-squared','adjusted r-squared',
                        'nobs','df residuals','df model',
                        'F-statistic','Prob (F-statistic)','RMSE',
                        'Log-Likelihood','AIC','BIC']
    list2_name = list(stats_cv['popt']['Feature'])
    list3_name = list1_name + list2_name

    list1_cv = [model_cv.alpha_, stats_cv["rsquared"], stats_cv["adj_rsquared"],
                       stats_cv["nobs"], stats_cv["df"], stats_cv["dfn"], 
                       stats_cv["Fstat"], stats_cv["pvalue"], stats_cv["RMSE"],  
                       stats_cv["log_likelihood"],stats_cv["aic"],stats_cv["bic"]]
    list2_cv = list(stats_cv['popt']['param'])
    list3_cv = list1_cv + list2_cv

    list1_lars_cv = [model_lars_cv.alpha_, stats_lars_cv["rsquared"], stats_lars_cv["adj_rsquared"], 
                       stats_lars_cv["nobs"], stats_lars_cv["df"], stats_lars_cv["dfn"], 
                       stats_lars_cv["Fstat"], stats_lars_cv["pvalue"], stats_lars_cv["RMSE"], 
                       stats_lars_cv["log_likelihood"],stats_lars_cv["aic"],stats_lars_cv["bic"]]
    list2_lars_cv = list(stats_lars_cv['popt']['param'])
    list3_lars_cv = list1_lars_cv + list2_lars_cv

    list1_aic = [model_aic.alpha_, stats_aic["rsquared"], stats_aic["adj_rsquared"], 
                       stats_aic["nobs"], stats_aic["df"], stats_aic["dfn"], 
                       stats_aic["Fstat"], stats_aic["pvalue"], stats_aic["RMSE"], 
                       stats_aic["log_likelihood"],AIC_for_LassoLarsAIC,BIC_for_LassoLarsAIC]
    list2_aic = list(stats_aic['popt']['param'])
    list3_aic = list1_aic + list2_aic

    list1_bic = [model_bic.alpha_, stats_bic["rsquared"], stats_bic["adj_rsquared"], 
                       stats_bic["nobs"], stats_bic["df"], stats_bic["dfn"], 
                       stats_bic["Fstat"], stats_bic["pvalue"], stats_bic["RMSE"], 
                       stats_bic["log_likelihood"],AIC_for_LassoLarsBIC,BIC_for_LassoLarsBIC]
    list2_bic = list(stats_bic['popt']['param'])
    list3_bic = list1_bic + list2_bic

    y_pred = pd.DataFrame(
        {
            "LassoCV": stats_cv['y_pred'],
            "LassoLarsCV": stats_lars_cv['y_pred'],
            "LassoLarsAIC": stats_aic['y_pred'],
            "LassoLarsBIC": stats_bic['y_pred']
        }
        )
    y_pred.index = y.index
    model_outputs['y_pred'] = y_pred

    residuals = pd.DataFrame(
        {
            "LassoCV": stats_cv['residuals'],
            "LassoLarsCV": stats_lars_cv['residuals'],
            "LassoLarsAIC": stats_aic['residuals'],
            "LassoLarsBIC": stats_bic['residuals']
        }
        )
    residuals.index = y.index
    model_outputs['residuals'] = residuals

    # Table of all popt incl coef=0
    popt_table = pd.DataFrame(
        {
            "Feature": list2_name,
            "LassoCV": list2_cv,
            "LassoLarsCV": list2_lars_cv,
            "LassoLarsAIC": list2_aic,
            "LassoLarsBIC": list2_bic
        }
        )
    popt_table.set_index('Feature',inplace=True)
    model_outputs['popt_table'] = popt_table
    
    # Calculate the covariance matrix of the features
    # popt, pcov, and vif of only the selected features (excl coef=0)
    popt_all = {}
    pcov_all = {}
    vif_all = {}
    # vif = pd.DataFrame()
    col = X_fit.columns
    # LassoCV
    model_ = model_objects['LassoCV']
    popt = stats_cv['popt'].copy()
    X_ = X_fit.copy()
    for i in range(len(model_.coef_)):   # set X col to zero if coef = 0
        if model_.coef_[i]==0:
            X_ = X_.drop(col[i], axis = 1)
            popt = popt.drop(index=i+1)
    if not X_has_dummies:
        X__ = sm.add_constant(X_)    # Add a constant for the intercept
        pcov = pd.DataFrame(np.cov(X__, rowvar=False), index=X__.columns)
        pcov.columns = X__.columns
        pcov_all['LassoCV'] = pcov
        vif = pd.DataFrame()
        vif['Feature'] = X__.columns
        vif["VIF"] = [variance_inflation_factor(X__.values, i)
                            for i in range(len(X__.columns))]
        vif.set_index('Feature',inplace=True)
        vif_all["LassoCV"] = vif
    popt.set_index('Feature',inplace=True)
    popt_all['LassoCV'] = popt
    # LassoLarsCV
    model_ = model_objects['LassoLarsCV']
    popt = stats_lars_cv['popt'].copy()
    X_ = X_fit.copy()
    for i in range(len(model_.coef_)):   # set X col to zero if coef = 0
        if model_.coef_[i]==0:
            X_ = X_.drop(col[i], axis = 1)
            popt = popt.drop(index=i+1)
    if not X_has_dummies:
        X__ = sm.add_constant(X_)    # Add a constant for the intercept
        pcov = pd.DataFrame(np.cov(X__, rowvar=False), index=X__.columns)
        pcov.columns = X__.columns
        pcov_all['LassoLarsCV'] = pcov
        vif = pd.DataFrame()
        vif['Feature'] = X__.columns
        vif["VIF"] = [variance_inflation_factor(X__.values, i)
                            for i in range(len(X__.columns))]
        vif.set_index('Feature',inplace=True)
        vif_all["LassoLarsCV"] = vif
    popt.set_index('Feature',inplace=True)
    popt_all['LassoLarsCV'] = popt
    # LassoLarsAIC
    model_ = model_objects['LassoLarsAIC']
    popt = stats_aic['popt'].copy()
    X_ = X_fit.copy()
    for i in range(len(model_.coef_)):   # set X col to zero if coef = 0
        if model_.coef_[i]==0:
            X_ = X_.drop(col[i], axis = 1)
            popt = popt.drop(index=i+1)
    if not X_has_dummies:
        X__ = sm.add_constant(X_)    # Add a constant for the intercept
        pcov = pd.DataFrame(np.cov(X__, rowvar=False), index=X__.columns)
        pcov.columns = X__.columns
        pcov_all['LassoLarsAIC'] = pcov
        vif = pd.DataFrame()
        vif['Feature'] = X__.columns
        vif["VIF"] = [variance_inflation_factor(X__.values, i)
                            for i in range(len(X__.columns))]
        vif.set_index('Feature',inplace=True)
        vif_all["LassoLarsAIC"] = vif
    popt.set_index('Feature',inplace=True)
    popt_all['LassoLarsAIC'] = popt
    # LassoLarsBIC
    model_ = model_objects['LassoLarsBIC']
    popt = stats_bic['popt'].copy()
    X_ = X_fit.copy()
    for i in range(len(model_.coef_)):   # set X col to zero if coef = 0
        if model_.coef_[i]==0:
            X_ = X_.drop(col[i], axis = 1)
            popt = popt.drop(index=i+1)
    if not X_has_dummies:
        X__ = sm.add_constant(X_)    # Add a constant for the intercept
        pcov = pd.DataFrame(np.cov(X__, rowvar=False), index=X__.columns)
        pcov.columns = X__.columns
        pcov_all['LassoLarsBIC'] = pcov
        vif = pd.DataFrame()
        vif['Feature'] = X__.columns
        vif["VIF"] = [variance_inflation_factor(X__.values, i)
                            for i in range(len(X__.columns))]
        vif.set_index('Feature',inplace=True)
        vif_all["LassoLarsBIC"] = vif
    popt.set_index('Feature',inplace=True)
    popt_all['LassoLarsBIC'] = popt
    # save pcov, vif, popt
    if not X_has_dummies:
        # save vif and pcov
        model_outputs['vif'] = vif_all
        model_outputs['pcov'] = pcov_all
    model_outputs['popt'] = popt_all

    if not X_has_dummies:
        # Make big VIF table of all models in one table
        # get row indicdes of non-zero coef values in each model col
        idx = popt_table.apply(lambda col: col[col != 0].index.tolist())
        # initialize vif_table same as popt_table but with nan values
        vif_table = pd.DataFrame(np.nan, index=popt_table.index, columns=popt_table.columns)
        # Put in the VIF values in each model column
        # LassoCV
        vif = model_outputs['vif']['LassoCV']['VIF'].values
        vif_table.loc[idx['LassoCV'], "LassoCV"] = vif
        # LassoLarsCV
        vif = model_outputs['vif']['LassoLarsCV']['VIF'].values
        vif_table.loc[idx['LassoLarsCV'], "LassoLarsCV"] = vif
        # LassoLarsAIC
        vif = model_outputs['vif']['LassoLarsAIC']['VIF'].values
        vif_table.loc[idx['LassoLarsAIC'], "LassoLarsAIC"] = vif
        # LassoLarsBIC
        vif = model_outputs['vif']['LassoLarsBIC']['VIF'].values
        vif_table.loc[idx['LassoLarsBIC'], "LassoLarsBIC"] = vif
        model_outputs['vif_table'] = vif_table
    
    stats = pd.DataFrame(
        {
            "Statistic": list1_name,
            "LassoCV": list1_cv,
            "LassoLarsCV": list1_lars_cv,
            "LassoLarsAIC": list1_aic,
            "LassoLarsBIC": list1_bic
        }
        )
    stats.set_index('Statistic',inplace=True)
    model_outputs['stats'] = stats
    
    # Print model_outputs
    if data['verbose'] == 'on':
        print("Lasso regression statistics of best models in model_outputs['stats']:")
        print("\n")
        print(model_outputs['stats'].to_markdown(index=True))
        print("\n")
        print("Coefficients of best models in model_outputs['popt']:")
        print("\n")
        print(model_outputs['popt_table'].to_markdown(index=True))
        print("\n")
        if not X_has_dummies:
            print("Variance Inflation Factors model_outputs['vif']:")
            print("Note: VIF>5 indicates excessive collinearity")
            print("\n")
            print(model_outputs['vif_table'].to_markdown(index=True))
            print("\n")

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return model_objects, model_outputs

def vif_ridge(X, pen_factors, is_corr=False):

    """
    Variance Inflation Factor for Ridge regression 

    adapted from statsmodels function vif_ridge by Josef Perktold https://gist.github.com/josef-pkt
    source: https://github.com/statsmodels/statsmodels/issues/1669
    source: https://stackoverflow.com/questions/23660120/variance-inflation-factor-in-ridge-regression-in-python
    author: https://stackoverflow.com/users/333700/josef
    Josef is statsmodels maintainer and developer, semi-retired from scipy.stats maintainance

    assumes penalization is on standardized feature variables
    assumes alpha is scaled by n_samples in calc of penalty factors if using sklearn Ridge (see note below)
    data should not include a constant

    Parameters
    ----------
    X : array_like with dimension n_samples x n_features
        correlation matrix if is_corr=True or standardized feature data if is_corr is False (default).
    pen_factors : iterable array of of regularization penalty factors with dimension n_alpha 
        If you are using sklearn Ridge for the analysis, then:
            pen_factor = alphas / n_samples
        If you are using statsmodels OLS .fit_regularized(L1_wt=0, then:
            pen_factor = alphas
        where alphas is the iterable array of alpha inputs to sklearn or statsmodels
        (see explanation in note below for difference between sklearn and statsmodels)
    is_corr : bool (default False)
        Boolean to indicate how corr_x is interpreted, see corr_x

    Returns
    -------
    vif : ndarray
        variance inflation factors for parameters in columns and 
        ridge penalization factors in rows

    could be optimized for repeated calculations

    Note about scaling of alpha in statsmodels vs sklearn 
    -------
    An analysis by Paul Zivich (https://sph.unc.edu/adv_profile/paul-zivich/) explains 
    how to get the same results of ridge regression from statsmodels and sklearn. 
    The difference is that sklearn's Ridge function scales the input of the 'alpha' 
    regularization term during excecution as alpha / n where n is the number of observations, 
    compared with statsmodels which does not apply this scaling of the regularization 
    parameter during execution. You can have the ridge implementations match 
    if you re-scale the sklearn input alpha = alpha / n for statsmodels. 
    Note that this rescaling of alpha only applies to ridge regression. 
    The sklearn and statsmodels results for Lasso regression using exactly 
    the same alpha values for input without rescaling.
    
    Here is a link to the original post of this analysis by Paul Zivich:
    
    https://stackoverflow.com/questions/72260808/mismatch-between-statsmodels-and-sklearn-ridge-regression
    
    -------
    Example use of vif_ridge using sklearn for the analysis:
    
    from sklearn.datasets import load_diabetes
    from stepAIC import vif_ridge
    import numpy as np
    import pandas as pd
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    n_samples = X.shape[0]
    alphas = np.array([0.01,0.1,1,10,100])
    pen_factors = alphas / n_samples 
    vifs = pd.DataFrame(vif_ridge(X, pen_factors))
    vifs.columns = X.columns
    vifs.index = alphas
    vifs = vifs.rename_axis("alpha")
    print(vifs)
    
    Output table of VIF vs alpha for each column of X:
                 age       sex       bmi        bp         s1         s2  \
    alpha                                                                  
    0.01    1.217226  1.277974  1.509267  1.459302  58.892664  38.997322   
    0.10    1.216506  1.277102  1.507746  1.458177  56.210859  37.300175   
    1.00    1.209504  1.268643  1.493706  1.447171  37.168552  25.227946   
    10.00   1.145913  1.192754  1.384258  1.347925   4.766889   4.250315   
    100.00  0.724450  0.724378  0.781325  0.767933   0.313543   0.465363   

                   s3        s4         s5        s6  
    alpha                                             
    0.01    15.338432  8.881508  10.032564  1.484506  
    0.10    14.786345  8.798110   9.656775  1.483458  
    1.00    10.827196  8.107172   6.979474  1.473147  
    10.00    3.300027  4.946247   2.222185  1.378038  
    100.00   0.615300  0.690887   0.758717  0.791855      
    """

    import numpy as np
    
    X = np.asarray(X)
    if not is_corr:
        # corr = np.corrcoef(X, rowvar=0, bias=True)    # bias is deprecated and has no effect
        corr = np.corrcoef(X, rowvar=0)
    else:
        corr = X

    eye = np.eye(corr.shape[1])
    res = []
    for k in pen_factors:
        minv = np.linalg.inv(corr + k * eye)
        vif = minv.dot(corr).dot(minv)
        res.append(np.diag(vif))

    return np.asarray(res)

def ridge(X, y, **kwargs):

    """
    Python function for Ridge linear regression 
    
    by
    Greg Pelletier
    gjpelletier@gmail.com
    21-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        standardize= 'on' (default) or 'off' where
            'on': standardize X using sklearn.preprocessing StandardScaler
            'off': do not standardize X
        alpha_min= minimum value of range of alphas to evaluate (default=1e-3)
        alpha_max= maximum value of range of alphas to evaluate (default=1e3)
        n_alpha= number of log-spaced alphas to evaluate (default=1000)
        vif_target= VIF target for use with RidgeVIF (default=1.0)
        verbose= 'on' (default) or 'off' where
            'on': display model summary on screen 
            'off': turn off display of model summary on screen

    Standardization is generally recommended for Ridge regression.

    RETURNS
        model_objects, model_outputs
            model_objects are the output objects from 
                sklearn.linear_model Ridge or RidgeCV
                of the final best models using the following four methods: 
                - RidgeCV: sklearn RidgeCV 
                - RidgeAIC: sklearn Ridge using minimum AIC to find best alpha
                - RidgeBIC: sklearn Ridge using minimum BIC to find best alpha
                - RidgeVIF: sklearn Ridge using target VIF to find best alpha
            model_outputs is a dictionary of the following outputs 
                from the four Ridge linear regression model methods:
                - 'scaler': sklearn.preprocessing StandardScaler for X
                - 'standardize': 'on' scaler was used for X, 'off' scaler not used
                - 'alpha_vs_coef': model coefficients for each X variable
                    as a function of alpha using Ridge
                - 'alpha_vs_penalty': penalty factors
                    as a function of alpha using Ridge
                - 'alpha_vs_AIC_BIC': AIC and BIC as a function of alpha 
                    using RidgeAIC and RidgeBIC
                - 'best_alpha_aic': alpha at the lowest AIC value from RidgeAIC
                - 'best_alpha_bic': alpha at the lowest BIC value from RidgeBIC
                - 'best_alpha_vif': alpha at the VIF closest to the target VIF value from RidgeVIF
                - 'y_pred': Predicted y values for each of the methods
                - 'residuals': Residuals (y-y_pred) for each of the methods
                - 'popt': Constant (intercept) and coefficients for the 
                    best fit models from each of the methods
                - 'popt_table': Constant (intercept) and coefficients
                    of best fit of all methods in one table
                - 'pcov': Covariance matrix of features 
                - 'vif': Variance Inlfation Factors of features of each method
                - 'vif_table': Variance Inlfation Factors of features of 
                    all methods in one table
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = ridge(X, y)

    """

    from stepAIC import linear_regression_stats, vif_ridge, detect_dummy_variables
    import time
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    import warnings
    import sys
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
   
    # Define default values of input data arguments
    defaults = {
        'standardize': 'on',
        'alpha_min': 1.0e-3,
        'alpha_max': 1.0e3,
        'n_alpha': 1000,
        'vif_target': 1.0,
        'verbose': 'on'
        }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # check for input errors
    ctrl = detect_dummy_variables(X)
    if ctrl:
        print('Check X: Ridge can not handle dummies. Try using lasso if X has dummies.','\n')
        sys.exit()
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
    ctrl = data['alpha_min'] > 0 
    if not ctrl:
        print('Check inputs of alpha_min, it must be greater than zero!','\n')
        sys.exit()
    ctrl = data['alpha_max'] > data['alpha_min'] 
    if not ctrl:
        print('Check inputs of alpha_max, it must be greater than alpha_min!','\n')
        sys.exit()
    ctrl = data['n_alpha'] > 1 
    if not ctrl:
        print('Check inputs of n_alpha, it must be greater than 1!','\n')
        sys.exit()
        
    # Suppress warnings
    warnings.filterwarnings('ignore')
    print('Fitting Ridge regression models, please wait ...')
    if data['verbose'] == 'on':
        print("\n")

    # Set start time for calculating run time
    start_time = time.time()

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

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
        X_fit = X.copy()

    # Calculate the role of alpha vs coefficient values
    alpha_min = np.log10(data['alpha_min'])
    alpha_max = np.log10(data['alpha_max'])    
    n_alpha = data['n_alpha']    
    alphas = 10**np.linspace(alpha_min,alpha_max,n_alpha)
    # alphas = np.logspace(data['alpha_min'],data['alpha_max'],data['n_alpha'])
    # ridge = Ridge(max_iter=15000)
    ridge = Ridge()
    coefs = []
    pen_factors = []
    n_samples, n_columns = X_fit.shape   # sklearn ridge scales alpha by n_samples
    for a in alphas:
        ridge.set_params(alpha=a)
        ridge.fit(X_fit, y)
        coefs.append(ridge.coef_)

    # Calculate pen_factors for input to vif_ridge function
    # sklearn ridge scales input alpha during execution by dividing by n_samples to calculate penalty factor
    # statsmodels ridge regression does not scale input alpha by n_samples during execution
    # https://stackoverflow.com/questions/72260808/mismatch-between-statsmodels-and-sklearn-ridge-regression
    # Also see discussion at this link for further explanation of why pen_factors = alphas / n_samples using sklearn:
    # https://github.com/statsmodels/statsmodels/issues/1669
    # uncomment one of the following two lines depending on if you are using sklearn or statsmodels
    pen_factors = alphas / n_samples   # use this line if using sklearn Ridge
    # pen_factors = alphas     # use this for pen_factors if using statsmodels OLS fit_regularized L1_wt=0

    alpha_vs_coef = pd.DataFrame({
        'alpha': alphas,
        'coef': coefs
        }).set_index("alpha")
    alpha_vs_penalty = pd.DataFrame({
        'alpha': alphas,
        'pen_factors':pen_factors
        }).set_index("alpha")
    vifs = pd.DataFrame(vif_ridge(X_fit, pen_factors))
    vifs.columns = X_fit.columns
    vifs_ = vifs.copy()     # vifs_ = vifs before inserting alphas
    vifs.insert(0, 'alpha', alphas)
    vifs.set_index('alpha',inplace=True)
        
    model_outputs['alpha_vs_coef'] = alpha_vs_coef
    model_outputs['alpha_vs_penalty'] = alpha_vs_penalty
    model_outputs['alpha_vs_vif'] = vifs
    
    # RidgeCV default using MSE
    model_cv = RidgeCV(alphas=alphas, store_cv_results=True).fit(X_fit, y)
    model_objects['RidgeCV'] = model_cv
    alpha_cv = model_cv.alpha_
    # Get the cross-validated MSE for each alpha
    model_cv_mse_each_fold = model_cv.cv_results_  # Shape: (n_samples, n_alphas)
    model_cv_mse_mean = np.mean(model_cv.cv_results_, axis=0)

    # RidgeAIC/BIC - Ridge with manual AIC and BIC scoring
    log_likelihood = lambda n, rss: -0.5 * n * (np.log(2 * np.pi) + np.log(rss/n) + 1)
    aic = lambda n, rss, k: -2 * log_likelihood(n,rss) + 2 * k
    bic = lambda n, rss, k: -2 * log_likelihood(n,rss) + k * np.log(n)
    # RidgeAIC = model_aic
    best_aic = float('inf')
    best_alpha_aic = None    
    aic_ = np.full(len(alphas), np.nan)
    for i,alpha in enumerate(alphas):
        # ridge = RidgeCV(alphas=[alpha], store_cv_results=True)
        # print(float(alpha)
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_fit, y)        
        # Calculate residual sum of squares (RSS)
        y_pred = ridge.predict(X_fit)
        rss = np.sum((y - y_pred) ** 2)        
        # Number of parameters (coefficients + intercept)
        k = X_fit.shape[1] + 1  # Features + intercept        
        # Calculate and track the best AIC
        aic_[i] = aic(n=X_fit.shape[0], rss=rss, k=k)        
        if aic_[i] < best_aic:
            best_aic = aic_[i]
            best_alpha_aic = alpha
    model_aic = Ridge(alpha=best_alpha_aic).fit(X_fit, y)
    model_objects['RidgeAIC'] = model_aic  
    model_outputs['best_alpha_aic'] = best_alpha_aic
    # RidgeBIC - model_bic
    best_bic = float('inf')
    best_alpha_bic = None    
    bic_ = np.full(len(alphas), np.nan)
    for i,alpha in enumerate(alphas):
        # ridge = RidgeCV(alphas=[alpha], store_cv_results=True)
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_fit, y)        
        # Calculate residual sum of squares (RSS)
        y_pred = ridge.predict(X_fit)
        rss = np.sum((y - y_pred) ** 2)        
        # Number of parameters (coefficients + intercept)
        k = X_fit.shape[1] + 1  # Features + intercept        
        # Calculate and track the best BIC
        bic_[i] = bic(n=X_fit.shape[0], rss=rss, k=k)        
        if bic_[i] < best_bic:
            best_bic = bic_[i]
            best_alpha_bic = alpha
    model_bic = Ridge(alpha=best_alpha_bic).fit(X_fit, y)
    model_objects['RidgeBIC'] = model_bic  
    model_outputs['best_alpha_bic'] = best_alpha_bic

    # RidgeVIF - Ridge with VIF target
    vif_target = data['vif_target']
    rmse_vif_res = np.sqrt(np.sum((vif_target-vifs_)**2,1))
    idx = (np.abs(rmse_vif_res)).argmin()
    best_alpha_vif = alphas[idx]
    model_vif = Ridge(alpha=best_alpha_vif).fit(X_fit, y)
    model_objects['RidgeVIF'] = model_vif  
    model_outputs['best_alpha_vif'] = best_alpha_vif
    
    # results of alphas to minimize AIC and BIC
    alpha_vs_AIC_BIC = pd.DataFrame(
        {
            "alpha": alphas,
            "AIC": aic_,
            "BIC": bic_,
        }
        ).set_index("alpha")
    model_outputs['alpha_vs_AIC_BIC'] = alpha_vs_AIC_BIC
    
    # Plot the results of ridge coef as function of alpha
    if data['verbose'] == 'on':
        ax = plt.gca()
        ax.plot(alphas, coefs)
        ax.set_xscale('log')
        plt.axis('tight')
        plt.xlabel(r"$\alpha$")
        plt.legend(X_fit.columns)
        plt.ylabel('Coefficients')
        plt.title(r'Ridge regression coefficients as a function of $\alpha$');
        plt.savefig("Ridge_alpha_vs_coef.png", dpi=300)

    # Plot the VIF of coefficients as function of alpha
    if data['verbose'] == 'on':
        # model = model_vif
        plt.figure()
        ax = plt.gca()
        ax.plot(alphas, vifs)
        ax.set_xscale('log')
        ax.set_yscale('log')
        # plt.yscale("log")
        plt.axvline(best_alpha_vif, linestyle="--", color="black")        
        ax.text(best_alpha_vif, np.percentile(vifs.values.flatten(),5), 
                "alpha at VIF target {:.1f} ={:.3e}".format(vif_target,best_alpha_vif), 
                rotation=90, va='bottom', ha='right')
        plt.axis('tight')
        plt.xlabel(r"$\alpha$")
        plt.legend(X_fit.columns)
        plt.ylabel('VIF')
        plt.title(r'VIF of coefficients as a function of $\alpha$');
        ax2 = ax.twinx()
        ax2.plot(alphas, rmse_vif_res, 'r--', label='target')
        ax2.set_ylabel('RMS difference between VIF and target {:.1f}'.format(vif_target), color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_yscale('log')
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        plt.savefig("Ridge_alpha_vs_vif.png", dpi=300)

    # RidgeCV Plot the MSE vs alpha for each fold
    if data['verbose'] == 'on':
        model = model_cv
        plt.figure()
        plt.semilogx(alphas, model_cv_mse_each_fold.T, linestyle=":")
        plt.plot(
            alphas,
            model_cv_mse_mean,
            color="black",
            label="Average across the folds",
            linewidth=2,
        )
        plt.axvline(model.alpha_, linestyle="--", color="black", 
                    label="CV selected alpha={:.3e}".format(model_cv.alpha_))        
        # ymin, ymax = 2300, 3800
        # plt.ylim(ymin, ymax)
        plt.xlabel(r"$\alpha$")
        plt.ylabel("Mean Square Error")
        # plt.yscale("log")
        plt.legend()
        _ = plt.title(
            "RidgeCV - Mean Square Error on each fold"
        )
        plt.savefig("RidgeCV_alpha_vs_MSE.png", dpi=300)
    
    # RidgeAIC/BIC Plot of alphas to minimize AIC and BIC
    if data['verbose'] == 'on':
        results = alpha_vs_AIC_BIC
        ax = results.plot()
        ax.vlines(
            best_alpha_aic,
            results["AIC"].min(),
            results["AIC"].max(),
            label="AIC selected alpha={:.3e}".format(best_alpha_aic),
            linestyles="--",
            color="tab:blue",
        )
        ax.vlines(
            best_alpha_bic,
            results["BIC"].min(),
            results["BIC"].max(),
            label="BIC selected alpha={:.3e}".format(best_alpha_bic),
            linestyle="--",
            color="tab:orange",
        )
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Information Criterion (AIC or BIC)")
        ax.set_xscale("log")
        ax.legend()
        _ = ax.set_title(
            "RidgeAIC/BIC - Information Criterion for model selection"
        )
        plt.savefig("Ridge_alpha_vs_AIC_BIC.png", dpi=300)

    # Calculate regression stats
    stats_cv = linear_regression_stats(X_fit, y, model_cv)
    stats_aic = linear_regression_stats(X_fit, y, model_aic)
    stats_bic = linear_regression_stats(X_fit, y, model_bic)
    stats_vif = linear_regression_stats(X_fit, y, model_vif)

    # residual plot for training error
    if data['verbose'] == 'on':
        y_pred_cv = stats_cv['y_pred']
        y_pred_aic = stats_aic['y_pred']
        y_pred_bic = stats_bic['y_pred']
        y_pred_vif = stats_vif['y_pred']
        res_cv = stats_cv['residuals']
        res_aic = stats_aic['residuals']
        res_bic = stats_bic['residuals']
        res_vif = stats_vif['residuals']
        rmse_cv = stats_cv['RMSE']
        rmse_aic = stats_aic['RMSE']
        rmse_bic = stats_bic['RMSE']
        rmse_vif = stats_vif['RMSE']
        plt.figure()
        plt.scatter(y_pred_cv, (res_cv), s=40, label=('RidgeCV (RMSE={:.2f})'.format(rmse_cv)))
        plt.scatter(y_pred_aic, (res_aic), s=20, label=('RidgeAIC (RMSE={:.2f})'.format(rmse_aic)))
        plt.scatter(y_pred_bic, (res_bic), s=10, label=('RidgeBIC (RMSE={:.2f})'.format(rmse_bic)))
        plt.scatter(y_pred_vif, (res_vif), s=5, label=('RidgeVIF (RMSE={:.2f})'.format(rmse_vif)))
        plt.hlines(y=0, xmin=min(y), xmax=max(y), color='k')
        plt.title("Residual plot for training error")
        plt.legend();
        plt.xlabel('y_pred')
        plt.ylabel('residual')
        plt.savefig("Ridge_residuals.png", dpi=300)

    # Make the model_outputs dataframes
    list1_name = ['alpha','r-squared','adjusted r-squared',
                        'nobs','df residuals','df model',
                        'F-statistic','Prob (F-statistic)','RMSE',
                        'Log-Likelihood','AIC','BIC']
    list2_name = list(stats_cv['popt']['Feature'])
    list3_name = list1_name + list2_name

    list1_cv = [model_cv.alpha_, stats_cv["rsquared"], stats_cv["adj_rsquared"],
                       stats_cv["nobs"], stats_cv["df"], stats_cv["dfn"], 
                       stats_cv["Fstat"], stats_cv["pvalue"], stats_cv["RMSE"],  
                       stats_cv["log_likelihood"],stats_cv["aic"],stats_cv["bic"]]
    list2_cv = list(stats_cv['popt']['param'])
    list3_cv = list1_cv + list2_cv

    list1_aic = [best_alpha_aic, stats_aic["rsquared"], stats_aic["adj_rsquared"], 
                       stats_aic["nobs"], stats_aic["df"], stats_aic["dfn"], 
                       stats_aic["Fstat"], stats_aic["pvalue"], stats_aic["RMSE"], 
                       stats_aic["log_likelihood"],stats_aic["aic"],stats_aic["bic"]]
    list2_aic = list(stats_aic['popt']['param'])
    list3_aic = list1_aic + list2_aic

    list1_bic = [best_alpha_bic, stats_bic["rsquared"], stats_bic["adj_rsquared"], 
                       stats_bic["nobs"], stats_bic["df"], stats_bic["dfn"], 
                       stats_bic["Fstat"], stats_bic["pvalue"], stats_bic["RMSE"], 
                       stats_bic["log_likelihood"],stats_bic["aic"],stats_bic["bic"]]
    list2_bic = list(stats_bic['popt']['param'])
    list3_bic = list1_bic + list2_bic

    list1_vif = [best_alpha_vif, stats_vif["rsquared"], stats_vif["adj_rsquared"], 
                       stats_vif["nobs"], stats_vif["df"], stats_vif["dfn"], 
                       stats_vif["Fstat"], stats_vif["pvalue"], stats_vif["RMSE"], 
                       stats_vif["log_likelihood"],stats_vif["aic"],stats_vif["bic"]]
    list2_vif = list(stats_vif['popt']['param'])
    list3_vif = list1_vif + list2_vif

    y_pred = pd.DataFrame(
        {
            "RidgeCV": stats_cv['y_pred'],
            "RidgeAIC": stats_aic['y_pred'],
            "RidgeBIC": stats_bic['y_pred'],
            "RidgeVIF": stats_vif['y_pred']
        }
        )
    y_pred.index = y.index
    model_outputs['y_pred'] = y_pred

    residuals = pd.DataFrame(
        {
            "RidgeCV": stats_cv['residuals'],
            "RidgeAIC": stats_aic['residuals'],
            "RidgeBIC": stats_bic['residuals'],
            "RidgeVIF": stats_vif['residuals']
        }
        )
    residuals.index = y.index
    model_outputs['residuals'] = residuals

    # Table of all popt incl coef=0
    popt_table = pd.DataFrame(
        {
            "Feature": list2_name,
            "RidgeCV": list2_cv,
            "RidgeAIC": list2_aic,
            "RidgeBIC": list2_bic,
            "RidgeVIF": list2_vif
        }
        )
    popt_table.set_index('Feature',inplace=True)
    model_outputs['popt_table'] = popt_table

    stats = pd.DataFrame(
        {
            "Statistic": list1_name,
            "RidgeCV": list1_cv,
            "RidgeAIC": list1_aic,
            "RidgeBIC": list1_bic,
            "RidgeVIF": list1_vif
        }
        )
    stats.set_index('Statistic',inplace=True)
    model_outputs['stats'] = stats

    # Calculate VIF of X at Ridge regression alpha values
    alphas = [model_outputs['stats']['RidgeCV']['alpha'],
                  model_outputs['stats']['RidgeAIC']['alpha'],
                  model_outputs['stats']['RidgeBIC']['alpha'], 
                  model_outputs['stats']['RidgeVIF']['alpha'],] 
    df = model_outputs['alpha_vs_penalty'].copy()
    df.reset_index(drop=False, inplace=True)
    pf_cv = np.array(df[df['alpha'] == alphas[0]]['pen_factors'])
    pf_aic = np.array(df[df['alpha'] == alphas[1]]['pen_factors'])
    pf_bic = np.array(df[df['alpha'] == alphas[2]]['pen_factors'])
    pf_vif = np.array(df[df['alpha'] == alphas[3]]['pen_factors'])
    pen_factors = [pf_cv, pf_aic, pf_bic, pf_vif]
    vif_calc = vif_ridge(X_fit,pen_factors)
    
    # Calculate the covariance matrix of the features
    popt_all = {}
    pcov_all = {}
    vif_all = {}
    col = X_fit.columns

    # RidgeCV
    model_ = model_objects['RidgeCV']
    popt = stats_cv['popt'].copy()
    X_ = X_fit.copy()
    for i in range(len(model_.coef_)):   # set X col to zero if coef = 0
        if model_.coef_[i]==0:
            X_ = X_.drop(col[i], axis = 1)
            popt = popt.drop(index=i+1)
    X__ = sm.add_constant(X_)    # Add a constant for the intercept
    pcov = pd.DataFrame(np.cov(X__, rowvar=False), index=X__.columns)
    pcov.columns = X__.columns
    popt.set_index('Feature',inplace=True)
    popt_all['RidgeCV'] = popt
    pcov_all['RidgeCV'] = pcov
    vif = pd.DataFrame()
    vif['Feature'] = X__.columns
    vif["VIF"] = np.insert(vif_calc[0], 0, np.nan) 
    #        
    vif.set_index('Feature',inplace=True)
    vif_all["RidgeCV"] = vif

    # RidgeAIC
    model_ = model_objects['RidgeAIC']
    popt = stats_aic['popt'].copy()
    X_ = X_fit.copy()
    for i in range(len(model_.coef_)):   # set X col to zero if coef = 0
        if model_.coef_[i]==0:
            X_ = X_.drop(col[i], axis = 1)
            popt = popt.drop(index=i+1)
    X__ = sm.add_constant(X_)    # Add a constant for the intercept
    pcov = pd.DataFrame(np.cov(X__, rowvar=False), index=X__.columns)
    pcov.columns = X__.columns
    popt.set_index('Feature',inplace=True)
    popt_all['RidgeAIC'] = popt
    pcov_all['RidgeAIC'] = pcov
    vif = pd.DataFrame()
    vif['Feature'] = X__.columns
    vif["VIF"] = np.insert(vif_calc[1], 0, np.nan) 
    #        
    vif.set_index('Feature',inplace=True)
    vif_all["RidgeAIC"] = vif

    # RidgeBIC
    model_ = model_objects['RidgeBIC']
    popt = stats_bic['popt'].copy()
    X_ = X_fit.copy()
    for i in range(len(model_.coef_)):   # set X col to zero if coef = 0
        if model_.coef_[i]==0:
            X_ = X_.drop(col[i], axis = 1)
            popt = popt.drop(index=i+1)
    X__ = sm.add_constant(X_)    # Add a constant for the intercept
    pcov = pd.DataFrame(np.cov(X__, rowvar=False), index=X__.columns)
    pcov.columns = X__.columns
    popt.set_index('Feature',inplace=True)
    popt_all['RidgeBIC'] = popt
    pcov_all['RidgeBIC'] = pcov
    vif = pd.DataFrame()
    vif['Feature'] = X__.columns
    vif["VIF"] = np.insert(vif_calc[2], 0, np.nan) 
    #        
    vif.set_index('Feature',inplace=True)
    vif_all["RidgeBIC"] = vif

    # RidgeVIF
    model_ = model_objects['RidgeVIF']
    popt = stats_vif['popt'].copy()
    X_ = X_fit.copy()
    for i in range(len(model_.coef_)):   # set X col to zero if coef = 0
        if model_.coef_[i]==0:
            X_ = X_.drop(col[i], axis = 1)
            popt = popt.drop(index=i+1)
    X__ = sm.add_constant(X_)    # Add a constant for the intercept
    pcov = pd.DataFrame(np.cov(X__, rowvar=False), index=X__.columns)
    pcov.columns = X__.columns
    popt.set_index('Feature',inplace=True)
    popt_all['RidgeVIF'] = popt
    pcov_all['RidgeVIF'] = pcov
    vif = pd.DataFrame()
    vif['Feature'] = X__.columns
    vif["VIF"] = np.insert(vif_calc[3], 0, np.nan) 
    #        
    vif.set_index('Feature',inplace=True)
    vif_all["RidgeVIF"] = vif
    
    # save vif and pcov in model_outputs
    model_outputs['popt'] = popt_all
    model_outputs['pcov'] = pcov_all
    model_outputs['vif'] = vif_all

    # Make big VIF table of all models in one table
    # get row indicdes of non-zero coef values in each model col
    idx = popt_table.apply(lambda col: col[col != 0].index.tolist())
    # initialize vif_table same as popt_table but with nan values
    vif_table = pd.DataFrame(np.nan, index=popt_table.index, columns=popt_table.columns)
    # Put in the VIF values in each model column
    # RidgeCV
    vif = model_outputs['vif']['RidgeCV']['VIF'].values
    vif_table.loc[idx['RidgeCV'], "RidgeCV"] = vif
    # RidgeAIC
    vif = model_outputs['vif']['RidgeAIC']['VIF'].values
    vif_table.loc[idx['RidgeAIC'], "RidgeAIC"] = vif
    # RidgeBIC
    vif = model_outputs['vif']['RidgeBIC']['VIF'].values
    vif_table.loc[idx['RidgeBIC'], "RidgeBIC"] = vif
    # RidgeVIF
    vif = model_outputs['vif']['RidgeVIF']['VIF'].values
    vif_table.loc[idx['RidgeVIF'], "RidgeVIF"] = vif
    # drop const row from VIF table and save in outputs
    vif_table = vif_table.drop(index=['const'])
    model_outputs['vif_table'] = vif_table

    # Print model_outputs
    if data['verbose'] == 'on':
        print("Ridge regression statistics of best models in model_outputs['stats']:")
        print("\n")
        print(model_outputs['stats'].to_markdown(index=True))
        print("\n")
        print("Coefficients of best models in model_outputs['popt']:")
        print("\n")
        print(model_outputs['popt_table'].to_markdown(index=True))
        print("\n")
        print("Variance Inflation Factors model_outputs['vif']:")
        print("Note: VIF>5 indicates excessive collinearity")
        print("\n")
        print(model_outputs['vif_table'].to_markdown(index=True))
        print("\n")

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return model_objects, model_outputs

