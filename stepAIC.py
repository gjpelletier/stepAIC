# -*- coding: utf-8 -*-

__version__ = "1.0.17"

def stepAIC(X, y, **kwargs):

    """
    Python function for stepwise linear regression to minimize AIC or BIC
    and eliminate non-signficant predictors

    BY
    Greg Pelletier
    gjpelletier@gmail.com
    05-May-2025

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
        direction= 'forward' (default) or 'backward' where
            'forward' (default): 
                1) Start with no predictors in the model
                2) Add the predictor that results in the lowest AIC
                3) Keep adding predictors as long as it reduces AIC
            'backward':
                1) Fit a model with all predictors.
                2) Remove the predictor that results in the lowest AIC
                3) Keep removing predictors as long as it reduces AIC
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
    columns as needed, but y should only be one column

    EXAMPLE USAGE (with the default using AIC as the criterion):
    best_features, best_model = stepAIC(X, y)

    EXAMPLE USAGE (with the option of using BIC as the criterion):
    best_features, best_model = stepAIC(X, y, criterion='BIC')

    """

    import statsmodels.api as sm
    
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
    if data['criterion'] == 'AIC':
        crit = 'AIC'
    elif data['criterion'] == 'BIC':
        crit = 'BIC'
        
    if data['direction'] == 'forward':
    
        # Forward selection to minimize AIC
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
                    print('\nFINAL FORWARD MODEL')
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

        # Backward selection to minimize AIC
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
                    print('\nFINAL BACKWARD MODEL')
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
    
    return selected_features, model
