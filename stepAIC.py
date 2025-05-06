# -*- coding: utf-8 -*-

__version__ = "1.0.11"

def stepAIC(X, y, **kwargs):

    """
    PURPOSE
    Stepwise linear regression to minimize AIC and/or eliminate non-signficant predictors

    BY
    Greg Pelletier
    gjpelletier@gmail.com
    05-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and only contain real numbers)
    X = pandas dataframe of the candidate independent variables (as many columns of data as needed)
    y = pandas dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        verbose= 'on' (default) or 'off' where
            'on': provide model summary at each step
            'off': provide model summary for only the final selected model
        direction= 'forward' (default) or 'backward' where
            'forward' (default): 
                1) Start with no predictors in the model
                2) Add the predictor that results in the lowest AIC
                3) Keep adding predictors as long as it improves the model (reduce AIC)
            'backward':
                1) Fit a model with all predictors.
                2) Remove the predictor that results in the lowest AIC
                3) Keep removing predictors as long as it improves the model (reduce AIC)
        drop_nonsig= 'on' (default) or 'off'
            'on': drop predictors with p-values below threshold p-value (default) 
            'off': keep all predictors regardless of p-value
        p_threshold= threshold p-value to eliminate predictors (default p_threshold= 0.05)                

    NOTE
    Do any necessary/optional cleaning and standardizing of data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column

    EXAMPLE USAGE
    best_features, best_model = stepAIC(df_X, df_y)
    """

    import statsmodels.api as sm
    
    # Define default values of input data arguments
    defaults = {
        'verbose': 'on',
        'direction': 'forward',
        'drop_nonsig': 'on',
        'p_threshold': 0.05
        }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}
    p_threshold = data['p_threshold']
        
    if data['direction'] == 'forward':
    
        # Forward selection to minimize AIC
        selected_features = []
        remaining_features = list(X.columns)
        best_aic = float('inf')
        istep = 0
        while remaining_features:
            aic_with_candidates = []        
            for candidate in remaining_features:
                model = sm.OLS(y, sm.add_constant(X[selected_features + [candidate]])).fit()
                aic_with_candidates.append((model.aic, candidate))
            aic_with_candidates.sort()  # Sort by AIC
            best_new_aic, best_candidate = aic_with_candidates[0]        
            if best_new_aic < best_aic:
                best_aic = best_new_aic
                selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
                istep += 1
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                if data['verbose'] == 'on':
                    print('\nFORWARD STEP',istep,", AIC= {:.2f}".format(model.aic))
                    print('Features added: ', selected_features,'\n')
                    print(model.summary())        
            else:            
                # best_aic = best_new_aic
                # selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                if (data['verbose'] == 'on' or 
                        (data['verbose'] == 'off' and data['drop_nonsig'] == 'off')):
                    print('\nFINAL FORWARD MODEL')
                    print('Best features: ', selected_features,'\n')
                    print(model.summary())
                break            

        if data['drop_nonsig'] == 'on':
    
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
                    print('\nFINAL MODEL AFTER ELIMINATING ANY NON-SIGNIFICANT PREDICTORS')
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
            aic_with_candidates = []        
            model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
            best_aic = model.aic
            # for candidate in remaining_features:
            for candidate in selected_features:
                # model = sm.OLS(y, sm.add_constant(X[selected_features - [candidate]])).fit()
                test_features = selected_features.copy()
                test_features.remove(candidate)
                model = sm.OLS(y, sm.add_constant(X[test_features])).fit()
                aic_with_candidates.append((model.aic, candidate))
            aic_with_candidates.sort()  # Sort by AIC
            best_new_aic, best_candidate = aic_with_candidates[0]        
            if best_new_aic < best_aic:
                best_aic = best_new_aic
                remaining_features.append(best_candidate)
                selected_features.remove(best_candidate)
                istep += 1
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                if data['verbose'] == 'on':
                    print('\nBACKWARD STEP',istep,", AIC= {:.2f}".format(model.aic))
                    print('Features added: ', selected_features,'\n')
                    print(model.summary())        
            else:            
                # remaining_features.remove(best_candidate)
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                if (data['verbose'] == 'on' or 
                        (data['verbose'] == 'off' and data['drop_nonsig'] == 'off')):
                    print('\nFINAL BACKWARD MODEL')
                    print('Best features: ', selected_features,'\n')
                    print(model.summary())
                break            

        if data['drop_nonsig'] == 'on':
    
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
                    print('\nFINAL MODEL AFTER ELIMINATING NON-SIGNIFICANT PREDICTORS')
                    print('Best features: ', selected_features,'\n')
                    print(model.summary())
                    break
    
    return selected_features, model

def stepBIC(X, y, **kwargs):

    """
    PURPOSE
    Stepwise linear regression to minimize BIC and/or eliminate non-signficant predictors

    BY
    Greg Pelletier
    gjpelletier@gmail.com
    05-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and only contain real numbers)
    X = pandas dataframe of the candidate independent variables (as many columns of data as needed)
    y = pandas dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        verbose= 'on' (default) or 'off' where
            'on': provide model summary at each step
            'off': provide model summary for only the final selected model
        direction= 'forward' (default) or 'backward' where
            'forward' (default): 
                1) Start with no predictors in the model
                2) Add the predictor that results in the lowest BIC
                3) Keep adding predictors as long as it improves the model (reduce BIC)
            'backward':
                1) Fit a model with all predictors.
                2) Remove the predictor that results in the lowest BIC
                3) Keep removing predictors as long as it improves the model (reduce BIC)
        drop_nonsig= 'on' (default) or 'off'
            'on': drop predictors with p-values below threshold p-value (default) 
            'off': keep all predictors regardless of p-value
        p_threshold= threshold p-value to eliminate predictors (default p_threshold= 0.05)                

    NOTE
    Do any necessary/optional cleaning and standardizing of data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column

    EXAMPLE USAGE
    best_features, best_model = stepBIC(df_X, df_y)
    """

    import statsmodels.api as sm
    
    # Define default values of input data arguments
    defaults = {
        'verbose': 'on',
        'direction': 'forward',
        'drop_nonsig': 'on',
        'p_threshold': 0.05
        }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}
    p_threshold = data['p_threshold']
        
    if data['direction'] == 'forward':
    
        # Forward selection to minimize BIC
        selected_features = []
        remaining_features = list(X.columns)
        best_bic = float('inf')
        istep = 0
        while remaining_features:
            bic_with_candidates = []        
            for candidate in remaining_features:
                model = sm.OLS(y, sm.add_constant(X[selected_features + [candidate]])).fit()
                bic_with_candidates.append((model.bic, candidate))
            bic_with_candidates.sort()  # Sort by BIC
            best_new_bic, best_candidate = bic_with_candidates[0]        
            if best_new_bic < best_bic:
                best_bic = best_new_bic
                selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
                istep += 1
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                if data['verbose'] == 'on':
                    print('\nFORWARD STEP',istep,", BIC= {:.2f}".format(model.bic))
                    print('Features added: ', selected_features,'\n')
                    print(model.summary())        
            else:            
                # best_bic = best_new_bic
                # selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                if (data['verbose'] == 'on' or 
                        (data['verbose'] == 'off' and data['drop_nonsig'] == 'off')):
                    print('\nFINAL FORWARD MODEL')
                    print('Best features: ', selected_features,'\n')
                    print(model.summary())
                break            

        if data['drop_nonsig'] == 'on':
    
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
                    print('\nFINAL MODEL AFTER ELIMINATING ANY NON-SIGNIFICANT PREDICTORS')
                    print('Best features: ', selected_features,'\n')
                    print(model.summary())
                    break
    
    if data['direction'] == 'backward':

        # Backward selection to minimize BIC
        selected_features = list(X.columns)
        remaining_features = []
        istep = 0
        # while remaining_features:
        while len(selected_features) > 0:
            bic_with_candidates = []        
            model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
            best_bic = model.bic
            # for candidate in remaining_features:
            for candidate in selected_features:
                # model = sm.OLS(y, sm.add_constant(X[selected_features - [candidate]])).fit()
                test_features = selected_features.copy()
                test_features.remove(candidate)
                model = sm.OLS(y, sm.add_constant(X[test_features])).fit()
                bic_with_candidates.append((model.bic, candidate))
            bic_with_candidates.sort()  # Sort by BIC
            best_new_bic, best_candidate = bic_with_candidates[0]        
            if best_new_bic < best_bic:
                best_bic = best_new_bic
                remaining_features.append(best_candidate)
                selected_features.remove(best_candidate)
                istep += 1
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                if data['verbose'] == 'on':
                    print('\nBACKWARD STEP',istep,", BIC= {:.2f}".format(model.bic))
                    print('Features added: ', selected_features,'\n')
                    print(model.summary())        
            else:            
                # remaining_features.remove(best_candidate)
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                if (data['verbose'] == 'on' or 
                        (data['verbose'] == 'off' and data['drop_nonsig'] == 'off')):
                    print('\nFINAL BACKWARD MODEL')
                    print('Best features: ', selected_features,'\n')
                    print(model.summary())
                break            

        if data['drop_nonsig'] == 'on':
    
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
                    print('\nFINAL MODEL AFTER ELIMINATING NON-SIGNIFICANT PREDICTORS')
                    print('Best features: ', selected_features,'\n')
                    print(model.summary())
                    break
    
    return selected_features, model

