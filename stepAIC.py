# -*- coding: utf-8 -*-

__version__ = "1.0.1"


def stepAIC(X, y, **kwargs):

    """
    PURPOSE
    Stepwise linear regression to minimize AIC and/or eliminate non-signficant predictors

    REQUIRED INPUTS (X and y should have same number of rows and only contain real numbers)
    X = pandas dataframe of the candidate independent variables (as many columns of data as needed)
    y = pandas dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        direction= 'forward', 'both', 'bidirectional', or 'backward', where
            'forward': 
                1) Start with no predictors in the model
                2) Add the predictor with the lowest p-value (i.e., the most significant)
                3) Keep adding predictors as long as they improve the model (reduce AIC)
            'bidirectional':
                1) Start with no predictors in the model
                2) Add predictors like in forward selection but also 
                    check if any previously added predictors have 
                    become insignificant and remove them
            'both' (default): 
                Start with 'forward', and then follow with 'bidirectional' step 2 to          
                    check if any previously added predictors have 
                    become insignificant and remove them
            'backward':
                1) Fit a model with all predictors.
                2) Remove the predictor with the highest p-value above p_threshold (e.g. 0.05)
                3) Repeat until all remaining predictors are significant
        p_threshold= threshold p-value to eliminate predictors (default p_threshold= 0.05)                

    NOTE
    Do any necessary/optional cleaning and standardizing of data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column

    REFERENCE
    Adapted from the following article by Ujang Riswanto on medium.com:
    https://ujangriswanto08.medium.com/how-to-perform-stepwise-regression-in-python-using-statsmodels-2d2cda4e900a

    EXAMPLE USAGE
    best_features, best_model = stepAIC(df_X, df_y)
    """

    import statsmodels.api as sm
    
    # Define default values of input data arguments
    defaults = {
        'direction': 'both',
        'p_threshold': 0.05
        }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}
    p_threshold = data['p_threshold']
        
    selected_features = []
    remaining_features = list(X.columns)
    best_aic = float('inf')

    if data['direction'] == 'forward' or data['direction'] == 'both':
    
        # Forward selection to minimize AIC
        istep = 0
        while remaining_features:
            aic_with_candidates = []        
            for candidate in remaining_features:
                model = sm.OLS(y, sm.add_constant(X[selected_features + [candidate]])).fit()
                aic_with_candidates.append((model.aic, candidate))
                if candidate == remaining_features[-1]:
                    istep += 1
                    print('\nFORWARD STEP',istep)
                    print('Features added: ', selected_features + [candidate],'\n')
                    print(model.summary())        
            aic_with_candidates.sort()  # Sort by AIC
            best_new_aic, best_candidate = aic_with_candidates[0]        
            if best_new_aic < best_aic:
                best_aic = best_new_aic
                selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
            else:            
                best_aic = best_new_aic
                selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
                print('\nFINAL FORWARD MODEL')
                print('Best features: ', selected_features,'\n')
                print(model.summary())
                break            
    
    if data['direction'] == 'both' or data['direction'] == 'bidirectional':

        if data['direction'] == 'bidirectional':
            selected_features = []
            remaining_features = list(X.columns)
            best_aic = float('inf')
        
        # Bi-directional selection to eliminate features with p < p_threshold
        while remaining_features or selected_features:
            # Forward step
            forward_candidates = [
                (sm.OLS(y, sm.add_constant(X[selected_features + [f]])).fit().aic, f)
                for f in remaining_features
            ]
            forward_candidates.sort()
            best_forward_aic, best_candidate = forward_candidates[0]
    
            if best_forward_aic < best_aic:
                best_aic = best_forward_aic
                selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
    
            # Backward step
            model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
            p_values = model.pvalues.iloc[1:]  # Ignore intercept
            max_p_value = p_values.max()
    
            if max_p_value > p_threshold:
                worst_feature = p_values.idxmax()
                selected_features.remove(worst_feature)
            else:
                print('\nFINAL BIDIRECTIONAL MODEL')
                print('Best features: ', selected_features,'\n')
                print(model.summary())
                break

    if data['direction'] == 'backward':

        selected_features = list(X.columns)
    
        while len(selected_features) > 0:
            model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
            p_values = model.pvalues.iloc[1:]  # Ignore the intercept
            max_p_value = p_values.max()
            
            if max_p_value > p_threshold:
                worst_feature = p_values.idxmax()
                selected_features.remove(worst_feature)
            else:
                print('\nFINAL BACKWARD MODEL')
                print('Best features: ', selected_features,'\n')
                print(model.summary())
                break
    
    return selected_features, model

