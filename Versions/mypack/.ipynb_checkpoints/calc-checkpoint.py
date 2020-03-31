import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

def calculate_kolmogorov_smirnov_distance(mu0, mu1, sigma0, sigma1):
    if np.isnan(mu0) or np.isnan(mu1) or np.isnan(sigma0) or np.isnan(sigma1):
        KS = np.nan
        
    else:    
        a = 1/(2*sigma0**2) - 1/(2*sigma1**2)
        b = mu1/sigma1**2 - mu0/sigma0**2
        c = mu0**2/(2*sigma0**2) - mu1**2/(2*sigma1**2) - np.log(sigma1/sigma0)
    
        if a == 0:
            if b == 0:
                KS = 0
            else:
                x_sup = -c/b
                KS = abs(sp.stats.norm.cdf(x_sup, mu0, sigma0) - sp.stats.norm.cdf(x_sup, mu1, sigma1))
        else:
            x_1 = (-b + (b**2-4*a*c)**0.5) / (2*a)
            x_2 = (-b - (b**2-4*a*c)**0.5) / (2*a)
        
            KS_1 = abs(sp.stats.norm.cdf(x_1, mu0, sigma0) - sp.stats.norm.cdf(x_1, mu1, sigma1))
            KS_2 = abs(sp.stats.norm.cdf(x_2, mu0, sigma0) - sp.stats.norm.cdf(x_2, mu1, sigma1))
        
            KS = max(KS_1,KS_2)
    
    return KS


def sort_portfolios(returns, ranking_variable, n_portfolios, lags=1, return_assets=False):
    # align periods
    sorting_variable = ranking_variable.shift(lags)
    
    # set up parameters
    [t,n] = returns.shape
    include = returns.notna() & sorting_variable.notna()
    n_period = include.sum(axis=1)

    # sort assets
    returns_include = returns[include]
    sorting_variable[~include] = np.nan
    cutoff_ranks = np.dot(n_period.values.reshape(t,1)/n_portfolios,np.arange(n_portfolios+1).reshape(1,n_portfolios+1)).round()
    asset_ranks = sorting_variable.rank(axis=1)
    
    # set up output frames
    portfolio_returns = pd.DataFrame(index=returns.index,columns=range(1,n_portfolios+1))
    portfolio_assets = pd.DataFrame(index=returns.index,columns=range(1,n_portfolios+1))
    portfolio_mapping = pd.DataFrame(index=returns.index, columns=returns.columns)
    
    # calculate outputs
    for i_portfolio in range(0,n_portfolios):
        lower = cutoff_ranks[:,i_portfolio].reshape(t,1).repeat(n, axis=1)
        upper = cutoff_ranks[:,i_portfolio+1].reshape(t,1).repeat(n, axis=1)
        portfolio_returns[i_portfolio+1] = returns_include[(asset_ranks>lower) & (asset_ranks<=upper)].mean(axis=1)
        portfolio_assets[i_portfolio+1] = ((asset_ranks>lower) & (asset_ranks<=upper)).sum(axis=1)
        portfolio_mapping[(asset_ranks>lower) & (asset_ranks<=upper)] = i_portfolio
    
    # outputs
    if return_assets == False:
        return portfolio_returns
    else:
        return portfolio_returns, portfolio_assets, portfolio_mapping

    
    
def double_sort_portfolios(returns, ranking_variable_1, ranking_variable_2, n_portfolios_1, n_portfolios_2, lags_1=1, lags_2=1, return_assets=False):
    # identify missing values
    exclude = returns.isna() | ranking_variable_1.shift(lags_1).isna() | ranking_variable_2.shift(lags_2).isna()
    returns[exclude] = np.nan
    
    # first sort
    portfolio_mapping_1 = sort_portfolios(returns, ranking_variable_1, n_portfolios_1, lags_1, return_assets=True)[2]
    
    # second sorts
    portfolio_mapping_2 = pd.DataFrame(0, index=portfolio_mapping_1.index, columns=portfolio_mapping_1.columns)
    for i_portfolio_2 in range(0,n_portfolios_2):
        subportfolio_returns = returns[portfolio_mapping_1 == i_portfolio_2]
        portfolio_mapping_2 += (sort_portfolios(subportfolio_returns, ranking_variable_2, n_portfolios_2, lags_2, return_assets=True)[2]).fillna(0)
    portfolio_mapping_2[exclude] = np.nan
    
    # combined sort
    portfolio_mapping = portfolio_mapping_1*n_portfolios_1 + portfolio_mapping_2
    
    # set up output frames
    portfolio_returns = pd.DataFrame(index=returns.index,columns=[str(i_portfolio_1+1)+','+str(i_portfolio_2+1) for i_portfolio_1 in range(0,n_portfolios_1) for i_portfolio_2 in range(0,n_portfolios_2)])
    portfolio_assets = pd.DataFrame(index=returns.index,columns=[str(i_portfolio_1+1)+','+str(i_portfolio_2+1) for i_portfolio_1 in range(0,n_portfolios_1) for i_portfolio_2 in range(0,n_portfolios_2)])
    
    # calculate outputs
    for i_portfolio_all in range(0,n_portfolios_1*n_portfolios_2):
        portfolio_returns.iloc[:,i_portfolio_all] = returns[portfolio_mapping == i_portfolio_all].mean(axis=1)
        portfolio_assets.iloc[:,i_portfolio_all] = (portfolio_mapping == i_portfolio_all).sum(axis=1)
        
    # outputs
    if return_assets == False:
        return portfolio_returns
    else:
        return portfolio_returns, portfolio_assets, portfolio_mapping
    
    

def double_sort_portfolios_simultaneously(returns, ranking_variable_1, ranking_variable_2, n_portfolios_1, n_portfolios_2, lags_1=1, lags_2=1, return_assets=False):
    # identify missing values
    exclude = returns.isna() | ranking_variable_1.shift(lags_1).isna() | ranking_variable_2.shift(lags_2).isna()
    returns[exclude] = np.nan
    
    # first sort
    portfolio_mapping_1 = sort_portfolios(returns, ranking_variable_1, n_portfolios_1, lags_1, return_assets=True)[2]
    
    # second sorts
    portfolio_mapping_2 = sort_portfolios(returns, ranking_variable_2, n_portfolios_2, lags_2, return_assets=True)[2]
    
    # combined sort
    portfolio_mapping = portfolio_mapping_1*n_portfolios_1 + portfolio_mapping_2
    
    # set up output frames
    portfolio_returns = pd.DataFrame(index=returns.index,columns=[str(i_portfolio_1+1)+','+str(i_portfolio_2+1) for i_portfolio_1 in range(0,n_portfolios_1) for i_portfolio_2 in range(0,n_portfolios_2)])
    portfolio_assets = pd.DataFrame(index=returns.index,columns=[str(i_portfolio_1+1)+','+str(i_portfolio_2+1) for i_portfolio_1 in range(0,n_portfolios_1) for i_portfolio_2 in range(0,n_portfolios_2)])
    
    # calculate outputs
    for i_portfolio_all in range(0,n_portfolios_1*n_portfolios_2):
        portfolio_returns.iloc[:,i_portfolio_all] = returns[portfolio_mapping == i_portfolio_all].mean(axis=1)
        portfolio_assets.iloc[:,i_portfolio_all] = (portfolio_mapping == i_portfolio_all).sum(axis=1)
        
    # outputs
    if return_assets == False:
        return portfolio_returns
    else:
        return portfolio_returns, portfolio_assets, portfolio_mapping
    
    
    
def standardise_dataframe(df, ax=0):
    df = df.subtract(df.mean(axis=ax), axis=ax)
    df = df.divide(df.std(axis=ax), axis=ax)
    return df


def total_return_from_returns(returns): #returns total return of a return series
    return (returns + 1).prod(skipna=False) - 1


def fill_inside_na(df, meth = 'zero', lim = None): #fills missing values in the middle of a series (but not at beginning and end)
    nans = pd.DataFrame(df.bfill().isna() | df.ffill().isna())
    if meth is 'zero':
        df = df.fillna(0)
    else:
        df = df.fillna(method = meth, limit = lim)
    df[nans] = np.nan
    return df


def realised_volatility_from_returns(returns):
    return ((returns**2).mean(skipna=False))**0.5


def normal_central_moment(sigma, moment):

    '''Central moments of a normal distribution with any mean.
    Note that the first input is a standard deviation, not a variance.'''

    if moment % 2 == 1:
        #odd moments of a normal are zero
        normal_moment = 0 
    else:
        #even moments are given by sigma^n times the double factorial
        normal_moment = sigma**moment * sp.special.factorialk(moment-1, 2) 
    return normal_moment


def calculate_steady_state_probabilities(transition_matrix):
    dim = np.array(transition_matrix).shape[0]
    q = np.c_[(transition_matrix-np.eye(dim)),np.ones(dim)]
    QTQ = np.dot(q, q.T)
    steady_state_probabilities = np.linalg.solve(QTQ,np.ones(dim))
    return steady_state_probabilities


def iterate_markov_chain(state_probabilities, transition_matrix, steps):
    new_state = np.dot(state_probabilities, np.linalg.matrix_power(transition_matrix, steps))
    return new_state


def calculate_binary_entropy(probability):
    entropy = -1 * (probability*np.log2(probability) + (1-probability)*np.log2(1-probability))
    return abs(entropy)


def calculate_columnwise_autocorrelation(df, lag=1):
    autocorr = pd.Series(index=df.columns)
    for column in df.columns:
        autocorr[column] = df[column].astype('float64').autocorr(lag=lag)
    return autocorr


def export_df_to_latex(df, filename='file.tex', **kwargs):
    if filename[-4:] != '.tex':
        filename += '.tex'
    df.to_latex(buf=filename, multirow=False, multicolumn_format ='c', na_rep='', escape=False, **kwargs)
    
    
def calculate_shannon_entropy(vector):
    '''
    calculate the shannon entropy measure from a vector of probabilities
    '''
    
    n = len(vector)
    entropy = 0
    for v in vector:
        if v == 0:
            pass
        else:
            entropy += v*np.log(v)/np.log(n)
    return abs(entropy)


def shrink_outliers(series, alpha=1.96, lamb=1, plot=False):
    '''
    This function shrinks outliers in a series towards the threshold values.
    The parameter alpha defines the threshold values as a multiple of one sample standard deviation.
    The parameter lamb defines the degree of shrinkage of outliers towards the thresholds.
    
    The transformation is as follows:
    if the z score is inside the thresholds f(x)=x
    if it is above the upper threshold f(x)=1+1/lamb*ln(x+(1-lamb)/lamb)-1/lamb*ln(1/lamb)
    if it is below the lower threshold f(x)=-1-1/lamb*ln(-x+(1-lamb)/lamb)+1/lamb*ln(1/lamb)
    '''
    
    z_scores = (series-series.mean())/series.std()
    adjusted_scores = z_scores/alpha
    adjusted_scores[adjusted_scores.values>1] = 1+1/lamb*np.log(adjusted_scores[adjusted_scores.values>1]+(1-lamb)/lamb)-1/lamb*np.log(1/lamb)
    adjusted_scores[adjusted_scores.values<-1] = -1-1/lamb*np.log((1-lamb)/lamb-adjusted_scores[adjusted_scores.values<-1])+1/lamb*np.log(1/lamb)
    new_z_scores = adjusted_scores*alpha
    new_series = new_z_scores*series.std()+series.mean()
    
    if plot:
        fig, ax = plt.subplots(1,1,figsize=[10,6])
        corrected = (adjusted_scores>1) | (adjusted_scores<-1)
        ax.scatter(series.loc[corrected],new_series.loc[corrected], marker='x', color='r', s=50, label='Transformed observations')
        ax.scatter(series.loc[~corrected],new_series.loc[~corrected], marker='+', color='g', s=30, label='Unaffected observations')
        ax.axhline(alpha*series.std()+series.mean(), color='grey', linewidth=1, zorder=0, label='Threshold')
        ax.axhline(-alpha*series.std()+series.mean(), color='grey', linewidth=1, zorder=0)
        #ax.axvline(series.mean()+series.std(), color='k', linewidth=1, zorder=0)
        #ax.axvline(series.mean()-series.std(), color='k', linewidth=1, zorder=0)
        ax.plot(np.linspace(*ax.get_ylim()),np.linspace(*ax.get_ylim()), color='k', linewidth=1, linestyle=':', zorder=0, label='45-degree line')
        ax.set_xlabel('Observed data')
        ax.set_ylabel('Transformed data')
        ax.set_title('Outlier Transformation (α='+'{:0.3f}'.format(alpha)+', λ='+'{:0.1f}'.format(lamb)+')')
        ax.legend()
        
        return new_series, fig
    
    else:
        return new_series


def get_unique_values_from_list_of_lists(input_list):
    full_list = [j for i in input_list for j in i]
    short_list = list(dict.fromkeys(full_list))
    return short_list


def create_regression_summary_column(regression, t_stats=True, add_outputs=[]):
    
    '''
    Creates a pandas series object that contains the formatted results of a statsmodels regression for academic presentation.
    Additional outputs can be:
        - R-squared
        - N'
        - Adj R-squared'
        - AIC
        - BIC
        - LL
        - F-stat
        - P(F-stat)
        - DF (model)
        - DF (residuals)
        - MSE (model)
        - MSE (residuals)
        - MSE (total)
    '''
    
    summary = pd.Series(index = pd.MultiIndex.from_product([reg.params.index,['coeff','t-stat']]))
    stars = (abs(reg.tvalues)>1.645).astype(int) + \
            (abs(reg.tvalues)>1.96).astype(int) + \
            (abs(reg.tvalues)>2.58).astype(int)
    summary.loc[(regression.params.index,'coeff')] = reg.params.map(lambda x: '%.4f' %x).values + \
                                                         stars.map(lambda x: x*'*').values
    
    if t_stats:
        summary.loc[(regression.params.index,'t-stat')] = regression.tvalues.map(lambda x: '(%.4f)' %x).values
    else:
        summary.index = pd.MultiIndex.from_product([regression.params.index,['coeff','s.e.']])
        summary.loc[(regression.params.index,'s.e.')] = regression.bse.map(lambda x: '(%.4f)' %x).values
    
    output_dict = {'R-squared': 'regression.rsquared',
                 'N': 'regression.nobs',
                 'Adj R-squared': 'regression.rsquared_adj',
                 'AIC': 'regression.aic',
                 'BIC': 'regression.bic',
                 'LL': 'regression.llf',
                 'F-stat': 'regression.fvalue',
                 'P(F-stat)': 'regression.f_pvalue',
                 'DF (model)': 'regression.df_model',
                 'DF (residuals)': 'regression.df_resid',
                 'MSE (model)': 'regression.mse_model',
                 'MSE (residuals)': 'regression.mse_resid',
                 'MSE (total)': 'regression.mse_total',
                 }
    
    for out in add_outputs:
        if out in ['N','DF (model)','DF (residuals)']:
            summary[(out,'')] = "{:.0f}".format(eval(output_dict[out]))
        else:
            summary[(out,'')] = "{:.4f}".format(eval(output_dict[out]))
    
    return summary