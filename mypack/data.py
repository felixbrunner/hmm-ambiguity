import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import requests
import io
import zipfile


def download_factor_data(freq='D'):
    if freq is 'D':
        # Download Carhartt 4 Factors
        factors_daily = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start='1/1/1900')[0]
        mom = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench', start='1/1/1900')[0]
        factors_daily = factors_daily.join(mom)
        factors_daily = factors_daily[['Mkt-RF','SMB','HML','Mom   ','RF']]
        factors_daily.columns = ['Mkt-RF','SMB','HML','Mom','RF']
        return factors_daily
        
    elif freq is 'M':
        # Download Carhartt 4 Factors
        factors_monthly = web.DataReader("F-F_Research_Data_Factors", "famafrench", start='1/1/1900')[0]
      #  mom = web.DataReader('F-F_Momentum_Factor', 'famafrench', start='1/1/1900')[0] #There seems to be a problem with the data file, fix if mom is needed
      #  factors_monthly = factors_monthly.join(mom)
      #  factors_monthly = factors_monthly[['Mkt-RF','SMB','HML','Mom   ','RF']]
        factors_monthly.index = factors_monthly.index.to_timestamp()
      #  factors_monthly.columns = ['Mkt-RF','SMB','HML','Mom','RF']
        factors_monthly.columns = ['Mkt-RF','SMB','HML','RF']
        factors_monthly.index = factors_monthly.index+pd.tseries.offsets.MonthEnd(0)
        return factors_monthly

    
def download_industry_data(freq='D', excessreturns = True):
    if freq is 'D':
        # Download Fama/French 49 Industries
        industries_daily = web.DataReader("49_Industry_Portfolios_Daily", "famafrench", start='1/1/1900')[0]
        industries_daily[(industries_daily <= -99.99) | (industries_daily == -999)] = np.nan #set missing data to NaN
        industries_daily = industries_daily.rename_axis('Industry', axis='columns')
        if excessreturns is True:
            factors_daily = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start='1/1/1900')[0]
            industries_daily = industries_daily.subtract(factors_daily['RF'], axis=0) #transform into excess returns
        return industries_daily
    
    elif freq is 'M':
        # Download Fama/French 49 Industries
        industries_monthly = web.DataReader("49_Industry_Portfolios", "famafrench", start='1/1/1900')[0]
        industries_monthly[(industries_monthly <= -99.99) | (industries_monthly == -999)] = np.nan #set missing data to NaN
        industries_monthly = industries_monthly.rename_axis('Industry', axis='columns')
        industries_monthly.index = industries_monthly.index.to_timestamp()
        if excessreturns is True:
            factors_monthly = web.DataReader("F-F_Research_Data_Factors", "famafrench", start='1/1/1900')[0]
            factors_monthly.index = factors_monthly.index.to_timestamp()
            industries_monthly = industries_monthly.subtract(factors_monthly['RF'], axis=0) #transform into excess returns
        industries_monthly.index = industries_monthly.index+pd.tseries.offsets.MonthEnd(0)
        return industries_monthly
    
    
def download_25portfolios_data(freq='D', excessreturns = True):
    if freq is 'D':
        # Download Fama/French 49 Industries
        portfolios_daily = web.DataReader("25_Portfolios_5x5_CSV", "famafrench", start='1/1/1900')[0]
        portfolios_daily[(portfolios_daily <= -99.99) | (portfolios_daily == -999)] = np.nan #set missing data to NaN
        if excessreturns is True:
            factors_daily = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start='1/1/1900')[0]
            portfolios_daily = portfolios_daily.subtract(factors_daily['RF'], axis=0) #transform into excess returns
        return portfolios_daily
    
    elif freq is 'M':
        # Download Fama/French 49 Industries
        portfolios_monthly = web.DataReader("25_Portfolios_5x5_Daily_CSV", "famafrench", start='1/1/1900')[0]
        portfolios_monthly[(industries_monthly <= -99.99) | (industries_monthly == -999)] = np.nan #set missing data to NaN
        portfolios_monthly.index = portfolios_monthly.index.to_timestamp()
        if excessreturns is True:
            factors_monthly = web.DataReader("F-F_Research_Data_Factors", "famafrench", start='1/1/1900')[0]
            factors_monthly.index = factors_monthly.index.to_timestamp()
            portfolios_monthly = portfolios_monthly.subtract(factors_monthly['RF'], axis=0) #transform into excess returns
        return portfolios_monthly
        
        
def download_recessions_data(freq='M', startdate='1/1/1900', enddate=dt.datetime.today()):
    USREC_monthly = web.DataReader('USREC', 'fred',start = startdate, end=enddate)
    if freq is 'M':
        return USREC_monthly
        
    if freq is 'D':
        first_day = USREC_monthly.index.min() - pd.DateOffset(day=1)
        last_day = USREC_monthly.index.max() + pd.DateOffset(day=31)
        dayindex = pd.date_range(first_day, last_day, freq='D')
        dayindex.name = 'DATE'
        USREC_daily = USREC_monthly.reindex(dayindex, method='ffill')
        return USREC_daily
    

def find_csv_filenames(path_to_dir, extension = ".csv"): #returns list of filenames with chosen extension in chosen path
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(extension)]


def unique(list): #returns a list of unique values in a list
    array = np.array(list)
    return np.unique(array).tolist()


def add_months(date,delta_months = 1): #returns the date of the first day of the month delta_months months ahead
    for i in range(0,delta_months):
        date = (date.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
    return date


def download_jpy_usd():    
    jpy = web.DataReader('DEXJPUS', 'fred', start = '1900-01-01')
    return jpy

def download_cad_usd():    
    cad = web.DataReader('DEXCAUS', 'fred', start = '1900-01-01')
    return cad


def download_vix():    
    vix = web.DataReader('VIXCLS', 'fred', start = '1900-01-01')
    return vix


def download_goyal_welch_svar():
    url = 'http://www.hec.unil.ch/agoyal/docs/PredictorData2017.xlsx'
    sheet = pd.read_excel(url, sheet_name='Monthly')
    dates = sheet['yyyymm']
    SVAR = pd.DataFrame(sheet['svar'])
    SVAR.index = [(dt.datetime(year = math.floor(date/100),month = date%100,day = 1)+dt.timedelta(days=32)).replace(day=1)-dt.timedelta(days=1) for date in dates]
    return SVAR


def download_pastor_stambaugh_liquidity(): #currently not working
    url = 'https://faculty.chicagobooth.edu/lubos.pastor/research/liq_data_1962_2017.txt'
    #PSLIQ = pd.read_csv(url, sep=' ')
    PSLIQ = pd.read_csv(url, delim_whitespace=True, header=None)
    return PSLIQ
    
    
def download_sadka_liquidity():
    url = 'http://www2.bc.edu/ronnie-sadka/Sadka-LIQ-factors-1983-2012-WRDS.xlsx'
    sheet = pd.read_excel(url, sheet_name='Sheet1')
    dates = sheet['Date']
    SadkaLIQ1 = pd.DataFrame(sheet['Fixed-Transitory'])
    SadkaLIQ1.index = [(dt.datetime(year = math.floor(date/100),month = date%100,day = 1)+dt.timedelta(days=32)).replace(day=1)-dt.timedelta(days=1) for date in dates]
    SadkaLIQ2 = pd.DataFrame(sheet['Variable-Permanent'])
    SadkaLIQ2.index = [(dt.datetime(year = math.floor(date/100),month = date%100,day = 1)+dt.timedelta(days=32)).replace(day=1)-dt.timedelta(days=1) for date in dates]
    return SadkaLIQ1, SadkaLIQ2


def download_manely_kelly_he_intermediary():
    url = 'http://apps.olin.wustl.edu/faculty/manela/hkm/intermediarycapitalrisk/He_Kelly_Manela_Factors.zip'
    filename = 'He_Kelly_Manela_Factors_monthly.csv'
    column1 = 'intermediary_capital_ratio'
    column2 = 'intermediary_capital_risk_factor'
    column3 = 'intermediary_value_weighted_investment_return'
    column4 = 'intermediary_leverage_ratio_squared'
    raw_data = pd.read_csv(zipfile.ZipFile(io.BytesIO(requests.get(url).content)).open(filename))
    Intermediary = pd.DataFrame(raw_data[[column1,column2,column3,column4]]) #HeKellyManela
    dates = raw_data['yyyymm']
    Intermediary.index = [(dt.datetime(year = math.floor(date/100),month = date%100,day = 1)+dt.timedelta(days=32)).replace(day=1)-dt.timedelta(days=1) for date in dates]
    return Intermediary


def download_jln_macro_uncertainty():
    url = 'https://sydney-ludvigson.squarespace.com/s/MacroFinanceUncertainty_202002_update.zip'
    filename = 'MacroUncertaintyToCirculate.csv'
    uncertainty = pd.read_csv(zipfile.ZipFile(io.BytesIO(requests.get(url).content)).open(filename), index_col='Date')
    uncertainty.index = pd.to_datetime(uncertainty.index, format='%b-%y')
    uncertainty.index = pd.DatetimeIndex([dt.datetime(year=i.year,month=i.month+1 if i.month<12 else 1,day=1) for i in uncertainty.index]) + dt.timedelta(days=-1)
    return uncertainty


def download_jln_real_uncertainty():
    url = 'https://sydney-ludvigson.squarespace.com/s/MacroFinanceUncertainty_202002_update.zip'
    filename = 'RealUncertaintyToCirculate.csv'
    uncertainty = pd.read_csv(zipfile.ZipFile(io.BytesIO(requests.get(url).content)).open(filename), index_col='Date')
    uncertainty.index = pd.to_datetime(uncertainty.index, format='%b-%y')
    uncertainty.index = pd.DatetimeIndex([dt.datetime(year=i.year,month=i.month+1 if i.month<12 else 1,day=1) for i in uncertainty.index]) + dt.timedelta(days=-1)
    return uncertainty


def download_jln_financial_uncertainty():
    url = 'https://sydney-ludvigson.squarespace.com/s/MacroFinanceUncertainty_202002_update.zip'
    filename = 'FinancialUncertaintyToCirculate.csv'
    uncertainty = pd.read_csv(zipfile.ZipFile(io.BytesIO(requests.get(url).content)).open(filename), index_col='Date')
    uncertainty.index = pd.to_datetime(uncertainty.index, format='%b-%y')
    uncertainty.index = pd.DatetimeIndex([dt.datetime(year=i.year,month=i.month+1 if i.month<12 else 1,day=1) for i in uncertainty.index]) + dt.timedelta(days=-1)
    return uncertainty