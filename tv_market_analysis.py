# Does TV channel view help to sign-up more?
# Signned up user has ID, prior exposure of certain TV channel (3 months)
# In total: 5000 users signed up for 3 months

# load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

# parameter names
tv_name = 'tv_campaigns.csv'
signup_name = 'user_signup_data.csv'
start_date = '2017-01-01'; end_date = '2017-04-01'; frequency = 'D'
n_channel = 4
n_dates = 91
channels = ['cpc', 'organic', 'affiliate', 'social']

# import csv dataframe
def read_csv_to_dataframe(csv_name):
    headers = channels
    csv_content = pd.read_csv(csv_name, sep=',')
    return csv_content

# dates in pandas dataframe
def date_into_pandas(start_date, end_date, frequency):
    dates_pd = pd.DataFrame({'Date':pd.date_range(start_date, end_date, freq=frequency)})
    return dates_pd#, dates_np

# organize the data according to date
def reorg_signup(csv_content, dates_pd):
    # change dates into numpy array (to match with numbers later)
    dates_np = dates_pd.Date.dt.strftime('%Y-%m-%d').astype(str)
    n_dates = len(dates_np)

    # predefine results
    customer_channel = np.zeros([n_dates])
    signup_pandas = dates_pd
    for channel in range(n_channel)
        for x_date in range(n_dates):
            channel_one_date = csv_content[csv_content['signup_date'] == dates_np[x_date]].iloc[:,channel+1].sum()
            customer_channel[x_date] = channel_one_date
        signup_pandas[channels[channel]] = customer_channel
    return signup_pandas

# plot the sign-up result
def plot_signup(dates_pd, customer_channel):
    fig = plt.plot()
    plt.plot(dates_pd, customer_channel, '.')
    plt.show()

# time series (in sliding window)
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(7).mean()#, window=12)
    rolstd = timeseries.rolling(7).std()#pd.rolling_std(timeseries, window=12)

    fig = plt.plot()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()#(block=False)

# linear regression

#==================================================
#=== make the estimate of doubling the TV spending


# linear regression of the TV spending vs signup
def main():
    # import csv
    tv_most_spent = read_csv_to_dataframe(tv_name)
    signup_content = read_csv_to_dataframe(signup_name)

    # reorganize data
    dates_pd = date_into_pandas(start_date, end_date, frequency)
    signup_channel = reorg_signup(signup_content, dates_pd)

    # plot signup (each channel) over time
    signup_channel_only = signup_channel.iloc[:,1:4]
    dates_only = signup_channel.iloc[:,0]
    plot_signup(dates_only, signup_channel_only)
    test_stationarity(signup_channel_only)
if __name__ == "__main__":
    main()
