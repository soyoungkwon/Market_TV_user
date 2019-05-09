# Does TV channel view help to sign-up more?
# Signned up user has ID, prior exposure of certain TV channel (3 months)
# In total: 5000 users signed up for 3 months

# load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
import copy
from scipy.stats import sem
from scipy.stats import ttest_ind
# from statsmodels.tsa.stattools import adfuller

# parameter names
tv_name = 'tv_campaigns.csv'
signup_name = 'user_signup_data.csv'
start_date = '2017-01-01'; end_date = '2017-04-01'; frequency = 'D'
n_channel = 4; n_dates = 91; n_week = 12; n_day_week = 7; avg_window = 7
campaign_value = 1; ymin = 60; ymax = 150
tv_turnon = 0
channels = ['cpc', 'organic', 'affiliate', 'social']
colors = ['red', 'blue', 'green', 'yellow']
all_label = copy.copy(channels);    all_label.extend(['campaign'])
mon2sun = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']#, 'Sun']
no_tv_only = 1
# import csv dataframe
def read_csv_to_dataframe(csv_name):
    csv_content = pd.read_csv(csv_name, sep=',')
    return csv_content

# dates in pandas dataframe
def date_into_pandas(start_date, end_date, frequency):
    dates_pandas = pd.DataFrame({'Date':pd.date_range(start_date, end_date, freq=frequency)})
    return dates_pandas

# signup result in pandas dataframe
# result ==> df['date','channel1', 'channel2', ... 'tv','dayofweek']
def reorg_signup(signup_content, dates_pandas, tv_campaign):
    # change dates into numpy array (to match with numbers later)
    dates_numpy = dates_pandas.Date.dt.strftime('%Y-%m-%d').astype(str)
    # extract day of week (e.g., Monday)
    day_week    = dates_pandas.Date.dt.day_name()

    # predefine results: signup from one channel, then all channel
    n_dates = len(dates_numpy)
    customer_one_channel_all_dates = np.zeros([n_dates])
    signup_pandas = dates_pandas.copy()
    for ch in range(n_channel): # each channel
        for x_date in range(n_dates): # each date
            # extract only one date --> extract only one channel and count
            pandas_one_date_only = signup_content[signup_content['signup_date'] == dates_numpy[x_date]]
            one_channel_count = (pandas_one_date_only.iloc[:,ch+1].sum())# ch+1 -> due to date column
            customer_one_channel_all_dates[x_date] = one_channel_count
        signup_pandas[channels[ch]] = customer_one_channel_all_dates

    # predefine result: tv_campaign_pd --> add it to the main pandas (signup_pandas)
    tv_campaign_pandas = np.zeros(n_dates)
    for date in range(n_dates): # each date check whether it matches with any list in tv_campaign
        tv_campaign_list = tv_campaign['date'].tolist()
        if any(dates_numpy[date] == tv_c for tv_c in tv_campaign_list):
            tv_campaign_pandas[date] = campaign_value

    # add few columns
    signup_pandas['tv'] = tv_campaign_pandas
    signup_pandas['dayofweek'] = day_week
    # return final signup with all channel and campaign
    return signup_pandas

def plot_tv_campaign_signup_channel(signup_pandas):
    timeseries = signup_pandas[channels]
    dates_only = signup_pandas['Date']
    #Determing rolling statistics
    fig = plt.figure(figsize=(8,3))
    for channel in range(n_channel):
        time_one_channel = timeseries.iloc[:,channel]
        orig = plt.plot(dates_only, time_one_channel, '.', color = colors[channel], label='Original')
    #Plot rolling statistics:
    for channel in range(n_channel):
        time_one_channel = timeseries.iloc[:,channel]
        rolmean = time_one_channel.rolling(avg_window).mean()
        rolstd = time_one_channel.rolling(avg_window).std()
        mean = plt.plot(dates_only, rolmean, color = colors[channel], label='Rolling Mean')
        std = plt.plot(dates_only, rolstd, '--', color='grey', label = 'Rolling Std')

    # find index for tv_campaign dates
    dates_index = dates_only[signup_pandas['tv']==1].index.tolist()

    # plot each vertical line separately
    for i in range(8):
        x=(dates_index[i]*np.ones(60)).tolist(); lineY = range(60)
        plt.plot(dates_only[x], lineY, 'pink')
    plt.legend(channels)
    plt.show()

# Sum of the channel
def plot_signup_sum(signup_pandas):
    signup_allchannel = np.sum(signup_pandas[channels], axis=1)
    plt.plot(signup_allchannel, '.')

# t-test of the signup on the difference between tv campaign day vs no campaign day
def signup_ttest(signup_pandas):
    tv_on = np.sum(signup_pandas[signup_pandas['tv']==1][channels], axis=1)
    tv_off = np.sum(signup_pandas[signup_pandas['tv']==0][channels], axis=1)
    [T, p] = ttest_ind(tv_on, tv_off)
    # plot bar (TV campaign on vs off days)
    x = np.arange(2);
    tv_both = pd.concat([tv_on, tv_off], axis = 1)
    plt.bar(x, tv_both.mean(),yerr = tv_both.sem())
    plt.xticks(x, ['TV on', 'TV off'])
    plt.show()
    print('TV campaign day vs no campaign day T: %.2f P: %2f' %(T, p))
    print(tv_on.mean()/tv_off.mean())

# pie chart of each channel users
def pie_chart_signup(signup_pandas, channels):
    signup_all = signup_pandas.iloc[:,1:5]
    # all day
    signup_allcount = signup_all.sum().sum()
    signup_channel_count = signup_all.sum()
    signup_percentage = (signup_channel_count/signup_allcount*100).tolist()
    fig = plt.plot()
    plt.pie(signup_percentage, labels = channels)
    plt.show()

# plot weekly signup
def week_summary(signup_pandas, channels):
    # each week summary (all channels, each channel)
    signup_week_summary = np.zeros(n_week)
    signup_week_channel = np.zeros([n_week, n_channel])
    for w in range(n_week):
        signup_week = signup_pandas.iloc[w:w+7,1:n_channel+1].sum()
        signup_week_summary[w] = signup_week.sum()
        signup_week_channel[w,:] = signup_week
        plt.title(('total signup = ' + str(signup_week.sum())))
    plt.subplot(2,1,1);    plt.title('weekly total')
    plt.plot(signup_week_summary, '--*')
    plt.subplot(2,1,2);   plt.title('weekly channel')
    plt.plot(signup_week_channel, '--*')
    plt.show()

# plot day signup (e.g., Monday)
def day_summary(signup_pandas, channels):
    if no_tv_only: signup_pandas = signup_pandas[signup_pandas['tv']==0]
    # check dayofweek is Monday and calculate mean
    mon2sun_summary = np.zeros([n_day_week, n_channel])
    tv_onoff=0
    for d in range(n_day_week):
        signup_monday = signup_pandas[signup_pandas['dayofweek'].str.contains(mon2sun[d])]
        mon2sun_summary[d, : ] = signup_monday.iloc[:, 1:n_channel+1].mean()
        plt.title('Mon - Sun')
        plt.bar(range(n_day_week), np.sum(mon2sun_summary, axis=1), yerr = sem(mon2sun_summary, axis=1)) # started from 0(sun), so must change
        plt.xticks(range(n_day_week), mon2sun);    plt.ylim([70, 120])
    plt.show()

def main():
    # import csv
    tv_campaign = read_csv_to_dataframe(tv_name)
    signup_content = read_csv_to_dataframe(signup_name)

    # all data as one pandas (date--> pandas, signup--> add to )
    dates_pandas = date_into_pandas(start_date, end_date, frequency)
    signup_pandas = reorg_signup(signup_content, dates_pandas, tv_campaign)

    # overview of the signup users
    plot_tv_campaign_signup_channel(signup_pandas)
    pie_chart_signup(signup_pandas, channels)

    # plot weekly, day (monday)
    week_summary(signup_pandas, channels)
    day_summary(signup_pandas, channels)
    print(signup_pandas[signup_pandas['tv']==1])
    # statistics
    signup_ttest(signup_pandas)

if __name__ == "__main__":
    main()
