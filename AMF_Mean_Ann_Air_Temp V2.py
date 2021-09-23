# -*- coding: utf-8 -*-
"""
Created (date here)
By Lily Klinek and Aadiraj Batlaw

"""

import pandas as pd 
import numpy as np 
import os
from scipy import optimize
import matplotlib.pyplot as plt
import datetime as dt
import plotly.express as px
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew, kurtosis
import statsmodels.api as sm


def cleaning(name, colname, colname2):
    '''
    name: site name (string)
    Description: removes unnecessary columns, adds air and soil temp units, reformats date, drops Nan columns
    '''
    # reading csv file
    pathname = os.getcwd() + name
    df = pd.read_csv(pathname, skiprows=2)
    
    # removing unecessary columns -- all except time, air temp, and soil temp
    column_names = ["TIMESTAMP_START",colname, colname2]
    if set(column_names).issubset(df.columns):
        
        #renaming columns to include temp unit (deg c)
        df = df[column_names].rename(columns= {colname:"TA (deg C)", colname2:"TS (deg C)"})
           
    #reformat timestamp to datetime
    df["TIMESTAMP_START"] = pd.to_datetime(df["TIMESTAMP_START"], format='%Y%m%d%H%M')
    
    #drop columns with missing data
    df = df.dropna() #drop columns with NaN
    missing_data = -9999.0 #missing data flag
    df = df[~df.eq(missing_data).any(1)].reset_index(drop = True) 
    
    return df


# clean each site
tonzi_df = cleaning('/AMF_SITE_DATA/AMF_US-Ton_BASE_HH_11-5.csv', colname = 'TA_PI_F_1_1_1', colname2 = 'TS_PI_F_1_3_A')
bartlett_df = cleaning('/AMF_SITE_DATA/AMF_US-Bar_BASE_HH_5-5.csv', colname = 'TA_PI_F_1_1_1', colname2 = 'TS_PI_F_1_2_1')
morgan_df = cleaning('/AMF_SITE_DATA/AMF_US-MMS_BASE_HR_17-5.csv', colname = 'TA_1_1_1', colname2 = 'TS_2_1_1')
umb_df = cleaning('/AMF_SITE_DATA/AMF_US-UMB_BASE_HH_15-5.csv', colname = 'TA', colname2 = 'TS_1_1_1')
willow_df = cleaning('/AMF_SITE_DATA/AMF_US-WCr_BASE_HH_19-5.csv', colname = 'TA_PI_F_1_1_1', colname2 = 'TS_PI_F_1_1_1')
chestnut_df = cleaning('/AMF_SITE_DATA/AMF_US-ChR_BASE_HH_2-1.csv', colname = 'TA', colname2 = 'TS_1')
ozark_df = cleaning('/AMF_SITE_DATA/AMF_US-MOz_BASE_HH_8-5.csv', colname = 'TA_1_1_1', colname2 = 'TS_1_1_1')
harvard_df = cleaning('/AMF_SITE_DATA/AMF_US-Ha1_BASE_HR_15-5.csv', colname = 'TA_PI_F_1_1_1', colname2 = 'TS_PI_1')
# freeman_df = cleaning('/AMF_SITE_DATA/AMF_US-FR3_BASE_HH_1-1.csv', colname = 'TA', colname2 = 'TS_2')
saska_df = cleaning('/AMF_SITE_DATA/AMF_CA-Oas_BASE_HH_1-1.csv', colname = 'TA', colname2 = 'TS_2')
# univkansas_df = cleaning('/AMF_SITE_DATA/AMF_US-xUK_BASE_HH_2-5.csv', colname = 'TA_1_1_1', colname2 = 'TS_1_5_1')
vaira_df = cleaning('/AMF_SITE_DATA/AMF_US-Var_BASE_HH_15-5.csv', colname = 'TA_PI_F', colname2 = 'TS_PI_F_1_3_A')
turkey_df = cleaning('/AMF_SITE_DATA/AMF_CA-TPD_BASE_HH_2-5.csv', colname = 'TA_PI_F', colname2 = 'TS_PI_F_3')


 #resample datasets to get daily means
def daily_ta(df):
        df = df.set_index(["TIMESTAMP_START"])
        # std_ta = df[colname].resample('D').std()
        df = df.resample('D').mean()
        df.reset_index(inplace = True)
        df["year"] = df['TIMESTAMP_START'].apply(lambda x: x.year)
        df = df.set_index(["TIMESTAMP_START"])
        return df

# saving daily TA / TS csv files        
tonzi_daily = daily_ta(tonzi_df)
tonzi_daily.to_csv("AMF_Tonzi_Daily_TA_TS.csv")

bartlett_daily = daily_ta(bartlett_df)
bartlett_daily.to_csv("AMF_Bart_Daily_TA_TS.csv")

morgan_daily = daily_ta(morgan_df)
morgan_daily.to_csv("AMF_Morgan_Daily_TA_TS.csv")

umb_daily = daily_ta(umb_df)
umb_daily.to_csv("AMF_UMB_Daily_TA_TS.csv")

willow_daily = daily_ta(willow_df)
willow_daily.to_csv("AMF_Willow_Daily_TA_TS.csv")

chestnut_daily = daily_ta(chestnut_df)
chestnut_daily.to_csv("AMF_Chestnut_Daily_TA_TS.csv")

ozark_daily = daily_ta(ozark_df)
ozark_daily.to_csv("AMF_Ozark_Daily_TA_TS.csv")

harvard_daily = daily_ta(harvard_df)
harvard_daily.to_csv("AMF_Harvard_Daily_TA_TS_attempt2.csv")

# freeman_daily = daily_ta(freeman_df)
# freeman_daily.to_csv("AMF_Freeman_Daily_TA_TS.csv")

saska_daily = daily_ta(saska_df)
saska_daily.to_csv("AMF_Saska_Daily_TA_TS.csv")

# univkansas_daily = daily_ta(univkansas_df)
# univkansas_daily.to_csv("AMF_UnivKansas_Daily_TA_TS.csv")

vaira_daily = daily_ta(vaira_df)
vaira_daily.to_csv("AMF_Vaira_Daily_TA_TS.csv")

turkey_daily = daily_ta(turkey_df)
turkey_daily.to_csv("AMF_Turkey_Daily_TA_TS.csv")




## recursive filter for mean annual air temp
# def maat(df):
   # temp_start = list(df['TA (deg C)'].dropna())
   # alpha = np.exp(-1/730)
   # _mean = temp_start.pop(0)
   # def helper1(temp, _mean):
    #    if temp == []:
    #        return _mean
    #    else:
    #        day = temp.pop(0)
    #        _mean = (1 - alpha) * _mean + alpha * day
    #        return helper1(temp, _mean)
        
   # return helper1(temp_start, _mean)

# testing recursive filter to get mean annual air temp
# maat(tonzi_daily[2005:2006])
# print()




# yearly resampling to get mean annual air temp
def yearly_ta(df):
    df["year"] = df['TIMESTAMP_START'].apply(lambda x: x.year)
    df['TA (deg C)'] = df['TA (deg C)'].mean()
    #df = df.set_index(["TIMESTAMP_START"])
    #df = df.resample().mean()
     # add column for year
    return df


morgan_yearly = yearly_ta(morgan_df)
bartlett_yearly = yearly_ta(bartlett_df)
tonzi_yearly = yearly_ta(tonzi_df)
umb_yearly = yearly_ta(umb_df)
willow_yearly = yearly_ta(willow_df)
chestnut_yearly = yearly_ta(chestnut_df)
ozark_yearly = yearly_ta(ozark_df)
harvard_yearly = yearly_ta(harvard_df)
# freeman_yearly = yearly_ta(freeman_df)
saska_yearly = yearly_ta(saska_df)
# univkansas_yearly = yearly_ta(univkansas_df)
vaira_yearly = yearly_ta(vaira_df)
turkey_yearly = yearly_ta(turkey_df)



# reading and cleaning PhenoCam data

def cleanPhenoCam(name):
    '''
    name: site name (string)
    Description: reads csv file, removes unnecessary columns
    '''
    # reading csv file
    pathname = os.getcwd() + name
    df = pd.read_csv(pathname, skiprows=16)
    
    # removing unecessary columns -- 50% version
    column_names = ["direction","transition_50"]
    if set(column_names).issubset(df.columns):
        
         df = df[column_names].rename(columns= {"direction":"direction","transition_50":"transition"})
         df.sort_values(by=['transition'], inplace=True)
         
     # removing unecessary columns -- 25% version
    # column_names = ["direction","transition_25"]
    # if set(column_names).issubset(df.columns):
        
         # df = df[column_names].rename(columns= {"direction":"direction","transition_25":"transition"})
         # df.sort_values(by=['transition'], inplace=True)
         
    # removing unecessary columns -- 10% version
    # column_names = ["direction","transition_10"]
    # if set(column_names).issubset(df.columns):
        
         # df = df[column_names].rename(columns= {"direction":"direction","transition_10":"transition"})
         # df.sort_values(by=['transition'], inplace=True)
         
    # drop columns with missing data
    df = df.dropna() #drop columns with NaN
    missing_data = -9999.0 #missing data flag
    df = df[~df.eq(missing_data).any(1)].reset_index(drop = True) 
    
    # getting only rising
    df = df[df['direction'] == 'rising']
    
    # convert to timestamp
    df['transition'] = pd.to_datetime(df['transition'])
    
    # add column for year
    df["year"] = df['transition'].apply(lambda x: x.year)
    
    # grouping by direction
    df = df.groupby(['direction','year']).first()
    
    # reset index
    df.reset_index(inplace = True)
    
    # set index
    df.set_index(['transition'], inplace = True)
    
    return df


tonzi_greenup = cleanPhenoCam('/PhenoCam_V2_1674/data/tonzi_DB_1000_3day_transition_dates.csv')
bartlett_greenup = cleanPhenoCam('/PhenoCam_V2_1674/data/bartlettir_DB_1000_3day_transition_dates.csv')
morgan_greenup = cleanPhenoCam('/PhenoCam_V2_1674/data/morganmonroe_DB_1000_3day_transition_dates.csv')
umb_greenup = cleanPhenoCam('/PhenoCam_V2_1674/data/umichbiological_DB_1000_3day_transition_dates.csv')
willow_greenup = cleanPhenoCam('/PhenoCam_V2_1674/data/willowcreek_DB_1000_3day_transition_dates.csv')
chestnut_greenup = cleanPhenoCam('/PhenoCam_V2_1674/data/oakridge1_DB_2000_3day_transition_dates.csv')
ozark_greenup = cleanPhenoCam('/PhenoCam_V2_1674/data/missouriozarks_DB_1000_3day_transition_dates.csv')
harvard_greenup = cleanPhenoCam('/PhenoCam_V2_1674/data/harvard_DB_1000_3day_transition_dates.csv')
saska_greenup = cleanPhenoCam('/PhenoCam_V2_1674/data/canadaOA_DB_1000_3day_transition_dates.csv')
# univkansas_greenup = cleanPhenoCam('/PhenoCam_V2_1674/data/
vaira_greenup = cleanPhenoCam('/PhenoCam_V2_1674/data/vaira_GR_1000_3day_transition_dates.csv')
turkey_greenup= cleanPhenoCam('/PhenoCam_V2_1674/data/turkeypointdbf_DB_1000_3day_transition_dates.csv')






# ~~~~~ Overall Analysis ~~~~~~~~





# listing sites

# All sites
sites = [(bartlett_daily, bartlett_greenup, bartlett_yearly, "bartlett"), (chestnut_daily, chestnut_greenup, chestnut_yearly, "chestnut"), (morgan_daily, morgan_greenup, morgan_yearly, "morgan"), (ozark_daily, ozark_greenup, ozark_yearly, "ozark"), (umb_daily, umb_greenup, umb_yearly, "umb"), (willow_daily, willow_greenup, willow_yearly, "willow"), (turkey_daily, turkey_greenup, turkey_yearly, "turkey"), (tonzi_daily, tonzi_greenup, tonzi_yearly, "tonzi")]    


# Sites without high altitude sites and Tonzi Ranch
best_sites = [(bartlett_daily, bartlett_greenup, bartlett_yearly, "bartlett"), (chestnut_daily, chestnut_greenup, chestnut_yearly, "chestnut"), (morgan_daily, morgan_greenup, morgan_yearly, "morgan"), (ozark_daily, ozark_greenup, ozark_yearly, "ozark")]    

# (umb_daily, umb_greenup, umb_yearly, "umb")
#  (willow_daily, willow_greenup, willow_yearly, "willow")
# (turkey_daily, turkey_greenup, turkey_yearly, "turkey")
# , (tonzi_daily, tonzi_greenup, tonzi_yearly, "tonzi") 


# function to convert green-up and Ts>Ta timestamps to day of year

def get_days_per_site(site_daily, site_greenup, site_yearly, site_name):
    
    # x = days in each yr when TS > TA
    x = []
    # y = green-up days via PhenoCam
    y = []
    
    names = []
    # finding years with both AmeriFlux and PhenoCam (w/ '.unique()')
    for year in site_daily['year'].unique():
        if year in list(site_greenup['year']):
            
            # splitting daily df into sub-dfs by given year
            df_year = site_daily[site_daily['year'] == year].reset_index()
           
            # finding mean annual air temp for given year
            #annual_temp = site_yearly[site_yearly['year'] == year]['TA (deg C)'][0]
            annual_temp = site_yearly['TA (deg C)'][0]
           
            # finding earliest day when TS > TA
            ta_ts_day = df_year[df_year['TS (deg C)'] >= annual_temp].sort_values(by = 'TIMESTAMP_START').reset_index().loc[0,'TIMESTAMP_START']
           
            # adding TS > TA to x list
            x.append(ta_ts_day)
            
            # reset index, finding green-up date for given year
            greenup_df = site_greenup.reset_index()
            greenup_day = greenup_df[greenup_df['year'] == year].iloc[0,0]
            
            # adding green-up date to y list
            y.append(greenup_day)
            
            # adding names as "Site Year" (ex. Ozark 2013)
            names.append(site_name + " " + str(year))
    
    
    x_days = [time_stamp.dayofyear for time_stamp in x]
    y_days = [time_stamp.dayofyear for time_stamp in y]
    
     
    return (names, x_days, y_days)

# new df with all Site-Years 
master = pd.DataFrame(columns = ["Site Name", "Day TS > TA", "Green-Up Day"])
master_2 = pd.DataFrame(columns = ["Site Name", "Day TS > TA", "Green-Up Day"])

def make_master(df, sites):    
    for site in sites:
        #print("sup")
        name, x_days, y_days = get_days_per_site(site[0], site[1], site[2], site[3])
        sub_df = pd.DataFrame({'Site Name': name, "Day TS > TA": x_days, "Green-Up Day": y_days})
        
        # adding all to master, with sub_dfs for each site
        df = pd.concat([df, sub_df])
        
    return df

master = make_master(master, sites)
master_2 = make_master(master_2, best_sites)
    



# plotting master scatterplot
plt.figure(figsize = (12,8))
sns.scatterplot("Day TS > TA", "Green-Up Day", data = master, hue = "Site Name")
plt.legend(loc='upper right', prop={'size':5}, bbox_to_anchor=(1,1))
plt.xlim(30,160)
plt.ylim(30,160)
x=master['Day TS > TA']
y=master['Green-Up Day']
plt.title("Green-Up Dates: PhenoCam VS Hyphothesized (All Sites)")
#mask = ~np.isnan(x) & ~np.isnan(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x.astype(float),y.astype(float))
fitted_values = x.astype(float) * slope + intercept
residuals = fitted_values - y.astype(float)
print(slope)
rr=str(round((r_value**2),3))
pp=str(round(p_value,3))
plt.text(40, 140, 'r-squared = ' +rr)
plt.text(40, 130, 'p-value = ' + pp)
plt.legend('',frameon=False)
#plt.savefig('Figures/Best_Working_Group.png')
plt.savefig('Figures/Best_Working_Group_No_Legend.png')
plt.show()

#Testing linearity
plt.figure()
sns.scatterplot("Day TS > TA", "Green-Up Day", data = master, hue = "Site Name")
plt.legend(loc='upper right', prop={'size':5}, bbox_to_anchor=(1,1))
plt.xlim(30,160)
plt.ylim(30,160)
x=master['Day TS > TA']
y=master['Green-Up Day']
plt.title("Green-Up Dates: PhenoCam VS Hyphothesized (All Sites)")
plt.savefig('Figures/Testing_Linearity.png')

#Testing linearity and homeoscadasticity
plt.scatter(fitted_values, residuals)
plt.axhline(y=0, color = 'black')
plt.title("Residual Plot")
plt.xlabel("Predicted Green-Up Day")
plt.ylabel("Residual")
plt.savefig("Figures/Testing_Homeoscadasticity")
plt.show()



#Testing normality
plt.hist(residuals, density=True, bins=15)
mu, std = norm.fit(residuals)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Histogram of Residuals")
plt.xlabel("Residual")
plt.ylabel("Density")
my_skew = skew(residuals,bias = False)
my_kurtosis = kurtosis(residuals, bias = False)
plt.text(20, 0.023, 'Skewness = ' + str(round(my_skew, 3)))
plt.text(20, 0.02, 'Kurtosis = ' + str(round(my_kurtosis, 3)))
plt.savefig("Figures/Testing_Normality")
plt.show()

sm.qqplot(residuals, line = 's')
plt.title("QQ Plot of Residuals")
plt.savefig("Figures/Testing_Normality_QQ")
plt.show()

#REPEATING STEPS FOR LOW ALTITUDE SITES

# plotting master scatterplot
plt.figure(figsize = (12,8))
sns.scatterplot("Day TS > TA", "Green-Up Day", data = master_2, hue = "Site Name")
plt.legend(loc='upper right', prop={'size':5}, bbox_to_anchor=(1,1))
plt.xlim(30,160)
plt.ylim(30,160)
x2=master_2['Day TS > TA']
y2=master_2['Green-Up Day']
plt.title("Green-Up Dates: PhenoCam VS Hyphothesized (Low Latitude Sites)")
#plt.savefig('Figures/Testing_Linearity2.png')
#mask = ~np.isnan(x) & ~np.isnan(y)
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x2.astype(float),y2.astype(float))
fitted_values2 = x2.astype(float) * slope2 + intercept2
residuals2 = fitted_values2 - y2.astype(float)
print(slope2)
rr2=str(round((r_value2**2),3))
pp2=str(round(p_value2,3))
plt.text(40, 140, 'r-squared = ' +rr2)
plt.text(40, 130, 'p-value = ' + pp2)
plt.legend('',frameon=False)
#plt.savefig('Figures/Best_Working_Group2.png')
plt.savefig('Figures/Best_Working_Group2_No_Legend.png')
plt.show()

#Testing linearity
plt.figure()
sns.scatterplot("Day TS > TA", "Green-Up Day", data = master_2, hue = "Site Name")
plt.legend(loc='upper right', prop={'size':5}, bbox_to_anchor=(1,1))
plt.xlim(30,160)
plt.ylim(30,160)
x2=master_2['Day TS > TA']
y2=master_2['Green-Up Day']
plt.title("Green-Up Dates: PhenoCam VS Hyphothesized (Low Altitude Sites)")
plt.savefig('Figures/Testing_Linearity2.png')

#Testing linearity and homeoscadasticity
plt.scatter(fitted_values2, residuals2)
plt.axhline(y=0, color = 'black')
plt.title("Residual Plot")
plt.xlabel("Predicted Green-Up Day")
plt.ylabel("Residual")
plt.savefig("Figures/Testing_Homeoscadasticity2")
plt.show()



#Testing normality
plt.hist(residuals2, density=True, bins=15)
mu, std = norm.fit(residuals2)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Histogram of Residuals")
plt.xlabel("Residual")
plt.ylabel("Density")
my_skew2 = skew(residuals2,bias = False)
my_kurtosis2 = kurtosis(residuals2, bias = False)
plt.text(7, 0.1, 'Skewness = ' + str(round(my_skew2, 3)))
plt.text(7, 0.09, 'Kurtosis = ' + str(round(my_kurtosis2, 3)))
plt.savefig("Figures/Testing_Normality2")
plt.show()

sm.qqplot(residuals2, line = 's')
plt.title("QQ Plot of Residuals")
plt.savefig("Figures/Testing_Normality_QQ2")
plt.show()

# PLOTTING


# Example Plots for Presentation
plt.figure()
ax = bartlett_daily[bartlett_daily['year'] == 2013]['TS (deg C)'].plot()
ax.axhline(y=10, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(bartlett_greenup[bartlett_greenup['year'] == 2013].index[0], color='green', label='Green-Up Date')
L = plt.legend()
L.get_texts()[0].set_text('Soil Temp')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.savefig('Figures/example.png')
plt.show()








    # Tonzi plots, using yearly resampling

# Tonzi 2018 plot
plt.figure()
ax = tonzi_daily[tonzi_daily['year'] == 2018]['TS (deg C)'].plot()
ax.axhline(y=16.704971570604997, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(tonzi_greenup[tonzi_greenup['year'] == 2018].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Tonzi Ranch, CA 2018 (yearly resampling)')
plt.savefig('Figures/Tonzi_2018_resampled_1.png')
plt.show()

# Tonzi 2012 plot
plt.figure()
ax = tonzi_daily[tonzi_daily['year'] == 2012]['TS (deg C)'].plot()
ax.axhline(y=16.54828628329909, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(tonzi_greenup[tonzi_greenup['year'] == 2012].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Tonzi Ranch, CA 2012 (yearly resampling)')
plt.savefig('Figures/Tonzi_2012_resampled.png')
plt.show()

# Tonzi 2013 plot
plt.figure()
ax = tonzi_daily[tonzi_daily['year'] == 2013]['TS (deg C)'].plot()
ax.axhline(y=16.65796920399535, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(tonzi_greenup[tonzi_greenup['year'] == 2013].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Tonzi Ranch, CA 2013 (yearly resampling)')
plt.savefig('Figures/Tonzi_2013_resampled.png')
plt.show()

# Tonzi 2014 plot
plt.figure()
ax = tonzi_daily[tonzi_daily['year'] == 2014]['TS (deg C)'].plot()
ax.axhline(y=17.762677796974895, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(tonzi_greenup[tonzi_greenup['year'] == 2014].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Tonzi Ranch, CA 2014 (yearly resampling)')
plt.savefig('Figures/Tonzi_2014_resampled.png')
plt.show()



    # Tonzi plots, using 20-year recursive filter MAAT

# Tonzi 2012
plt.figure()
ax = tonzi_daily[tonzi_daily['year'] == 2012]['TS (deg C)'].plot()
ax.axhline(y=21.21197159972164, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(tonzi_greenup[tonzi_greenup['year'] == 2012].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Tonzi Ranch, CA 2012 (20-year recursive filter MAAT)')
plt.savefig('Figures/Tonzi_2012_20yr.png')
plt.show()

# Tonzi 2013
plt.figure()
ax = tonzi_daily[tonzi_daily['year'] == 2013]['TS (deg C)'].plot()
ax.axhline(y=21.21197159972164, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(tonzi_greenup[tonzi_greenup['year'] == 2013].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Tonzi Ranch, CA 2013 (20-year recursive filter MAAT)')
plt.savefig('Figures/Tonzi_2013_20yr.png')
plt.show()

# Tonzi 2014
plt.figure()
ax = tonzi_daily[tonzi_daily['year'] == 2014]['TS (deg C)'].plot()
ax.axhline(y=21.21197159972164, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(tonzi_greenup[tonzi_greenup['year'] == 2014].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Tonzi Ranch, CA 2014 (20-year recursive filter MAAT)')
plt.savefig('Figures/Tonzi_2014_20yr.png')
plt.show()



# Tonzi plots, using 10-year recursive filter MAAT

# Tonzi 2012
plt.figure()
ax = tonzi_daily[tonzi_daily['year'] == 2012]['TS (deg C)'].plot()
ax.axhline(y=24.32516047162206, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(tonzi_greenup[tonzi_greenup['year'] == 2012].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Tonzi Ranch, CA 2012 (10-year recursive filter MAAT)')
plt.savefig('Figures/Tonzi_2012_10yr.png')
plt.show()

# Tonzi 2013
plt.figure()
ax = tonzi_daily[tonzi_daily['year'] == 2013]['TS (deg C)'].plot()
ax.axhline(y=24.32516047162206, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(tonzi_greenup[tonzi_greenup['year'] == 2013].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Tonzi Ranch, CA 2013 (10-year recursive filter MAAT)')
plt.savefig('Figures/Tonzi_2013_10yr.png')
plt.show()

# Tonzi 2014
plt.figure()
ax = tonzi_daily[tonzi_daily['year'] == 2014]['TS (deg C)'].plot()
ax.axhline(y=24.32516047162206, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(tonzi_greenup[tonzi_greenup['year'] == 2014].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Tonzi Ranch, CA 2014 (10-year recursive filter MAAT)')
plt.savefig('Figures/Tonzi_2014_10yr.png')
plt.show()










# Bartlett plots, using yearly resampling

# Bartlett 2008
plt.figure()
ax = bartlett_daily[bartlett_daily['year'] == 2008]['TS (deg C)'].plot()
ax.axhline(y=7.158035246470858, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(bartlett_greenup[bartlett_greenup['year'] == 2008].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Bartlett Forest, NH 2008')
plt.savefig('Figures/Bartlett_2008.png')
plt.show()

# Bartlett 2009
plt.figure()
ax = bartlett_daily[bartlett_daily['year'] == 2009]['TS (deg C)'].plot()
ax.axhline(y=6.721109412100431, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(bartlett_greenup[bartlett_greenup['year'] == 2009].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Bartlett Forest, NH 2009')
plt.savefig('Figures/Bartlett_2009.png')
plt.show()

# Bartlett 2010
plt.figure()
ax = bartlett_daily[bartlett_daily['year'] == 2010]['TS (deg C)'].plot()
ax.axhline(y=8.522011672945206, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(bartlett_greenup[bartlett_greenup['year'] == 2010].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Bartlett Forest, NH 2010')
plt.savefig('Figures/Bartlett_2010.png')
plt.show()

# Bartlett 2011
plt.figure()
ax = bartlett_daily[bartlett_daily['year'] == 2011]['TS (deg C)'].plot()
ax.axhline(y=7.8707626147260035, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(bartlett_greenup[bartlett_greenup['year'] == 2011].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Bartlett Forest, NH 2011')
plt.savefig('Figures/Bartlett_2011.png')
plt.show()

# Bartlett 2012
plt.figure()
ax = bartlett_daily[bartlett_daily['year'] == 2012]['TS (deg C)'].plot()
ax.axhline(y=8.470718469945298, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(bartlett_greenup[bartlett_greenup['year'] == 2012].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Bartlett Forest, NH 2012')
plt.savefig('Figures/Bartlett_2012.png')
plt.show()

# Bartlett 2013
plt.figure()
ax = bartlett_daily[bartlett_daily['year'] == 2013]['TS (deg C)'].plot()
ax.axhline(y=7.175704175228293, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(bartlett_greenup[bartlett_greenup['year'] == 2013].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Bartlett Forest, NH 2013')
plt.savefig('Figures/Bartlett_2013.png')
plt.show()

# Bartlett 2014
plt.figure()
ax = bartlett_daily[bartlett_daily['year'] == 2014]['TS (deg C)'].plot()
ax.axhline(y=6.776349437594051, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(bartlett_greenup[bartlett_greenup['year'] == 2014].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Bartlett Forest, NH 2014')
plt.savefig('Figures/Bartlett_2014.png')
plt.show()

# Bartlett 2015
plt.figure()
ax = bartlett_daily[bartlett_daily['year'] == 2015]['TS (deg C)'].plot()
ax.axhline(y=7.20975191324201, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(bartlett_greenup[bartlett_greenup['year'] == 2015].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Bartlett Forest, NH 2015')
plt.savefig('Figures/Bartlett_2015.png')
plt.show()

# Barrlett 2016
plt.figure()
ax = bartlett_daily[bartlett_daily['year'] == 2016]['TS (deg C)'].plot()
ax.axhline(y=7.939494575933541, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(bartlett_greenup[bartlett_greenup['year'] == 2016].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Bartlett Forest, NH 2016')
plt.savefig('Figures/Bartlett_2016.png')
plt.show()






# Morgan plots, using yearly resampling

# Morgan 2009
plt.figure()
ax = morgan_daily[morgan_daily['year'] == 2009]['TS (deg C)'].plot()
ax.axhline(y=11.84538308227112, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(morgan_greenup[morgan_greenup['year'] == 2009].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Morgan-Monroe, IN 2009')
plt.savefig('Figures/Morgan_2009.png')
plt.show()

# Morgan 2010
plt.figure()
ax = morgan_daily[morgan_daily['year'] == 2010]['TS (deg C)'].plot()
ax.axhline(y=12.683578376486937, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(morgan_greenup[morgan_greenup['year'] == 2010].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Morgan-Monroe, IN 2010')
plt.savefig('Figures/Morgan_2010.png')
plt.show()

# Morgan 2011
plt.figure()
ax = morgan_daily[morgan_daily['year'] == 2011]['TS (deg C)'].plot()
ax.axhline(y=12.788095293031825, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(morgan_greenup[morgan_greenup['year'] == 2011].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Morgan-Monroe, IN 2011')
plt.savefig('Figures/Morgan_2011.png')
plt.show()

# Morgan 2012
plt.figure()
ax = morgan_daily[morgan_daily['year'] == 2012]['TS (deg C)'].plot()
ax.axhline(y=14.20182363761775, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(morgan_greenup[morgan_greenup['year'] == 2012].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Morgan-Monroe, IN 2012')
plt.savefig('Figures/Morgan_2012.png')
plt.show()

# Morgan 2013
plt.figure()
ax = morgan_daily[morgan_daily['year'] == 2013]['TS (deg C)'].plot()
ax.axhline(y=11.741968144044348, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(morgan_greenup[morgan_greenup['year'] == 2013].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Morgan-Monroe, IN 2013')
plt.savefig('Figures/Morgan_2013.png')
plt.show()

# Morgan 2014
plt.figure()
ax = morgan_daily[morgan_daily['year'] == 2014]['TS (deg C)'].plot()
ax.axhline(y=11.143821287931473, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(morgan_greenup[morgan_greenup['year'] == 2014].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Morgan-Monroe, IN 2014')
plt.savefig('Figures/Morgan_2014.png')
plt.show()

# Morgan 2015
plt.figure()
ax = morgan_daily[morgan_daily['year'] == 2015]['TS (deg C)'].plot()
ax.axhline(y=15.829055260461422, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(morgan_greenup[morgan_greenup['year'] == 2015].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Morgan-Monroe, IN 2015')
plt.savefig('Figures/Morgan_2015.png')
plt.show()

# Morgan 2016
plt.figure()
ax = morgan_daily[morgan_daily['year'] == 2016]['TS (deg C)'].plot()
ax.axhline(y=12.497229993261511, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(morgan_greenup[morgan_greenup['year'] == 2016].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Morgan-Monroe, IN 2016')
plt.savefig('Figures/Morgan_2016.png')
plt.show()

# Morgan 2017
plt.figure()
ax = morgan_daily[morgan_daily['year'] == 2017]['TS (deg C)'].plot()
ax.axhline(y=13.221003257221998, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(morgan_greenup[morgan_greenup['year'] == 2017].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Morgan-Monroe, IN 2017')
plt.savefig('Figures/Morgan_2017.png')
plt.show()

# Morgan 2018
plt.figure()
ax = morgan_daily[morgan_daily['year'] == 2018]['TS (deg C)'].plot()
ax.axhline(y=11.751812649680135, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(morgan_greenup[morgan_greenup['year'] == 2018].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Morgan-Monroe, IN 2018')
plt.savefig('Figures/Morgan_2018.png')
plt.show()



# UMB plots

# UMB 2009
plt.figure()
ax = umb_daily[umb_daily['year'] == 2009]['TS (deg C)'].plot()
ax.axhline(y=6.238936277438999, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(umb_greenup[umb_greenup['year'] == 2009].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('U Mich, MI 2009')
plt.savefig('Figures/UMB_2009.png')
plt.show()

# UMB 2010
plt.figure()
ax = umb_daily[umb_daily['year'] == 2010]['TS (deg C)'].plot()
ax.axhline(y=8.33064750030802, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(umb_greenup[umb_greenup['year'] == 2010].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('U Mich, MI 2010')
plt.savefig('Figures/UMB_2010.png')
plt.show()

# UMB 2011
plt.figure()
ax = umb_daily[umb_daily['year'] == 2011]['TS (deg C)'].plot()
ax.axhline(y=7.429604377519229, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(umb_greenup[umb_greenup['year'] == 2011].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('U Mich, MI 2011')
plt.savefig('Figures/UMB_2011.png')
plt.show()

# UMB 2012
plt.figure()
ax = umb_daily[umb_daily['year'] == 2012]['TS (deg C)'].plot()
ax.axhline(y=9.18656815208935, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(umb_greenup[umb_greenup['year'] == 2012].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('U Mich, MI 2012')
plt.savefig('Figures/UMB_2012.png')
plt.show()

# UMB 2013
plt.figure()
ax = umb_daily[umb_daily['year'] == 2013]['TS (deg C)'].plot()
ax.axhline(y=6.534313777901732, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(umb_greenup[umb_greenup['year'] == 2013].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('U Mich, MI 2013')
plt.savefig('Figures/UMB_2013.png')
plt.show()



# Willow Plots ----- this site really didn't seem to work too well w the hypothesis -- very cold MAATs

# Willow 2012
plt.figure()
ax = willow_daily[willow_daily['year'] == 2012]['TS (deg C)'].plot()
ax.axhline(y=7.124280896303204, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(willow_greenup[willow_greenup['year'] == 2012].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Willow Creek, WI 2012')
plt.savefig('Figures/Willow_2012.png')
plt.show()

# Willow 2013
plt.figure()
ax = willow_daily[willow_daily['year'] == 2013]['TS (deg C)'].plot()
ax.axhline(y=4.019773671903515, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(willow_greenup[willow_greenup['year'] == 2013].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Willow Creek, WI 2013')
plt.savefig('Figures/Willow_2013.png')
plt.show()

# Willow 2014
plt.figure()
ax = willow_daily[willow_daily['year'] == 2014]['TS (deg C)'].plot()
ax.axhline(y=3.21751924313755, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(willow_greenup[willow_greenup['year'] == 2014].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Willow Creek, WI 2014')
plt.savefig('Figures/Willow_2014.png')
plt.show()

# Willow 2015
plt.figure()
ax = willow_daily[willow_daily['year'] == 2015]['TS (deg C)'].plot()
ax.axhline(y=5.791206285186376, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(willow_greenup[willow_greenup['year'] == 2015].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Willow Creek, WI 2015')
plt.savefig('Figures/Willow_2015.png')
plt.show()

# Willow 2016
plt.figure()
ax = willow_daily[willow_daily['year'] == 2016]['TS (deg C)'].plot()
ax.axhline(y=6.430041891791115, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(willow_greenup[willow_greenup['year'] == 2016].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Willow Creek, WI 2016')
plt.savefig('Figures/Willow_2016.png')
plt.show()





# Missouri Ozark Plots

# Ozark 2012 (Deciduous Broadleaf)
plt.figure()
ax = ozark_daily[ozark_daily['year'] == 2012]['TS (deg C)'].plot()
ax.axhline(y=15.38531943698328, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(ozark_greenup[ozark_greenup['year'] == 2012].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Missouri Ozark 2012 (DB)')
plt.savefig('Figures/Ozark_DB_2012.png')
plt.show()

# Ozark 2013
plt.figure()
ax = ozark_daily[ozark_daily['year'] == 2013]['TS (deg C)'].plot()
ax.axhline(y=12.662980696679153, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(ozark_greenup[ozark_greenup['year'] == 2013].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Missouri Ozark 2013 (DB)')
plt.savefig('Figures/Ozark_DB_2013.png')
plt.show()

# Ozark 2014
plt.figure()
ax = ozark_daily[ozark_daily['year'] == 2014]['TS (deg C)'].plot()
ax.axhline(y=12.374499981366833, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(ozark_greenup[ozark_greenup['year'] == 2014].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Missouri Ozark 2014 (DB)')
plt.savefig('Figures/Ozark_DB_2014.png')
plt.show()

# Ozark 2015
plt.figure()
ax = ozark_daily[ozark_daily['year'] == 2015]['TS (deg C)'].plot()
ax.axhline(y=13.816207800987298, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(ozark_greenup[ozark_greenup['year'] == 2015].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Missouri Ozark 2015 (DB)')
plt.savefig('Figures/Ozark_DB_2015.png')
plt.show()

# Ozark 2016
plt.figure()
ax = ozark_daily[ozark_daily['year'] == 2016]['TS (deg C)'].plot()
ax.axhline(y=14.905154829356464, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(ozark_greenup[ozark_greenup['year'] == 2016].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Missouri Ozark 2016 (DB)')
plt.savefig('Figures/Ozark_DB_2016.png')
plt.show()

# Ozark 2016
plt.figure()
ax = ozark_daily[ozark_daily['year'] == 2017]['TS (deg C)'].plot()
ax.axhline(y=14.70990102359901, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(ozark_greenup[ozark_greenup['year'] == 2017].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Missouri Ozark 2017 (DB)')
plt.savefig('Figures/Ozark_DB_2017.png')
plt.show()


# Harvard Forest Plots

# Harvard 2008
plt.figure()
ax = harvard_daily[harvard_daily['year'] == 2008]['TS (deg C)'].plot()
ax.axhline(y=8.795650571323257, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(harvard_greenup[harvard_greenup['year'] == 2008].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Harvard Forest 2008')
plt.savefig('Figures/Harvard_2008.png')
plt.show()

# Harvard 2011
plt.figure()
ax = harvard_daily[harvard_daily['year'] == 2011]['TS (deg C)'].plot()
ax.axhline(y=7.439340572033897, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(harvard_greenup[harvard_greenup['year'] == 2011].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Harvard Forest 2011')
plt.savefig('Figures/Harvard_2011.png')
plt.show()

# Harvard 2012
plt.figure()
ax = harvard_daily[harvard_daily['year'] == 2012]['TS (deg C)'].plot()
ax.axhline(y=12.186417535034147, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(harvard_greenup[harvard_greenup['year'] == 2012].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Harvard Forest 2012')
plt.savefig('Figures/Harvard_2012.png')
plt.show()

# Harvard 2013
plt.figure()
ax = harvard_daily[harvard_daily['year'] == 2013]['TS (deg C)'].plot()
ax.axhline(y=8.506677074576661, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(harvard_greenup[harvard_greenup['year'] == 2013].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Harvard Forest 2013')
plt.savefig('Figures/Harvard_2013.png')
plt.show()

# Harvard 2014
plt.figure()
ax = harvard_daily[harvard_daily['year'] == 2014]['TS (deg C)'].plot()
ax.axhline(y=6.990969854044476, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(harvard_greenup[harvard_greenup['year'] == 2014].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Harvard Forest 2014')
plt.savefig('Figures/Harvard_2014.png')
plt.show()

# Harvard 2015
plt.figure()
ax = harvard_daily[harvard_daily['year'] == 2015]['TS (deg C)'].plot()
ax.axhline(y=9.176531683977105, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(harvard_greenup[harvard_greenup['year'] == 2015].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Harvard Forest 2015')
plt.savefig('Figures/Harvard_2015.png')
plt.show()

# Harvard 2016
plt.figure()
ax = harvard_daily[harvard_daily['year'] == 2016]['TS (deg C)'].plot()
ax.axhline(y=10.139076994434145, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(harvard_greenup[harvard_greenup['year'] == 2016].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Harvard Forest 2016')
plt.savefig('Figures/Harvard_2016.png')
plt.show()



# Vaira plotting

# Vaira 2012
plt.figure()
ax = vaira_daily[vaira_daily['year'] == 2012]['TS (deg C)'].plot()
ax.axhline(y=16.041575484858736, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(vaira_greenup[vaira_greenup['year'] == 2012].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Vaira Ranch 2012')
plt.savefig('Figures/Vaira_2012.png')
plt.show()

# Vaira 2014
plt.figure()
ax = vaira_daily[vaira_daily['year'] == 2014]['TS (deg C)'].plot()
ax.axhline(y=17.4243094151256, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(vaira_greenup[vaira_greenup['year'] == 2014].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Vaira Ranch 2014')
plt.savefig('Figures/Vaira_2014.png')
plt.show()

# Vaira 2016
plt.figure()
ax = vaira_daily[vaira_daily['year'] == 2016]['TS (deg C)'].plot()
ax.axhline(y=16.778545465619384, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(vaira_greenup[vaira_greenup['year'] == 2016].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Vaira Ranch 2016')
plt.savefig('Figures/Vaira_2016.png')
plt.show()

# Vaira 2018
plt.figure()
ax = vaira_daily[vaira_daily['year'] == 2018]['TS (deg C)'].plot()
ax.axhline(y=16.235964185159744, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(vaira_greenup[vaira_greenup['year'] == 2018].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Vaira Ranch 2018')
plt.savefig('Figures/Vaira_2018.png')
plt.show()


# Turkey plottings 

# Turkey 2012
plt.figure()
ax = turkey_daily[turkey_daily['year'] == 2012]['TS (deg C)'].plot()
ax.axhline(y=12.022366400740001, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(turkey_greenup[turkey_greenup['year'] == 2012].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Turkey Point 2012')
plt.savefig('Figures/Turkey_2012.png')
plt.show()

# Turkey 2013
plt.figure()
ax = turkey_daily[turkey_daily['year'] == 2013]['TS (deg C)'].plot()
ax.axhline(y=9.226724149757121, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(turkey_greenup[turkey_greenup['year'] == 2013].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Turkey Point 2013')
plt.savefig('Figures/Turkey_2013.png')
plt.show()

# Turkey 2014
plt.figure()
ax = turkey_daily[turkey_daily['year'] == 2014]['TS (deg C)'].plot()
ax.axhline(y=8.020265162764233, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(turkey_greenup[turkey_greenup['year'] == 2014].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Turkey Point 2014')
plt.savefig('Figures/Turkey_2014.png')
plt.show()

# Turkey 2015
plt.figure()
ax = turkey_daily[turkey_daily['year'] == 2015]['TS (deg C)'].plot()
ax.axhline(y=9.65813603959232, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(turkey_greenup[turkey_greenup['year'] == 2015].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Turkey Point 2015')
plt.savefig('Figures/Turkey_2015.png')
plt.show()

# Turkey 2016
plt.figure()
ax = turkey_daily[turkey_daily['year'] == 2016]['TS (deg C)'].plot()
ax.axhline(y=11.386590253905053, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(turkey_greenup[turkey_greenup['year'] == 2016].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Turkey Point 2016')
plt.savefig('Figures/Turkey_2016.png')
plt.show()

# Turkey 2017
plt.figure()
ax = turkey_daily[turkey_daily['year'] == 2017]['TS (deg C)'].plot()
ax.axhline(y=10.023506396344946, color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(turkey_greenup[turkey_greenup['year'] == 2017].index[0], color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Turkey Point 2017')
plt.savefig('Figures/Turkey_2017.png')
plt.show()


#Merging Sites Across Years
def get_avg_day(site_greenup):
    avg_day = int(np.mean([int(date[3:]) for date in site_greenup["Month/Day"]]))
    return pd.to_datetime(site_greenup["Month/Day"][0][:3] + str(avg_day), format = "%m/%d")


#Willow Averaged
willow_daily["Month/Day"] = willow_daily.index.strftime('%m/%d')
willow_avg = willow_daily.groupby("Month/Day").mean()
willow_avg.drop('02/29', inplace = True)
willow_avg.index = pd.to_datetime(willow_avg.index, format = '%m/%d')
willow_greenup["Month/Day"] = willow_greenup.index.strftime('%m/%d')
willow_avg_date = get_avg_day(willow_greenup)


plt.figure()
ax = willow_avg['TS (deg C)'].plot()
ax.axhline(y=willow_yearly['TA (deg C)'][0], color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(willow_avg_date, color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Willow Creek')
plt.savefig('Figures/Willow_Avg.png')
plt.show()


#UMB Averaged
umb_daily["Month/Day"] = umb_daily.index.strftime('%m/%d')
umb_avg = umb_daily.groupby("Month/Day").mean()
umb_avg.drop('02/29', inplace = True)
umb_avg.index = pd.to_datetime(umb_avg.index, format = '%m/%d')
umb_greenup["Month/Day"] = umb_greenup.index.strftime('%m/%d')
umb_avg_date = get_avg_day(umb_greenup)


plt.figure()
ax = umb_avg['TS (deg C)'].plot()
ax.axhline(y=umb_yearly["TA (deg C)"][0], color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(umb_avg_date, color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('U Mich, MI')
plt.savefig('Figures/UMB_Avg.png')
plt.show()


#Ozark Averaged
ozark_daily["Month/Day"] = ozark_daily.index.strftime('%m/%d')
ozark_avg = ozark_daily.groupby("Month/Day").mean()
ozark_avg.drop('02/29', inplace = True)
ozark_avg.index = pd.to_datetime(ozark_avg.index, format = '%m/%d')
ozark_greenup["Month/Day"] = ozark_greenup.index.strftime('%m/%d')
ozark_avg_date = get_avg_day(ozark_greenup)


plt.figure()
ax = ozark_avg['TS (deg C)'].plot()
ax.axhline(y=ozark_yearly["TA (deg C)"][0], color='black', lw=1, label='Mean Ann Air Temp')
plt.axvline(ozark_avg_date, color='green', label='Green-Up')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Missouri Ozark (DB)')
plt.savefig('Figures/Ozark_Avg.png')
plt.show()





