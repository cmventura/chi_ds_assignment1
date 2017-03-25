
#==============================================================================
# Assignment 1: Data Exploration
# 
# Goal: Explore Q4 2016 Divvy Ridership Data
#==============================================================================

#==============================================================================
# 0.0 Import Libraries
#==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

#==============================================================================
# 1.0 Import Data
# Original Source: https://www.divvybikes.com/system-data
# Note: Only importing Q4 data for ease of analysis
#==============================================================================

df_trips = pd.read_csv("/Users/cventura/GitHub/Homework/Divvy_Trips_2016_Q3Q4/Divvy_Trips_2016_Q4.csv")
df_stations = pd.read_csv("/Users/cventura/GitHub/Homework/Divvy_Trips_2016_Q3Q4/Divvy_Stations_2016_Q4.csv")

#==============================================================================
# 2.0 Explore Data
# Note: Supporting documentation indicates data is one record per trip
#==============================================================================

#list columns
print list(df_trips.columns.values)

#find number of records
print len(df_trips)

#describe dataset
print df_trips.describe()

print df_trips.isnull() #Note - Nulls in gender/birthyead consistent with documentation - null for non subscribers
#Note - I will try to fill in gender below for fun!  But these nulls will not impact other analyses

#==============================================================================
# 3.0 Create some additional columns
#==============================================================================

#Create is_subscriber dummy variable
df_trips["is_subscriber"] = df_trips['usertype'].apply(lambda x: 1 if x == 'Subscriber' else 0)

#Create is_male dummy variable for predictions later
df_trips["is_male"] = df_trips['gender'].apply(lambda x: 1 if x == 'Male'  else 0 if x == 'Female' else None)

#Create additional variable converting tripduration to minutes
df_trips["trip_dur_min"] = df_trips['tripduration'] / 60
        

print df_trips["trip_dur_min"].describe() #much better

#==============================================================================
# 4.0 Let's graph trip duration by amount of trips
#==============================================================================

#Group by unique trip durations (rounded to minutes) and calculate count
df_trips["trip_dur_min_rd"] = df_trips['trip_dur_min'].round(0)
                                                               
df_tripct_dur = pd.DataFrame({'tripcount': df_trips.groupby(["trip_dur_min_rd"]).size()}).reset_index()
  
#Limit to only durations with 100+ rides
df_tripct_dur = df_tripct_dur[df_tripct_dur['tripcount'] > 100]

#Graph trips             
for index, row in df_tripct_dur.iterrows():
    x = row['trip_dur_min_rd']
    y = row['tripcount']
    plt.scatter(x, y, s=1, c='blue', alpha=0.5)
plt.show() #skewed normal distribution

#==============================================================================
# 5.0 Let's find the top 10 destination stations
#==============================================================================

#Group by destination station and count
df_topdes = pd.DataFrame({'tripcount': df_trips.groupby(["to_station_name"]).size()}).reset_index()
dt_todes_10 =  df_topdes.nlargest(10, 'tripcount') #All downtown, so surprised

#Graph them

bars = dt_todes_10['to_station_name']
y_pos = np.arange(len(bars))
y = dt_todes_10['tripcount']

plt.bar(y_pos, y, align='center', alpha=0.5)
plt.xticks(y_pos, bars, rotation='vertical')
plt.ylabel('Number of Trips')
plt.title('Top Divvy Trip Destinations')
 
plt.show()  #Lots of people going to Navy Pier

#==============================================================================
# 6.0 Let's find the top 10 destination stations for subscribers only
# Note: I'm curious if subscribers go to Navy pier the most, or if the
#       top destination is driven by one-time users
#==============================================================================

#Create subscriber only dataset then repeat process
df_trips_sub = df_trips[df_trips['is_subscriber'] == 1]

#Repeat Process Above
#Group by destination station and count
df_topdes_sub = pd.DataFrame({'tripcount': df_trips_sub.groupby(["to_station_name"]).size()}).reset_index()
df_topdes_sub_10 =  df_topdes_sub.nlargest(10, 'tripcount') #All downtown, so surprised

#Graph them

bars = df_topdes_sub_10['to_station_name']
y_pos = np.arange(len(bars))
y = df_topdes_sub_10['tripcount']

plt.bar(y_pos, y, align='center', alpha=0.5)
plt.xticks(y_pos, bars, rotation='vertical')
plt.ylabel('Number of Trips')
plt.title('Top Divvy Trip Destinations - Subscribers Only')
 
plt.show() #Now all the top trips for subscribers are people going to Union/Ogilvie

#==============================================================================
# 7.0 Correlation between trip length and subscribership
# Note: Hypothesis is that susbscribers take shorter trips
#==============================================================================

df_trips_corr = df_trips[['is_subscriber', 'trip_dur_min_rd']]

print df_trips_corr.groupby(["is_subscriber"]).agg([np.mean, np.median, np.std])
print df_trips_corr.corr()
#looks like there is a negative correlation, but somewhat weak.  Mean and Median values are higher
#   for non-subscribers, but stddev is higher, so much more variation

#==============================================================================
# 8.0 Further areas for research
#     1 - Examine additional possible predictors of destination or trip duration such as:
#         a. Time of day
#         b. Day of week
#         c. Gender
#         d. Age (derived from birth year)
#     2 - Combine with datasets from other quarters and examine seasonality
#     3 - Plot common routes and examine correlation with time
#     4 - Exclude downtown or look at ride trends within specific neighborhoods
#     5 - Create destination prediction model
#         - Use provided variables in a regression (maybe KNN?) to predict destination
#==============================================================================

#==============================================================================
# Bonus: merge trips and station info datasets
# Note: contains latitude/longitude info for section 10
#==============================================================================
df_stations = pd.read_csv("/Users/cventura/GitHub/Homework/Divvy_Trips_2016_Q3Q4/Divvy_Stations_2016_Q4.csv")
df_full = df_trips.merge(df_stations, left_on="from_station_id", right_on="id")

#==============================================================================
# Note - Everything below is me just testing some things out.
#==============================================================================

#==============================================================================
# 9.0 Use KNN to predict gender for non-subscribers
#==============================================================================

#predict gender just for fun, I'm sure this would not be a super valid way of predicting gender

#limiting to 50 because this takes a long time
df_top50 = df_full.head(50)

df_nona = df_full.dropna()

neigh = KNeighborsRegressor(n_neighbors=1)
neigh.fit(df_nona[["from_station_id","to_station_id","is_male"]], df_nona["is_male"])

df_top50.loc[:,'is_male_neighborimpute'] = df_top50.apply(lambda x: neigh.predict(df_top50[["from_station_id","to_station_id","is_subscriber"]])[0] if pd.isnull(x['is_male']) else x['is_male'], axis=1)

print df_top50

#==============================================================================
# 10.0 - Plot rides on a map
# Note - Requires basemap, which I had to install
#==============================================================================

from mpl_toolkits.basemap import Basemap

#aggregate rides by station
df_stationtripcount = pd.DataFrame({'trips': df_full.groupby(["from_station_name", "latitude", "longitude"]).size()}).reset_index()


#Set Map parameters

fig, ax = plt.subplots()

urcornerlat = 42.03
urcornerlon = -87.51
llcornerlat = 41.71
llcornerlon = -87.73

m = Basemap(projection='merc', resolution='i', ax=ax, llcrnrlat=llcornerlat,
            urcrnrlat=urcornerlat,llcrnrlon=llcornerlon,urcrnrlon=urcornerlon)
m.drawmapboundary()
m.drawcoastlines()
m.fillcontinents(color='gray',lake_color='navy')

# Get the location of each station and plot it, with number of trips used for size of dot
for index, row in df_stationtripcount.iterrows():
    lon = row['longitude']
    lat = row['latitude']
    x,y = m(lon, lat)
    m.plot(x, y, 'ro', markersize= (.001 * row['trips']))
    
plt.show()