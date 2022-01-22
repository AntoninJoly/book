import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from pre_processing import PreProcess
from learn_preferences import LearnPreferences
import sys
from tqdm import tqdm
from utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator

# base = 'C:/Users/Antonin_JOLY/Desktop/mansion_fit'

# reco = pd.read_csv(os.path.join(base, 'building_recommended_logs.csv'),index_col=0)
# reco = reco.drop(columns=['type','user_id','recommended_at','created_at','updated_at','deleted_at','updated_by',])
# reco = reco[~reco['building_id'].isin([36,46,53])]

# user = pd.read_csv(os.path.join(base, 'user_attributes.csv'),index_col=0)
# user = user.drop(columns=['user_id','created_at','updated_at','deleted_at','updated_by',])

# building = pd.read_csv(os.path.join(base, 'buildings.csv'),index_col=0)
# building = building.drop(columns=['rooms','feature','buk_code','pref_code','feature_1_tag','feature_2_tag','feature_3_tag','station_1_line_code','station_1_code','station_1_lon','station_1_lat','station_2_line_code','station_2_code','station_2_lon','station_2_lat','station_3_line_code','station_3_code','station_3_lon','station_3_lat','most_floor_plan','most_price','address_1','address_2','name','disp_name','prj_code','closed','type','most_width','time_resident','lat','lon','created_at','updated_at','deleted_at','updated_by','opened_at','closed_at','overview_url','official_website_url','reservation_started_at','reservation_ended_at'])
# building = building.drop([53,46,36])

# building_processed, not_indexed = process_station_price(building)

# for i in not_indexed:
#     reco = reco[reco['building_id']!=i]

# X_user = StandardScaler().fit_transform(user_processed)
# k_means_fit = KMeans(n_clusters=7).fit(X_user)
# user['cluster'] = k_means_fit.labels_



df_sea = pd.read_csv('./seattle.csv', index_col=0)
df_sf = pd.read_csv('./san_fran.csv', index_col=0)

sea_ref = df_sea[['city', 'state', 'street', 'finishedsqft', 'bedrooms', 'bathrooms', 'trans_score', 'walkscore_score']]
sf_ref = df_sf[['city', 'state', 'street', 'finishedsqft', 'bedrooms', 'bathrooms', 'trans_score', 'walkscore_score']]

# create the PreProcesss objects
prep_sf = PreProcess(df_sf)
prep_sea = PreProcess(df_sea)

# drop the unneccessary columns, clean_up NA's and normalize use in the recommender
sf = prep_sf.drop_columns()
sf = prep_sf.preprocess_df(sf)
sf = prep_sf.create_parking_index(sf)
sf = prep_sf.normalize_columns(sf)

sea = prep_sea.drop_columns()
sea = prep_sea.preprocess_df(sea)
sea = prep_sf.create_parking_index(sea)
# sea = prep_sea.normalize_columns(sea)

# specify the metrics to use for the similarity matrix 
metrics = ['walk_distance', 'house_superficy','family_amenities']

# # init a LearnPreferences object with seed house of SanFran index 3
# lp = LearnPreferences(sf, sea, sf_ref, sea_ref, metrics, 100)
lp = LearnPreferences(sf, sea.drop('school_index',axis=1), metrics, 1, 100)

# init the recommendations
lp.update_recommendations()

for i in tqdm(range(75)):
    lp.show_recommendations()
    lp.get_user_choice()
    lp.guess_preferences()
    lp.generate_images()

# my_past_house = sea.loc[1] # user has previously lived in house with index 1
# recommendations = lp.get_most_similar(lp.get_sim_mat(), my_past_house, 3) # provides the 4 most similar listings
# choice = get_user_choice(recommendations)
# update_user_history(recommendations)