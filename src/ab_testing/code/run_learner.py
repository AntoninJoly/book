import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import sys
from learn_preferences import LearnPreferences
import sys
from tqdm import tqdm
from utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator

base = 'C:/Users/Antonin_JOLY/Desktop/mansion_fit'

reco = pd.read_csv(os.path.join(base, 'building_recommended_logs.csv'),index_col=0).drop(columns=['type','user_id','recommended_at','created_at','updated_at','deleted_at','updated_by',])
reco = reco[~reco['building_id'].isin([36,46,53])]

# users_df = pd.read_csv(os.path.join(base, 'user_attributes.csv'),index_col=0)
# users_df = users_df.drop(columns=['user_id','created_at','updated_at','deleted_at','updated_by',
#                                   'home_station_pref_code','home_station_line_code','home_station_code',
#                                   'office_station_pref_code','office_station_line_code','office_station_code',
#                                   'type_analysis_result_id','area_analysis_result_id'])

user_df = pd.read_csv(os.path.join(base, 'PwsKokyakuBase.csv'),index_col=0)
user_df = user_df.drop(['NayoseCd','NayoseFlg','TanPrjCd','KodomoSu','Kod1Nenrei','Kod2Nenrei','Kod3Nenrei',
                        'Seibetu','JitakuJisCD','JitakuEnsen','JitakuEki','JitakuFun','KinmuJisCD',
                        'KinmuEnsen','KinmuEki','TintaiYatin','GassanNensyu','Syokugyo','Gyousyu'],axis=1)
user_df = user_df.drop([user_df.columns[i] for i in range(5,len(user_df.columns))], axis='columns')
user_df.dropna(subset = user_df.columns, inplace=True)
user_df = user_df[(user_df[user_df.columns] != 0).all(axis=1)]
user_df.columns = ['Age', 'Family_members','Desired_price','Yearly salary','Assets']

building_df = pd.read_csv(os.path.join(base, 'buildings.csv'),index_col=0)
building_df = building_df.drop(columns=['rooms','feature','buk_code','pref_code','feature_1_tag','feature_2_tag',
                                       'feature_3_tag','station_1_line_code','station_1_code','station_1_lon',
                                       'station_1_lat','station_2_line_code','station_2_code','station_2_lon',
                                       'station_2_lat','station_3_line_code','station_3_code','station_3_lon',
                                       'station_3_lat','most_floor_plan','most_price','address_1','address_2',
                                       'name','disp_name','prj_code','closed','type','most_width','time_resident',
                                       'lat','lon','created_at','updated_at','deleted_at','updated_by','opened_at',
                                       'closed_at','overview_url','official_website_url','reservation_started_at',
                                       'reservation_ended_at']).drop([53,46,36])

building_df, building_score, not_indexed = process_station_price(building_df, base)

for i in not_indexed:
    reco = reco[reco['building_id']!=i]

X_user = StandardScaler().fit_transform(user_df)
k_means_fit = KMeans(n_clusters=7).fit(X_user)
user_df['cluster'] = k_means_fit.labels_
metrics = ['walking_distance', 'spaciosity','family_amenities','price']

for idx in sorted(np.unique(user_df['cluster'])):
    lp = LearnPreferences(building_score, user_df[user_df['cluster']==idx], metrics)
    lp.run()