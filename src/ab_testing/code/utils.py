import numpy as np
import pandas as pd
import sys
import os
import re
import unicodedata

def create_score(df):
    df_orig = df.copy()

    df['walking_score'] = df['station_no'] / df['station_average_time']
    df['spaciosity_score'] = df['house_surface'] / df['room_no']
    df['family_score'] = df['bedroom_no'] / df['room_no']
    df['price_score'] = (df['room_no'] + df['house_surface']/10 + df['station_no']) / df['price']

    df = normalization(df,'minmax')
    for j in ['walking_score','spaciosity_score','family_score','price_score']:
        df_orig[j]=df[j]
    
    return df, df_orig

def normalization(data,method):
    if method == 'mean':
        # Mean normalization
        data = (data-data.mean())/data.std()
    elif method == 'minmax':
        # Min-max normalization
        data = (data-data.min())/(data.max()-data.min())
    return data

def process_station_price(df, base):
    keyDict = {'station_no','station_average_time','station_min_time','station_max_time'}
    d = dict([(key, []) for key in keyDict])
    
    for index, content in df['access'].iteritems():
        minutes = [int(i) for i in re.findall(r"(\d+)分", content)]
        d['station_no'].append(content.count('分'))
        d['station_average_time'].append(int(np.floor(np.mean(minutes))))
        d['station_min_time'].append(min(minutes))
        d['station_max_time'].append(max(minutes))
    df_station = pd.DataFrame(data=d,columns=list(keyDict),index=df.index)
    df_res = pd.concat([df,df_station],axis=1).drop(columns=['access'])
    
    keyDict = {'house_template','house_no_in_sale','house_surface'}
    d = dict([(key, []) for key in keyDict])
    
    for index, content in df_res['floor_plan'].iteritems():
        content_res = [unicodedata.normalize('NFKC', i) for i in content if i in ['L','D','Ｄ','K','Ｋ','S','Ｓ','F','～','〜','・',' ','、'] or i.isdigit()]
        content_res = ''.join(map(str, content_res)).replace('〜','~').replace('・','~').replace('42 ','').replace(' ','').replace('、','~').replace('4LDK3LDK','4LDK~3LDK')
        d['house_template'].append(content_res.split('~'))
        d['house_no_in_sale'].append(sum(c.isdigit() for c in content))
        d['house_surface'].append([i for i in re.findall(r"[+-]?\d+\.\d+", df_res['width'][index])])
        
    df_h = pd.DataFrame(data=d,columns=list(keyDict),index=df.index)
    price = pd.read_csv(os.path.join(base, 'buildings_price.csv'),index_col=0)
    price['price'] = pd.eval(price['price'].str.split('-', expand = False))

    df_h = pd.concat([df_h,price],axis=1)
    
    df_h['feat0'] = df_h['house_surface'].str.len()
    df_h['feat1'] = df_h['house_template'].str.len()
    df_h['feat2'] = df_h['price'].str.len()
    
    df_h['max_feat'] = df_h[['house_no_in_sale','feat0','feat1','feat2']].max(axis=1)
    
    keyDict = {'id','house_surface','room_no','bedroom_no','price','station_no','station_average_time','station_min_time','station_max_time'}
    d = dict([(key, []) for key in keyDict])
    not_indexed = []
#     print(df_h.index)
    for idx,l in enumerate(df_h['max_feat']):
        if len(np.unique([df_h['feat0'][idx+1],df_h['feat1'][idx+1],df_h['feat2'][idx+1],l]))==1:
#             print('lenbght',int(l))
            for j in range(0,int(l)):
#                 print(df_h.index[idx])
                d['id'].append(df_h.index[idx])
                d['house_surface'].append(float(df_h['house_surface'][idx+1][j]))
                d['room_no'].append(float(len(df_h['house_template'][idx+1][j]) + int(list(filter(str.isdigit, df_h['house_template'][idx+1][j]))[0])-1))
                d['bedroom_no'].append(float(list(filter(str.isdigit, df_h['house_template'][idx+1][j]))[0]))
                d['price'].append(float(price['price'][idx+1][j]))
                
                d['station_no'].append(float(df_station['station_no'][idx+1]))
                d['station_average_time'].append(float(df_station['station_average_time'][idx+1]))
                d['station_min_time'].append(float(df_station['station_min_time'][idx+1]))
                d['station_max_time'].append(float(df_station['station_max_time'][idx+1]))
        else:
            not_indexed.append(df_h.index[idx])
    df_h_res = pd.DataFrame(data=d,columns=list(keyDict)).set_index('id')
    # df_res = pd.concat([df_res,price],axis=1)
    res, res_orig = create_score(df_h_res)
    return res, res_orig, not_indexed




